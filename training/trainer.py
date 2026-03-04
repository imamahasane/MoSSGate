from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.metrics import compute_metrics_from_logits
from utils.logging import setup_logger

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_dice: float = -1.0

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def ddp_barrier():
    if is_distributed():
        dist.barrier()

def save_checkpoint(path: Path, model: nn.Module, optimizer, scaler, state: TrainState, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    payload = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "state": state.__dict__,
        "cfg": cfg,
    }
    torch.save(payload, path)

def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None) -> Tuple[nn.Module, Optional[object], Optional[object], TrainState, Dict]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    state = TrainState(**ckpt["state"])
    cfg = ckpt.get("cfg", {})
    return model, optimizer, scaler, state, cfg

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    dice_sum = iou_sum = acc_sum = 0.0
    n = 0
    for batch in loader:
        img, mask, _ = batch
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        m = compute_metrics_from_logits(logits, mask, thr=thr)
        dice_sum += m.dice; iou_sum += m.iou; acc_sum += m.acc
        n += 1

    t = torch.tensor([dice_sum, iou_sum, acc_sum, float(n)], device=device)
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dice_sum, iou_sum, acc_sum, n = t.tolist()
    n = max(int(n), 1)
    return {"dice": dice_sum / n, "iou": iou_sum / n, "acc": acc_sum / n}

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict,
    out_dir: str | os.PathLike,
    resume: Optional[str] = None,
) -> Dict[str, float]:
    rank = get_rank()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir, rank=rank)

    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if rank == 0 else None
    amp = bool(cfg["train"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    state = TrainState()
    if resume:
        logger.info(f"Resuming from {resume}")
        model, optimizer, scaler, state, _ = load_checkpoint(Path(resume), model, optimizer, scaler)

    if is_distributed():
        model = DDP(model.to(device), device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)
    else:
        model = model.to(device)

    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    log_every = int(cfg["train"].get("log_every", 50))
    eval_every = int(cfg["train"].get("eval_every", 1))
    save_every = int(cfg["train"].get("save_every", 1))
    thr = float(cfg.get("inference", {}).get("threshold", 0.5))

    for epoch in range(state.epoch, epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, disable=(rank != 0))
        for it, batch in enumerate(pbar):
            img, mask, _ = batch
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(img)
                loss = loss_fn(logits, mask)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            state.global_step += 1

            if rank == 0 and (it + 1) % log_every == 0:
                pbar.set_description(f"epoch {epoch+1}/{epochs} loss {running/(it+1):.4f}")

        train_loss = running / max(len(train_loader), 1)

        val_metrics = {"dice": 0.0, "iou": 0.0, "acc": 0.0}
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, thr=thr)

        if rank == 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, TrainState(epoch=epoch+1, global_step=state.global_step, best_val_dice=state.best_val_dice), cfg)
            if val_metrics["dice"] > state.best_val_dice:
                state.best_val_dice = val_metrics["dice"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, TrainState(epoch=epoch+1, global_step=state.global_step, best_val_dice=state.best_val_dice), cfg)

        dt = time.time() - t0
        if rank == 0:
            logger.info(
                f"[epoch {epoch+1:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} val_dice={val_metrics['dice']:.4f} "
                f"val_iou={val_metrics['iou']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"best={state.best_val_dice:.4f} time={dt:.1f}s"
            )
            if writer:
                writer.add_scalar("loss/train", train_loss, epoch+1)
                writer.add_scalar("dice/val", val_metrics["dice"], epoch+1)
                writer.add_scalar("iou/val", val_metrics["iou"], epoch+1)
                writer.add_scalar("acc/val", val_metrics["acc"], epoch+1)

        ddp_barrier()

    if writer:
        writer.close()
    return {"best_val_dice": float(state.best_val_dice)}
