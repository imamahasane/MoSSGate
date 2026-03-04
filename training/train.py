from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from datasets.isic_dataset import ISICDataConfig, ISICDataset
from models.full_model import build_model
from training.losses import DiceBCELoss
from training.trainer import train_loop, is_distributed
from utils.logging import setup_logger

def parse_args():
    p = argparse.ArgumentParser("MoSSGate training")
    p.add_argument("--config", type=str, required=True, help="Path to configs/training_config.yaml")
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return p.parse_args()

def init_distributed():
    # torchrun sets these env vars
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return True
    return False

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    distributed = init_distributed()
    rank = dist.get_rank() if distributed else 0

    # device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger = setup_logger(args.out, rank=rank)
    if rank == 0:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out) / "config_resolved.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # data
    dc = cfg["data"]
    data_cfg = ISICDataConfig(
        root=str(dc["root"]),
        image_dir=str(dc.get("image_dir", "images")),
        mask_dir=str(dc.get("mask_dir", "masks")),
        train_split=str(dc.get("train_split", "splits/train.txt")),
        val_split=str(dc.get("val_split", "splits/val.txt")),
        test_split=str(dc.get("test_split", "splits/test.txt")),
        img_size=int(dc.get("img_size", 352)),
    )
    train_set = ISICDataset(data_cfg, split="train")
    val_set = ISICDataset(data_cfg, split="val")

    batch_size = int(cfg["train"].get("batch_size", 8))
    num_workers = int(dc.get("num_workers", 8))
    pin_memory = bool(dc.get("pin_memory", True))

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(cfg)
    loss_cfg = cfg.get("loss", {})
    loss_fn = DiceBCELoss(
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_smooth=float(loss_cfg.get("dice_smooth", 1e-6)),
    )
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-5)),
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        out_dir=args.out,
        resume=args.resume,
    )

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
