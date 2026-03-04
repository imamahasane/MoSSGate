from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.isic_dataset import ISICDataConfig, ISICDataset
from models.full_model import build_model
from evaluation.metrics import compute_metrics_from_logits
from utils.visualization import save_overlay

def parse_args():
    p = argparse.ArgumentParser("MoSSGate inference/evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, required=True, help="Resolved or base config yaml")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, required=True, help="Output directory for predictions/visualizations")
    p.add_argument("--save_vis", action="store_true", help="Save overlay images for a few samples")
    p.add_argument("--max_vis", type=int, default=20)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ds = ISICDataset(data_cfg, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(dc.get("num_workers", 4)))

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(cfg.get("inference", {}).get("threshold", 0.5))

    dice_sum=iou_sum=acc_sum=0.0
    n=0
    vis_saved=0

    for img, mask, _id in loader:
        img = img.to(device)
        mask = mask.to(device)
        logits = model(img)
        m = compute_metrics_from_logits(logits, mask, thr=thr)
        dice_sum += m.dice; iou_sum += m.iou; acc_sum += m.acc
        n += 1

        if args.save_vis and vis_saved < args.max_vis:
            img_np = img[0].detach().cpu().permute(1,2,0).numpy()
            gt_np = mask[0,0].detach().cpu().numpy()
            pr_np = (torch.sigmoid(logits)[0,0].detach().cpu().numpy() > thr).astype("float32")
            save_overlay(img_np, gt_np, pr_np, out_dir / f"{_id[0]}_overlay.png", title=_id[0])
            vis_saved += 1

    n = max(n,1)
    metrics = {"dice": dice_sum/n, "iou": iou_sum/n, "acc": acc_sum/n, "n": n}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    print(metrics)

if __name__ == "__main__":
    main()
