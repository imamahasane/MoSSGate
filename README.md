# MoSSGate — Memory-Modulated State-Space Gating for Efficient Skin Lesion Segmentation

This repository contains a clean, research-grade PyTorch implementation of **MoSSGate** as described in the provided manuscript: a plug-and-play module for U-Net that combines:
- **Boundary Gate** (boundary-aware spatial gating),
- **External Memory Modulator** (sample-adaptive modulation),
- **Parallel 2D State-Space Modeling** (linear-time long-range context).

The codebase is organized for reproducibility and multi-GPU training (DDP via `torchrun`).

## 1) Repository Structure

```
project_root/
├── configs/
│   └── training_config.yaml
├── models/
│   ├── mossgate_block.py
│   ├── unet_backbone.py
│   └── full_model.py
├── datasets/
│   └── isic_dataset.py
├── training/
│   ├── train.py
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   └── run_inference.py
├── utils/
│   ├── augmentations.py
│   ├── logging.py
│   └── visualization.py
├── scripts/
│   ├── train.sh
│   └── inference.sh
├── requirements.txt
└── README.md
```

## 2) Dataset Layout (Expected)

Set `data.root` in `configs/training_config.yaml` to a directory like:

```
ISIC_ROOT/
├── images/
│   ├── ISIC_0000000.jpg
│   └── ...
├── masks/
│   ├── ISIC_0000000.png
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each split file should contain one **image id per line** (without extension), e.g. `ISIC_0000000`.

## 3) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Training

### Single GPU
```bash
python -m training.train --config configs/training_config.yaml --out runs/exp1
```

### Multi-GPU (DDP)
```bash
torchrun --standalone --nproc_per_node=2 -m training.train --config configs/training_config.yaml --out runs/exp1
```

Checkpoints:
- `runs/exp1/last.pt`
- `runs/exp1/best.pt`

TensorBoard:
```bash
tensorboard --logdir runs/exp1/tb
```

## 5) Evaluation / Inference

```bash
python -m evaluation.run_inference --ckpt runs/exp1/best.pt --config runs/exp1/config_resolved.yaml --split test --out runs/exp1/infer_test
```

To save overlay images:
```bash
python -m evaluation.run_inference --ckpt runs/exp1/best.pt --config runs/exp1/config_resolved.yaml --split test --out runs/exp1/infer_test --save_vis
```

Outputs:
- `runs/exp1/infer_test/metrics.json`
- optional `*_overlay.png` images

## 6) Notes

- The MoSSGate block follows Algorithm 1 in the manuscript:
  - normalization -> boundary gate -> memory modulation -> 4-direction 2D state-space scans -> 1x1 projection + residual.
- State-space recurrence is implemented with a stable explicit loop. At deep U-Net stages (e.g., 22x22 for 352 input at bottleneck), the scan length is small.

