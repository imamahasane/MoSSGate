# MoSSGate: Memory-Modulated State-Space Gating for Efficient Skin Lesion Segmentation

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)

Official implementation of the paper:  
> **“MoSSGate: Memory-Modulated State-Space Gating for Efficient Skin Lesion Segmentation”**  
> *Anum Malik, A. F. M Abdun Noor, Md Imam Ahasan, Mahnoor Buriro*
> *International Joint Conference on Neural Networks (IJCNN), 2026, Accepted*

---

## 1. Project Description

**MoSSGate** is a plug-and-play module for **medical image segmentation** (skin lesion segmentation on ISIC benchmarks), designed to improve boundary sensitivity and global context modeling while remaining efficient.

**Core features:**
- **Boundary Gate**: boundary-aware spatial gating via depthwise dilated convolutions.
- **External Memory Modulator**: sample-adaptive channel modulation using learnable memory keys/values.
- **Parallel 2D State-Space Modeling (SSM)**: efficient long-range context via 4-direction scans (→, ←, ↓, ↑).
- **Full U-Net Integration**: insert MoSSGate at deep encoder/bottleneck stages (e.g., `enc4`, optionally `enc3`).
- Training features: **AMP (mixed precision)**, **multi-GPU DDP**, config-based experiments, reproducible pipeline.

---
## 2. Dataset Information

This project supports ISIC-style lesion segmentation datasets (**ISIC 2017, ISIC 2018**).  
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
Each split file should contain one **image id per line** (without extension),  `ISIC_0000000`.

---

## 3. Repository Structure

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

## 4. Installation

Option A (recommended: venv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Option B (existing env):
```bash
pip install -r requirements.txt
```
---

## 5. Dataset Preparation
**Download ISIC images + masks and arrange into:**
```bash
ISIC_ROOT/images
ISIC_ROOT/masks
```
**Create split files:**
```bash
ISIC_ROOT/splits/train.txt
ISIC_ROOT/splits/val.txt
ISIC_ROOT/splits/test.txt
```
**Edit the config:**
> Open
```bash
configs/training_config.yaml
```
> Set
```bash
data.root: /path/to/ISIC_ROOT
```
---

## 6. Training

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
---
## 7. Requirements
```bash
torch>=2.1
torchvision>=0.16
numpy
opencv-python
Pillow
PyYAML
tqdm
albumentations>=1.4.0
tensorboard
matplotlib
```
---


> MoSSGate combines boundary-aware gating, external memory modulation, and parallel 2D state-space modeling for efficient and accurate skin lesion segmentation.
