from __future__ import annotations
import torch.nn as nn
from models.unet_backbone import UNetEncoderDecoder

def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    return UNetEncoderDecoder(
        in_channels=int(m.get("in_channels", 3)),
        out_channels=int(m.get("out_channels", 1)),
        base_channels=int(m.get("base_channels", 64)),
        bilinear=bool(m.get("bilinear", True)),
        norm=str(m.get("norm", "bn")),
        moss_stages=list(m.get("moss_stages", ["enc4"])),
        moss_groups=int(m.get("moss_groups", 4)),
        moss_mem_slots=int(m.get("moss_mem_slots", 32)),
        moss_gate_dilation=int(m.get("moss_gate_dilation", 3)),
    )
