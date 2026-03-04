from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mossgate_block import MoSSGateBlock, make_norm

class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = make_norm(norm, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_ch, out_ch, norm=norm),
            ConvNormAct(out_ch, out_ch, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, norm: str = "bn"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.reduce = nn.Identity()

        self.conv = DoubleConv(out_ch + out_ch, out_ch, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
        norm: str = "bn",
        moss_stages: Optional[List[str]] = None,
        moss_groups: int = 4,
        moss_mem_slots: int = 32,
        moss_gate_dilation: int = 3,
    ):
        super().__init__()
        if moss_stages is None:
            moss_stages = ["enc4"]

        ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

        self.inc = DoubleConv(in_channels, ch[0], norm=norm)
        self.down1 = Down(ch[0], ch[1], norm=norm)
        self.down2 = Down(ch[1], ch[2], norm=norm)
        self.down3 = Down(ch[2], ch[3], norm=norm)
        self.down4 = Down(ch[3], ch[4], norm=norm)

        self.moss_enc3 = MoSSGateBlock(ch[3], groups=moss_groups, mem_slots=moss_mem_slots,
                                       gate_dilation=moss_gate_dilation, norm=norm) if "enc3" in moss_stages else nn.Identity()
        self.moss_enc4 = MoSSGateBlock(ch[4], groups=moss_groups, mem_slots=moss_mem_slots,
                                       gate_dilation=moss_gate_dilation, norm=norm) if "enc4" in moss_stages else nn.Identity()

        self.up1 = Up(ch[4], ch[3], bilinear=bilinear, norm=norm)
        self.up2 = Up(ch[3], ch[2], bilinear=bilinear, norm=norm)
        self.up3 = Up(ch[2], ch[1], bilinear=bilinear, norm=norm)
        self.up4 = Up(ch[1], ch[0], bilinear=bilinear, norm=norm)
        self.outc = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.moss_enc3(x4)
        x5 = self.down4(x4)
        x5 = self.moss_enc4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
