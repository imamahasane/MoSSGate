from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        return nn.GroupNorm(min(32, channels), channels)
    raise ValueError("norm must be 'bn' or 'gn'")

class BoundaryGate(nn.Module):
    def __init__(self, channels: int, dilation: int = 3):
        super().__init__()
        self.dw_r1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.dw_rd = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.dw_r1(x) + self.dw_rd(x)
        g = torch.sigmoid(self.pw(u))
        return g

class ExternalMemoryModulator(nn.Module):
    def __init__(self, channels: int, mem_slots: int = 32):
        super().__init__()
        self.channels = channels
        self.mem_slots = mem_slots

        self.Mk = nn.Parameter(torch.randn(mem_slots, channels) * 0.02)
        self.Mv = nn.Parameter(torch.randn(mem_slots, channels) * 0.02)

        self.to_alpha = nn.Linear(channels, channels, bias=True)
        self.to_beta = nn.Linear(channels, channels, bias=True)
        nn.init.zeros_(self.to_alpha.weight); nn.init.zeros_(self.to_alpha.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

    def forward(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x_norm.mean(dim=(2, 3))  # (B,C)
        att = torch.softmax(z @ self.Mk.t(), dim=-1)  
        c = att @ self.Mv  # (B,C)
        alpha = torch.tanh(self.to_alpha(c))
        beta = torch.tanh(self.to_beta(c))
        return alpha, beta

def ssm_scan_1d(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    B, C, L = u.shape
    a = a.clamp(0.0, 0.999).unsqueeze(-1)  
    y = torch.zeros((B, C, 1), device=u.device, dtype=u.dtype)
    outs = []
    one_minus_a = 1.0 - a
    for t in range(L):
        ut = u[:, :, t:t+1]
        y = a * y + one_minus_a * ut
        outs.append(y)
    return torch.cat(outs, dim=2)

class ParallelSSM2D(nn.Module):
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.channels = channels
        self.groups = groups
        self.c_per = channels // groups

        self.logit_a = nn.Parameter(torch.zeros(groups, self.c_per))
        self.d = nn.Parameter(torch.zeros(groups, self.c_per))

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xs = torch.chunk(x, self.groups, dim=1)
        alphas = torch.chunk(alpha.view(B, C), self.groups, dim=1)
        betas = torch.chunk(beta.view(B, C), self.groups, dim=1)

        outs = []
        for k in range(self.groups):
            u = xs[k]  
            a0 = torch.sigmoid(self.logit_a[k]).unsqueeze(0)  
            ak = (a0 * (1.0 + alphas[k])).clamp(0.0, 0.999)   

            u_lr = u.reshape(B, self.c_per * H, W)
            a_lr = ak.repeat_interleave(H, dim=1)
            y_lr = ssm_scan_1d(u_lr, a_lr).reshape(B, self.c_per, H, W)

            u_rl = torch.flip(u, dims=[3]).reshape(B, self.c_per * H, W)
            y_rl = ssm_scan_1d(u_rl, a_lr).reshape(B, self.c_per, H, W)
            y_rl = torch.flip(y_rl, dims=[3])

            u_tb = u.permute(0, 1, 3, 2).contiguous().reshape(B, self.c_per * W, H)
            a_tb = ak.repeat_interleave(W, dim=1)
            y_tb = ssm_scan_1d(u_tb, a_tb).reshape(B, self.c_per, W, H).permute(0, 1, 3, 2)

            u_bt = torch.flip(u, dims=[2]).permute(0, 1, 3, 2).contiguous().reshape(B, self.c_per * W, H)
            y_bt = ssm_scan_1d(u_bt, a_tb).reshape(B, self.c_per, W, H).permute(0, 1, 3, 2)
            y_bt = torch.flip(y_bt, dims=[2])

            y = (y_lr + y_rl + y_tb + y_bt) / 4.0

            d = self.d[k].unsqueeze(0)  
            skip = (d * (1.0 + betas[k])).view(B, self.c_per, 1, 1) * u
            outs.append(y + skip)

        return torch.cat(outs, dim=1)

class MoSSGateBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int = 4,
        mem_slots: int = 32,
        gate_dilation: int = 3,
        norm: str = "bn",
    ):
        super().__init__()
        self.norm1 = make_norm(norm, channels)
        self.gate = BoundaryGate(channels, dilation=gate_dilation)
        self.mem = ExternalMemoryModulator(channels, mem_slots=mem_slots)
        self.ssm = ParallelSSM2D(channels, groups=groups)

        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = make_norm(norm, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x_bar = self.norm1(x)

        g = self.gate(x_bar)          
        x_tilde = x_bar * g           
        alpha, beta = self.mem(x_bar) 

        o = self.ssm(x_tilde, alpha=alpha, beta=beta)  
        o = self.proj(o)
        o = self.act(self.norm2(o))
        return identity + o
