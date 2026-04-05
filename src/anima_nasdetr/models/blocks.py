from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, bottleneck: int, pool: bool = False) -> None:
        super().__init__()
        pad = kernel // 2
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_ch)

        self.conv1 = nn.Conv2d(in_ch, bottleneck, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv3 = nn.Conv2d(bottleneck, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.bn_proj(self.proj(x))
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.act(out + identity)
        return self.pool(out)


class TransformerStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_dim: int, ffn_dim: int, layers: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.in_proj = nn.Linear(out_ch, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = self.in_proj(seq)
        seq = self.encoder(seq)
        seq = self.out_proj(seq)
        return seq.transpose(1, 2).reshape(b, c, h, w)
