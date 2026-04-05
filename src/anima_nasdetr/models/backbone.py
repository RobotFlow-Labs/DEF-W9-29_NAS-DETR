from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from anima_nasdetr.config import ModelConfig, StageSpec
from .blocks import ResBlock, TransformerStage


class CnnTransformerBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stages = nn.ModuleDict()

        for spec in cfg.stages:
            self.stages[spec.name] = self._make_stage(spec)

    @staticmethod
    def _make_stage(spec: StageSpec) -> nn.Module:
        if spec.block == "transformer":
            return TransformerStage(
                in_ch=spec.in_channels,
                out_ch=spec.out_channels,
                hidden_dim=max(spec.hidden_dim, spec.out_channels),
                ffn_dim=max(spec.ffn_dim, spec.out_channels * 2),
                layers=max(1, spec.layers),
            )

        blocks = []
        in_ch = spec.in_channels
        for i in range(max(1, spec.layers)):
            stride = spec.stride if i == 0 else 1
            out_ch = spec.out_channels
            pool = spec.block == "res_pool" and i == spec.layers - 1
            blocks.append(
                ResBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel=spec.kernel,
                    stride=stride,
                    bottleneck=max(8, spec.bottleneck),
                    pool=pool,
                )
            )
            in_ch = out_ch
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        cur = x
        for spec in self.cfg.stages:
            cur = self.stages[spec.name](cur)
            out[spec.name] = cur
        return out


def estimate_backbone_gflops(_model: nn.Module, h: int, w: int) -> float:
    # Lightweight placeholder FLOPs estimate for search filtering.
    return float((h * w) / 1e6)


def flatten_multiscale(
    features: dict[str, torch.Tensor],
    target_dim: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    seqs = []
    shapes = []
    for _, feat in features.items():
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        if c < target_dim:
            pad = torch.zeros((b, tokens.shape[1], target_dim - c), device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, pad], dim=-1)
        elif c > target_dim:
            tokens = tokens[..., :target_dim]
        seqs.append(tokens)
        shapes.append((h, w))
    return torch.cat(seqs, dim=1), shapes
