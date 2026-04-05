from __future__ import annotations

import math

import torch


def differential_entropy_per_channel(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # feat: [B, C, H, W]
    var = feat.var(dim=(0, 2, 3), unbiased=False).clamp_min(eps)
    return 0.5 * torch.log(2 * math.pi * math.e * var)


def weighted_entropy_score(features: dict[str, torch.Tensor], weights: tuple[int, int, int, int, int, int]) -> float:
    keys = ["C1", "C2", "C3", "C4", "C5", "C6"]
    score = 0.0
    denom = max(1, sum(abs(w) for w in weights))
    for i, key in enumerate(keys):
        if key not in features:
            continue
        h = differential_entropy_per_channel(features[key]).mean().item()
        score += weights[i] * h
    return score / denom
