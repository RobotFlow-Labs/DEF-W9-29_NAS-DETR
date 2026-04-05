from __future__ import annotations

import torch
import torch.nn as nn


class DeformableLikeDecoder(nn.Module):
    """A lightweight decoder approximating multi-scale deformable behavior."""

    def __init__(self, d_model: int, num_layers: int, num_classes: int) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=max(256, d_model * 4),
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4), nn.Sigmoid())

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        x = queries
        for layer in self.layers:
            x = layer(tgt=x, memory=memory)

        logits = self.cls_head(x)
        boxes = self.box_head(x)
        return {"pred_logits": logits, "pred_boxes": boxes, "decoder_tokens": x}
