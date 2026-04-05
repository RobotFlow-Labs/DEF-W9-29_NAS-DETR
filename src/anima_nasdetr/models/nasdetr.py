from __future__ import annotations

import torch
import torch.nn as nn

from anima_nasdetr.config import ModuleConfig
from .backbone import CnnTransformerBackbone, flatten_multiscale
from .decoder import DeformableLikeDecoder
from .query import QuerySelector


class NASDETR(nn.Module):
    def __init__(self, cfg: ModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CnnTransformerBackbone(cfg.model)
        d_model = cfg.model.stages[-1].out_channels
        self.query_selector = QuerySelector(d_model=d_model, num_queries=cfg.model.num_queries)
        self.decoder = DeformableLikeDecoder(
            d_model=d_model,
            num_layers=cfg.model.decoder_layers,
            num_classes=cfg.data.num_classes,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(images)
        memory, _ = flatten_multiscale(feats, target_dim=self.cfg.model.stages[-1].out_channels)
        queries, query_idx = self.query_selector(memory)
        out = self.decoder(queries=queries, memory=memory)
        out["query_idx"] = query_idx
        out["features"] = feats
        return out
