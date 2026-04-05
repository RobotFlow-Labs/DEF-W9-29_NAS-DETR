from __future__ import annotations

import torch
import torch.nn as nn


class QuerySelector(nn.Module):
    def __init__(self, d_model: int, num_queries: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, N, C]
        scores = self.score(tokens).squeeze(-1)
        k = min(self.num_queries, tokens.shape[1])
        topk = scores.topk(k, dim=1).indices
        selected = torch.gather(tokens, 1, topk.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        return selected, topk
