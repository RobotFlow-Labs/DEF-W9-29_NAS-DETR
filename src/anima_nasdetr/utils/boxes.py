from __future__ import annotations

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def clamp_boxes_xyxy(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[..., 0].clamp_(0, w - 1)
    boxes[..., 2].clamp_(0, w - 1)
    boxes[..., 1].clamp_(0, h - 1)
    boxes[..., 3].clamp_(0, h - 1)
    return boxes
