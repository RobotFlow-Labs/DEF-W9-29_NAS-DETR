from __future__ import annotations

import torch
import torch.nn.functional as F

from anima_nasdetr.config import LossConfig
from anima_nasdetr.utils.boxes import cxcywh_to_xyxy


def varifocal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    probs = logits.sigmoid()
    pos = targets * (1 - probs).pow(gamma) * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    neg = (1 - targets) * probs.pow(gamma) * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (pos + neg).mean()


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes in xyxy
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    a1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    a2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    union = a1 + a2 - inter + 1e-6
    return inter / union


def giou_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pred_xyxy = cxcywh_to_xyxy(pred)
    tgt_xyxy = cxcywh_to_xyxy(tgt)
    iou = pairwise_iou(pred_xyxy, tgt_xyxy)

    # diagonal matching assumption for this scaffold
    diag = torch.diag(iou) if iou.numel() else torch.zeros((), device=pred.device)
    return (1.0 - diag).mean() if diag.numel() else torch.tensor(0.0, device=pred.device)


def detr_loss(outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], cfg: LossConfig) -> dict[str, torch.Tensor]:
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    b, q, c = logits.shape

    cls_targets = torch.zeros((b, q, c), device=logits.device)
    box_losses = []
    dn_losses = []

    for i, t in enumerate(targets):
        labels = t["labels"].to(logits.device)
        tgt_boxes = t["boxes"].to(logits.device)
        n = min(q, labels.numel())
        if n > 0:
            cls_targets[i, torch.arange(n), labels[:n].clamp(max=c - 1)] = 1.0
            pred_boxes_i = boxes[i, :n]
            # Convert xyxy->cxcywh normalization fallback assumption.
            xyxy = tgt_boxes[:n]
            wh = (xyxy[:, 2:] - xyxy[:, :2]).clamp(min=0)
            cxcy = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
            tgt_cxcywh = torch.cat([cxcy, wh], dim=-1)
            box_l1 = F.l1_loss(pred_boxes_i, tgt_cxcywh, reduction="mean")
            box_giou = giou_loss(pred_boxes_i, tgt_cxcywh)
            box_losses.append(cfg.lambda_l1 * box_l1 + cfg.lambda_giou * box_giou)

            noise = torch.randn_like(tgt_cxcywh) * cfg.denoise_sigma
            dn_losses.append(F.mse_loss(pred_boxes_i, tgt_cxcywh + noise, reduction="mean"))

    cls = varifocal_loss(logits, cls_targets, gamma=cfg.gamma)
    box = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0, device=logits.device)
    dn = torch.stack(dn_losses).mean() if dn_losses else torch.tensor(0.0, device=logits.device)

    total = cfg.lambda_cls * cls + cfg.lambda_box * box + cfg.lambda_dn * dn
    return {"loss": total, "loss_cls": cls, "loss_box": box, "loss_dn": dn}
