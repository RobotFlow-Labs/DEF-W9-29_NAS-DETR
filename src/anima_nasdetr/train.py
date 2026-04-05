from __future__ import annotations

import argparse

import torch
from torch import optim

from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.losses import detr_loss
from anima_nasdetr.models.nasdetr import NASDETR


def synthetic_batch(batch: int, h: int, w: int, num_classes: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.randn(batch, 3, h, w)
    targets = []
    for i in range(batch):
        n = 8
        boxes = torch.rand(n, 4)
        labels = torch.randint(0, num_classes, (n,))
        targets.append({"boxes": boxes, "labels": labels, "image_id": i})
    return images, targets


def train_debug(variant: str = "A1", steps: int = 10) -> None:
    cfg = ModuleConfig.from_variant(PaperVariant(variant))
    model = NASDETR(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for step in range(steps):
        images, targets = synthetic_batch(2, *cfg.data.image_size, cfg.data.num_classes)
        out = model(images)
        losses = detr_loss(out, targets, cfg.losses)

        opt.zero_grad()
        losses["loss"].backward()
        opt.step()

        print(f"step={step:03d} loss={losses['loss'].item():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="NAS-DETR debug trainer")
    p.add_argument("--variant", choices=["A1", "A2"], default="A1")
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()
    train_debug(args.variant, args.steps)


if __name__ == "__main__":
    main()
