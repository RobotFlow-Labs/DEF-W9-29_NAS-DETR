import torch

from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.losses import detr_loss
from anima_nasdetr.models.nasdetr import NASDETR


def test_loss_is_finite() -> None:
    cfg = ModuleConfig.from_variant(PaperVariant.A1)
    cfg.model.num_queries = 32
    cfg.model.decoder_layers = 2
    model = NASDETR(cfg).eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))

    targets = [{"boxes": torch.rand(8, 4), "labels": torch.randint(0, cfg.data.num_classes, (8,)), "image_id": 0}]
    losses = detr_loss(out, targets, cfg.losses)
    assert torch.isfinite(losses["loss"]).item()
