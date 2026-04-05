import torch

from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.models.nasdetr import NASDETR


def test_model_forward_shapes() -> None:
    cfg = ModuleConfig.from_variant(PaperVariant.A1)
    cfg.model.num_queries = 64
    cfg.model.decoder_layers = 2
    model = NASDETR(cfg).eval()

    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 256))

    assert out["pred_logits"].shape[:2] == (2, 64)
    assert out["pred_boxes"].shape == (2, 64, 4)
