import torch

from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.models.backbone import CnnTransformerBackbone


def test_backbone_emits_multiscale_features() -> None:
    cfg = ModuleConfig.from_variant(PaperVariant.A1)
    model = CnnTransformerBackbone(cfg.model).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert set(out.keys()) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert out["C6"].shape[1] == 256
