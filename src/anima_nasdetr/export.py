from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.models.nasdetr import NASDETR


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: NASDETR) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


def export_onnx(path: str, variant: str = "A1") -> None:
    cfg = ModuleConfig.from_variant(PaperVariant(variant))
    model = NASDETR(cfg).eval()
    wrapped = ExportWrapper(model)

    dummy = torch.randn(1, 3, *cfg.data.image_size)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        path,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={"images": {0: "batch"}, "pred_logits": {0: "batch"}, "pred_boxes": {0: "batch"}},
        opset_version=17,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export NAS-DETR to ONNX")
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="A1", choices=["A1", "A2"])
    args = p.parse_args()

    export_onnx(args.out, args.variant)
    print(f"Exported ONNX to {args.out}")


if __name__ == "__main__":
    main()
