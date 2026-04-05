from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.models.nasdetr import NASDETR


def _load_image(path: str, image_size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("L").resize(image_size[::-1], Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    return x


def run_infer(image: str, variant: str = "A1", num_queries: int | None = None) -> dict:
    pv = PaperVariant(variant)
    cfg = ModuleConfig.from_variant(pv)
    if num_queries is not None:
        cfg.model.num_queries = num_queries

    model = NASDETR(cfg).eval()
    x = _load_image(image, cfg.data.image_size)

    with torch.no_grad():
        out = model(x)

    probs = out["pred_logits"].sigmoid()[0]
    boxes = out["pred_boxes"][0]

    scores, labels = probs.max(dim=-1)
    topk = min(20, scores.numel())
    idx = scores.topk(topk).indices

    preds = []
    for i in idx.tolist():
        preds.append(
            {
                "query": int(i),
                "score": float(scores[i].item()),
                "label": int(labels[i].item()),
                "box_cxcywh": [float(v) for v in boxes[i].tolist()],
            }
        )

    return {
        "variant": variant,
        "num_predictions": len(preds),
        "predictions": preds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS-DETR local inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--variant", type=str, default="A1", choices=["A1", "A2"])
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_infer(args.image, args.variant, args.num_queries)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
