from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class URPCDetectionDataset(Dataset):
    """COCO-like adapter for URPC-style sonar datasets."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_json: str | Path,
        image_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_json = Path(annotations_json)
        data = json.loads(self.annotations_json.read_text())

        self.images = {item["id"]: item for item in data.get("images", [])}
        self.anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.ids = sorted(self.images.keys())
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.ids[idx]
        info = self.images[image_id]
        image = Image.open(self.images_dir / info["file_name"]).convert("L").resize(self.image_size[::-1], Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)

        anns = self.anns_by_image.get(image_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))

        sample = {
            "image_id": image_id,
            "image": image_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "size": (info.get("height", image.height), info.get("width", image.width)),
        }
        return sample


def urpc_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {
        "images": images,
        "targets": [
            {"boxes": item["boxes"], "labels": item["labels"], "image_id": item["image_id"], "size": item["size"]}
            for item in batch
        ],
    }
