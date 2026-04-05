# DEF-nasdetr

## Module Identity
- Module ID: DEF-nasdetr
- Research paper: arXiv:2505.06694v1
- Focus: Sonar object detection with NAS-optimized DETR backbone

## What Exists In This Repo
- `papers/2505.06694.pdf`: source paper
- `repositories/rtdetr`: upstream RT-DETR reference
- `repositories/deformable-detr`: upstream Deformable-DETR reference
- `ASSETS.md`: paper-linked data/weights/metrics manifest
- `prds/`: 7-PRD implementation suite
- `tasks/`: granular execution tasks
- `src/anima_nasdetr`: runnable core implementation

## Local Quickstart
```bash
uv sync
uv run pytest -q
uv run python -m anima_nasdetr.infer --image path/to/image.png --variant A1
```

## CUDA Server Migration Notes
- Keep functional parity with local package interfaces.
- Swap CPU-safe operators with CUDA-optimized ops behind same API.
- Preserve config compatibility (`configs/paper.toml`) and loss definitions.
- Export path targets in PRD-07 should map to ANIMA server conventions.

