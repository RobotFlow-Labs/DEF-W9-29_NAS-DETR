# 29_NAS-DETR (DEF-nasdetr)

Paper-faithful ANIMA module scaffold for NAS-DETR (arXiv:2505.06694).

## Included
- PRD suite (`prds/PRD-01..07`)
- Build tasks (`tasks/INDEX.md` + per-task markdown)
- Core local code in `src/anima_nasdetr`
  - NAS search loop (paper Table 3 mutations)
  - CNN-Transformer backbone (paper Table 2-derived A1/A2 variants)
  - DETR-style decoder head and losses (VFL + L1/GIoU + denoising)
  - CLI inference + training skeleton + API stub
- Reference repos:
  - `repositories/rtdetr`
  - `repositories/deformable-detr`

## Quickstart
```bash
uv sync
uv run pytest -q
uv run python -m anima_nasdetr.infer --image ./sample.png --variant A1
```

## Notes
- No official NAS-DETR code repository was identified at generation time.
- This implementation is a reproducible starting point aligned to paper specs and ready for CUDA-side optimization.

