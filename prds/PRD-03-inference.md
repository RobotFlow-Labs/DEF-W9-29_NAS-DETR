# PRD-03: Inference Pipeline

> Module: DEF-nasdetr | Priority: P0
> Depends on: PRD-02
> Status: ✅ Implemented (local baseline)

## Objective
Provide a runnable local inference pipeline for single image and batch workflows.

## Context (from paper)
NAS-DETR is designed for practical detection with strong latency-performance tradeoff and query-driven decoding.
Paper reference: §3.3, §4.5.

## Acceptance Criteria
- [x] CLI inference accepts image path and A1/A2 variant.
- [x] Outputs class logits and normalized boxes.
- [x] Includes light preprocessing for sonar-like grayscale images.
- [x] Emits deterministic JSON-style output for downstream integration.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_nasdetr/infer.py` | CLI inference entrypoint | §3.3, §4.5 | ~170 |
| `src/anima_nasdetr/utils/boxes.py` | Box conversion/clamp utilities | Eq. (30) | ~70 |

## Test Plan
```bash
uv run python -m anima_nasdetr.infer --help
```

## References
- Paper §4.5 Computational efficiency analysis
