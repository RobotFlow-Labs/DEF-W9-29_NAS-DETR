# PRD-07: Production Hardening and Export

> Module: DEF-nasdetr | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⏳ Scaffolded

## Objective
Prepare export, observability, and failure handling for CUDA-server deployment.

## Context (from paper)
Paper reports strong FPS and major TensorRT speedups, implying export/runtime paths are critical.
Paper reference: §4.5.

## Acceptance Criteria
- [x] ONNX export utility exists.
- [x] Production checklist exists.
- [ ] TensorRT FP16/FP32 export validated on server.
- [ ] Runtime monitoring and alerting integrated.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_nasdetr/export.py` | ONNX/TRT export entry | §4.5 | ~110 |
| `docs/production_checklist.md` | Deployment gate list | §4.5 | ~60 |

