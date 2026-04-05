# PRD-02: Core NAS-DETR Model

> Module: DEF-nasdetr | Priority: P0
> Depends on: PRD-01
> Status: ✅ Implemented (initial)

## Objective
Implement a working NAS-DETR core with CNN-Transformer backbone, query selection, decoder, and multi-loss training objective.

## Context (from paper)
The model combines a NAS-optimized CNN-Transformer backbone with deformable-attention decoder and denoising-augmented optimization.
Paper reference: §3.1, §3.2, §3.3, §3.4.

## Acceptance Criteria
- [x] Backbone supports A1/A2 architecture variants from Table 2.
- [x] NAS search loop implements mutation rules from Table 3.
- [x] Decoder consumes selected queries and predicts class + box.
- [x] Loss includes VFL + L1/GIoU + denoising terms.
- [x] End-to-end forward pass runs on synthetic tensors.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_nasdetr/models/blocks.py` | ResBlock and Transformer blocks | Table 2 | ~140 |
| `src/anima_nasdetr/models/backbone.py` | C1..C6 feature extraction | Table 2 | ~200 |
| `src/anima_nasdetr/models/query.py` | Top-k query selection | §3.3 | ~60 |
| `src/anima_nasdetr/models/decoder.py` | Deformable-like decoder head | §3.3 Eq. (30) | ~150 |
| `src/anima_nasdetr/models/nasdetr.py` | Full model composition | §3 | ~140 |
| `src/anima_nasdetr/losses.py` | Eq. (31)-(34) losses | §3.4 | ~170 |
| `src/anima_nasdetr/nas/search.py` | Evolutionary NAS loop | §3.2, Table 3 | ~210 |
| `src/anima_nasdetr/nas/entropy.py` | Entropy score helpers | Eq. (29) | ~80 |

## Test Plan
```bash
uv run pytest tests/test_backbone.py tests/test_model_forward.py tests/test_search.py -q
```

## References
- Paper Table 2 (architecture)
- Paper Table 3 (mutation)
- Eq. (29), Eq. (31)-(34)
