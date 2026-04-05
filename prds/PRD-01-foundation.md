# PRD-01: Foundation and Project Scaffolding

> Module: DEF-nasdetr | Priority: P0
> Depends on: None
> Status: ✅ Implemented

## Objective
Create a reproducible, paper-aligned project skeleton with configs, data contracts, and baseline validation tests.

## Context (from paper)
NAS-DETR requires a fixed A1/A2 search weighting setup, URPC2021/2022 dataset splits, and a training stack grounded on PyTorch and MMDetection-like data flow.
Paper reference: §4.1.1, §4.1.2, §4.1.3.

## Acceptance Criteria
- [x] `pyproject.toml` and package structure exist and are importable.
- [x] Configs include default, debug, and paper variants.
- [x] Dataset loader accepts URPC-style image/annotation layout.
- [x] `ASSETS.md` captures datasets, weights, hyperparameters, targets.
- [x] Unit tests for config and import path pass.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `pyproject.toml` | Runtime + packaging | §4.1.2 | ~45 |
| `src/anima_nasdetr/config.py` | Paper config dataclasses | Table 2/3, §4.1 | ~180 |
| `src/anima_nasdetr/data/urpc.py` | Dataset + collate | §4.1.1 | ~120 |
| `configs/default.toml` | Default training/search setup | §4.1 | ~40 |
| `configs/paper.toml` | Paper-faithful setup | §4.1 | ~35 |
| `configs/debug.toml` | Fast local debug | engineering | ~25 |

## Test Plan
```bash
uv run pytest tests/test_config.py -q
```

## References
- Paper §4.1.1 Sonar Image Dataset
- Paper §4.1.2 Experiment settings
- Paper §4.1.3 NAS
