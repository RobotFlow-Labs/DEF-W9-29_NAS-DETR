# PRD-04: Evaluation and Benchmarking

> Module: DEF-nasdetr | Priority: P1
> Depends on: PRD-03
> Status: ✅ Implemented (scaffold)

## Objective
Create repeatable evaluation tooling for URPC2021/URPC2022 and paper-metric comparison.

## Context (from paper)
Reported key metrics are mmAP, mAP50, and mAP75 across URPC2021/2022 with A1/A2 settings.
Paper reference: §4.2, §4.3, Table 4, Table 6.

## Acceptance Criteria
- [x] Evaluation runner can execute model on validation split.
- [x] Metrics report includes mmAP/mAP50/mAP75 placeholders.
- [x] Paper baseline table is included for side-by-side comparison.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_nasdetr/eval.py` | Evaluation runner and report | §4.2, §4.3 | ~140 |
| `benchmarks/paper_metrics.md` | Paper vs run metric sheet | Table 4/6 | ~60 |

## References
- URPC2021: mmAP 0.538 (A1)
- URPC2022: mmAP 0.492 (A2)
