# PRD-05: API and Docker Serving

> Module: DEF-nasdetr | Priority: P1
> Depends on: PRD-03
> Status: ✅ Implemented (minimal)

## Objective
Expose model inference behind stable HTTP API and container entrypoint.

## Context (from paper)
The method targets practical deployment and high throughput, including TensorRT acceleration in production.
Paper reference: §4.5.

## Acceptance Criteria
- [x] FastAPI app exposes `/health` and `/predict`.
- [x] Docker serving file exists for local reproducibility.
- [x] API request/response schema is deterministic.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_nasdetr/api.py` | FastAPI inference service | §4.5 | ~130 |
| `docker/Dockerfile.serve` | Containerized inference runtime | deployment | ~40 |
| `docker/docker-compose.serve.yml` | Local service startup | deployment | ~30 |

