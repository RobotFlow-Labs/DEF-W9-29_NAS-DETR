# NAS-DETR Task Index

## Build Order
| Task | Title | Depends | Status |
|---|---|---|---|
| PRD-0101 | Project packaging + config dataclasses | None | DONE |
| PRD-0102 | URPC dataset adapter + collate | PRD-0101 | DONE |
| PRD-0201 | ResBlock + TransformerBlock primitives | PRD-0101 | DONE |
| PRD-0202 | CNN-Transformer backbone (A1/A2) | PRD-0201 | DONE |
| PRD-0203 | Query selection + decoder head | PRD-0202 | DONE |
| PRD-0204 | NAS search + entropy scoring | PRD-0202 | DONE |
| PRD-0205 | Loss stack (VFL + box + dn) | PRD-0203 | DONE |
| PRD-0206 | Integrated NAS-DETR model wrapper | PRD-0203, PRD-0205 | DONE |
| PRD-0301 | CLI inference entrypoint | PRD-0206 | DONE |
| PRD-0401 | Evaluation runner + metric report skeleton | PRD-0301 | DONE |
| PRD-0501 | FastAPI service endpoints | PRD-0301 | DONE |
| PRD-0502 | Docker serving files | PRD-0501 | DONE |
| PRD-0601 | ROS2 node scaffold | PRD-0501 | DONE |
| PRD-0701 | ONNX export utility | PRD-0206 | DONE |
| PRD-0702 | Production checklist docs | PRD-0401, PRD-0701 | DONE |
| PRD-0207 | Unit tests (model/backbone/search/config) | PRD-0206 | DONE |

## Notes
- Marked complete for local scaffold implementation.
- CUDA-optimized kernels and TRT benchmarking remain server-side next steps.
