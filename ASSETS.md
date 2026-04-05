# NAS-DETR — Asset Manifest

## Paper
- Title: Underwater object detection in sonar imagery with detection transformer and Zero-shot neural architecture search
- ArXiv: 2505.06694v1
- Authors: XiaoTong Gu, Shengyu Tang, Yiming Cao, Changdong Yu
- PDF: `papers/2505.06694.pdf`

## Status: ALMOST
- Paper parsed and implementation-ready.
- No official NAS-DETR repository found from arXiv/GitHub search at build time.
- Reference implementations used for structure alignment:
  - `repositories/rtdetr` (RT-DETR)
  - `repositories/deformable-detr` (Deformable-DETR)

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---|---:|---|---|---|---|
| URPC2021 | 6,000 sonar images | train/test = 5,000 / 1,000 | URPC challenge | `/mnt/forge-data/datasets/urpc2021/` | MISSING (local) |
| URPC2022 | 9,200 sonar images | train/test = 8,400 / 800 | URPC challenge | `/mnt/forge-data/datasets/urpc2022/` | MISSING (local) |

## Core Hyperparameters (Paper)
| Param | Value | Ref |
|---|---|---|
| optimizer | Adam | §4.1.2 |
| learning_rate | 1e-4 (constant) | §4.1.2 |
| classification loss | Varifocal Loss, gamma=2.0 | Eq. (32) |
| box loss weights | λL1 : λGIoU = 5 : 2 | Eq. (33) |
| denoising noise std | σ = 0.1 | Eq. (34) |
| total loss weights | λcls : λbox : λdn = 1 : 2.5 : 0.5 | §3.4 |
| NAS rounds | 20,000 | §4.1.3 |
| NAS entropy weights A1 | {0,0,1,1,2,4} | §4.1.3 |
| NAS entropy weights A2 | {0,0,1,1,3,6} | §4.1.3 |

## NAS Backbone (Paper Table 2)
| Stage | Block | kernel (A1/A2) | in (A1/A2) | out (A1/A2) | stride | bottleneck (A1/A2) | layers |
|---|---|---|---|---|---|---|---|
| C1 | ResBlock* | 3/3 | 3/3 | 32/32 | 4/4 | 32/32 | 1/1 |
| C2 | ResBlock | 5/5 | 32/32 | 128/112 | 1/1 | 40/48 | 3/3 |
| C3 | ResBlock | 5/5 | 128/112 | 448/448 | 2/2 | 80/72 | 8/8 |
| C4 | ResBlock | 5/5 | 448/448 | 1280/1024 | 2/2 | 128/104 | 10/10 |
| C5 | ResBlock | 5/5 | 1280/1024 | 2048/2000 | 2/2 | 240/304 | 10/10 |
| C6 | TransformerBlock | -/- | 2048/2000 | 256/256 | -/- | -/- | 1/1 |

Transformer-specific (C6):
- hidden_dim: 424/504
- feedforward_dim: 912/1024

## Mutation Rules (Paper Table 3)
| Mutation | Operation | Applied To |
|---|---|---|
| kernel | choose [3,5] | ResBlock |
| layer depth | ± [1,2] | ResBlock |
| channel | x [1.5,1.25,0.8,0.6,0.5] | ResBlock, ResBlock* |
| bottleneck width | x [1.5,1.25,0.8,0.6,0.5] | ResBlock, ResBlock* |
| hidden dim | ± [8,16,32,64,128] | TransformerBlock |
| ffn dim | ± [8,16,32] | TransformerBlock |

## Expected Metrics (Paper)
| Benchmark | Metric | Paper Value | Target |
|---|---|---:|---:|
| URPC2021 | mmAP | 0.538 (NAS-DETR A1) | >= 0.53 |
| URPC2021 | mAP50 | 0.919 | >= 0.91 |
| URPC2021 | mAP75 | 0.569 | >= 0.55 |
| URPC2022 | mmAP | 0.492 (NAS-DETR A2) | >= 0.48 |
| URPC2022 | mAP50 | 0.954 | >= 0.94 |
| URPC2022 | mAP75 | 0.447 | >= 0.43 |
| Efficiency | FPS | 73.8 (A1), 71.2 (A2) | report |
| TRT quantized | FPS | 294.9 (A1), 288 (A2) | report |

## Runtime Stack (Local-first)
- Python: 3.14+
- PyTorch: 2.10+ (local), CUDA migration planned on server
- Config format: TOML
- Primary package: `src/anima_nasdetr`

