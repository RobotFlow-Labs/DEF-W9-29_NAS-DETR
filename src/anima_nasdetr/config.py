from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class PaperVariant(str, Enum):
    A1 = "A1"
    A2 = "A2"


@dataclass
class StageSpec:
    name: str
    block: Literal["res", "res_pool", "transformer"]
    kernel: int
    in_channels: int
    out_channels: int
    stride: int
    bottleneck: int
    layers: int
    hidden_dim: int = 0
    ffn_dim: int = 0


@dataclass
class LossConfig:
    gamma: float = 2.0
    lambda_l1: float = 5.0
    lambda_giou: float = 2.0
    lambda_cls: float = 1.0
    lambda_box: float = 2.5
    lambda_dn: float = 0.5
    denoise_sigma: float = 0.1


@dataclass
class SearchConfig:
    rounds: int = 20_000
    population_size: int = 12
    entropy_weights: tuple[int, int, int, int, int, int] = (0, 0, 1, 1, 2, 4)
    flops_cap_g: float = 160.0


@dataclass
class DataConfig:
    image_size: tuple[int, int] = (640, 640)
    num_classes: int = 10
    urpc2021_train_test: tuple[int, int] = (5000, 1000)
    urpc2022_train_test: tuple[int, int] = (8400, 800)


@dataclass
class ModelConfig:
    num_queries: int = 300
    decoder_layers: int = 6
    variant: PaperVariant = PaperVariant.A1
    stages: list[StageSpec] = field(default_factory=list)

    @staticmethod
    def from_variant(variant: PaperVariant) -> "ModelConfig":
        # Table 2 values from paper (A1/A2).
        a1 = {
            "c1": StageSpec("C1", "res_pool", 3, 3, 32, 4, 32, 1),
            "c2": StageSpec("C2", "res", 5, 32, 128, 1, 40, 3),
            "c3": StageSpec("C3", "res", 5, 128, 448, 2, 80, 8),
            "c4": StageSpec("C4", "res", 5, 448, 1280, 2, 128, 10),
            "c5": StageSpec("C5", "res", 5, 1280, 2048, 2, 240, 10),
            "c6": StageSpec("C6", "transformer", 1, 2048, 256, 1, 0, 1, hidden_dim=424, ffn_dim=912),
        }
        a2 = {
            "c1": StageSpec("C1", "res_pool", 3, 3, 32, 4, 32, 1),
            "c2": StageSpec("C2", "res", 5, 32, 112, 1, 48, 3),
            "c3": StageSpec("C3", "res", 5, 112, 448, 2, 72, 8),
            "c4": StageSpec("C4", "res", 5, 448, 1024, 2, 104, 10),
            "c5": StageSpec("C5", "res", 5, 1024, 2000, 2, 304, 10),
            "c6": StageSpec("C6", "transformer", 1, 2000, 256, 1, 0, 1, hidden_dim=504, ffn_dim=1024),
        }
        spec = a1 if variant == PaperVariant.A1 else a2
        return ModelConfig(variant=variant, stages=list(spec.values()))


@dataclass
class ModuleConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig.from_variant(PaperVariant.A1))
    losses: LossConfig = field(default_factory=LossConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @staticmethod
    def from_variant(variant: PaperVariant) -> "ModuleConfig":
        cfg = ModuleConfig()
        cfg.model = ModelConfig.from_variant(variant)
        if variant == PaperVariant.A2:
            cfg.search.entropy_weights = (0, 0, 1, 1, 3, 6)
        return cfg

