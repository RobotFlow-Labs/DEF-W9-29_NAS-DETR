from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch

from anima_nasdetr.config import ModuleConfig, PaperVariant, StageSpec
from anima_nasdetr.models.backbone import CnnTransformerBackbone, estimate_backbone_gflops
from .entropy import weighted_entropy_score


@dataclass
class Candidate:
    stages: list[StageSpec]
    score: float = -1e9


def _mutate_stage(spec: StageSpec) -> StageSpec:
    out = copy.deepcopy(spec)
    if out.block in {"res", "res_pool"}:
        op = random.choice(["kernel", "layers", "channel", "bottleneck"])
        if op == "kernel":
            out.kernel = random.choice([3, 5])
        elif op == "layers":
            out.layers = max(1, out.layers + random.choice([-2, -1, 1, 2]))
        elif op == "channel":
            ratio = random.choice([1.5, 1.25, 0.8, 0.6, 0.5])
            out.out_channels = max(16, int(out.out_channels * ratio))
        elif op == "bottleneck":
            ratio = random.choice([1.5, 1.25, 0.8, 0.6, 0.5])
            out.bottleneck = max(8, int(out.bottleneck * ratio))
    else:
        op = random.choice(["hidden", "ffn"])
        if op == "hidden":
            out.hidden_dim = max(64, out.hidden_dim + random.choice([-128, -64, -32, -16, -8, 8, 16, 32, 64, 128]))
        else:
            out.ffn_dim = max(128, out.ffn_dim + random.choice([-32, -16, -8, 8, 16, 32]))
    return out


def _repair_connectivity(stages: list[StageSpec]) -> list[StageSpec]:
    repaired = copy.deepcopy(stages)
    for i in range(1, len(repaired)):
        repaired[i].in_channels = repaired[i - 1].out_channels
    return repaired


def evaluate_candidate(stages: list[StageSpec], entropy_weights: tuple[int, int, int, int, int, int]) -> float:
    cfg = ModuleConfig.from_variant(PaperVariant.A1)
    cfg.model.stages = _repair_connectivity(stages)
    model = CnnTransformerBackbone(cfg.model).eval()

    gflops = estimate_backbone_gflops(model, 640, 640)
    if gflops > cfg.search.flops_cap_g:
        return -1e8

    with torch.no_grad():
        x = torch.randn(2, 3, 640, 640)
        feats = model(x)
    return weighted_entropy_score(feats, entropy_weights)


def run_evolutionary_search(
    base_cfg: ModuleConfig,
    rounds: int | None = None,
    population_size: int | None = None,
) -> Candidate:
    rounds = rounds or base_cfg.search.rounds
    population_size = population_size or base_cfg.search.population_size

    base = Candidate(copy.deepcopy(base_cfg.model.stages))
    base.score = evaluate_candidate(base.stages, base_cfg.search.entropy_weights)
    population = [base]

    for _ in range(max(1, rounds)):
        parent = random.choice(population)
        child_stages = copy.deepcopy(parent.stages)

        for idx in random.sample(range(len(child_stages)), k=min(4, len(child_stages))):
            child_stages[idx] = _mutate_stage(child_stages[idx])
            child_stages[idx] = _mutate_stage(child_stages[idx])

        child_stages = _repair_connectivity(child_stages)
        child = Candidate(child_stages)
        child.score = evaluate_candidate(child.stages, base_cfg.search.entropy_weights)

        population.append(child)
        population.sort(key=lambda x: x.score, reverse=True)
        population = population[:population_size]

    return population[0]
