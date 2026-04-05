from anima_nasdetr.config import ModuleConfig, PaperVariant
from anima_nasdetr.nas.search import run_evolutionary_search


def test_search_returns_candidate() -> None:
    cfg = ModuleConfig.from_variant(PaperVariant.A1)
    cfg.search.rounds = 2
    cfg.search.population_size = 2
    candidate = run_evolutionary_search(cfg, rounds=2, population_size=2)
    assert len(candidate.stages) == 6
    assert isinstance(candidate.score, float)
