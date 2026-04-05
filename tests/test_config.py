from anima_nasdetr.config import ModuleConfig, PaperVariant


def test_variants_have_expected_entropy_weights() -> None:
    a1 = ModuleConfig.from_variant(PaperVariant.A1)
    a2 = ModuleConfig.from_variant(PaperVariant.A2)
    assert a1.search.entropy_weights == (0, 0, 1, 1, 2, 4)
    assert a2.search.entropy_weights == (0, 0, 1, 1, 3, 6)


def test_model_has_six_stages() -> None:
    cfg = ModuleConfig.from_variant(PaperVariant.A1)
    assert len(cfg.model.stages) == 6
