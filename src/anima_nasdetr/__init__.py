"""ANIMA NAS-DETR local scaffold."""

from .config import ModuleConfig, PaperVariant
from .models.nasdetr import NASDETR

__all__ = ["ModuleConfig", "PaperVariant", "NASDETR"]
