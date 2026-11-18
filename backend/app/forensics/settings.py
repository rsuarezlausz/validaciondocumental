"""Carga de configuración YAML (umbrales calibrados, plantilla del documento, reglas)."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent
DEFAULTS = DIR / "config"


def _path(name: str, env: str) -> Path:
    override = os.getenv(env)
    return Path(override) if override else DEFAULTS / name


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def thresholds() -> dict:
    return _load(str(_path("thresholds.yaml", "FORENSIC_THRESHOLDS")))


def template() -> dict:
    return _load(str(_path("template_dni.yaml", "FORENSIC_TEMPLATE")))


def rules() -> dict:
    return _load(str(_path("rules.yaml", "FORENSIC_RULES")))


def profiles() -> dict:
    return _load(str(_path("profiles.yaml", "FORENSIC_PROFILES")))


def ml_enabled() -> bool:
    return os.getenv("ENABLE_ML_LOCALIZATION", "false").lower() in ("1", "true", "yes")


def reload() -> None:
    _load.cache_clear()
