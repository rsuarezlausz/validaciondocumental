"""Perfil de documento: decide qué plantilla y reglas son aplicables según lo que el OCR identifica."""
import re
import unicodedata
from dataclasses import dataclass

from . import settings


@dataclass(frozen=True)
class Profile:
    name: str
    uses_template: bool = False
    check_validity: bool = False


GENERIC = Profile("generic")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s)).strip()


def get(name: str | None) -> Profile:
    if not name or name == "generic":
        return GENERIC
    cfg = settings.profiles().get("profiles", {}).get(name)
    if not cfg:
        return GENERIC
    return Profile(name, bool(cfg.get("uses_template")), bool(cfg.get("check_validity")))


def detect(ocr_text: str) -> Profile:
    """Perfil por palabras clave del OCR. Sin coincidencia -> genérico (no se asume ningún documento concreto)."""
    text = _norm(ocr_text)
    for name, cfg in settings.profiles().get("profiles", {}).items():
        if any(_norm(k) in text for k in cfg.get("keywords", [])):
            return get(name)
    return GENERIC
