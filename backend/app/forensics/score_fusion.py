"""Fusión ponderada y explicable de las señales forenses.

Cada señal aporta `peso_máximo * score`. Las señales fuertes (barcode, copy-move, costura, reglas) se
combinan con noisy-OR; las débiles (ELA, ruido, ghost JPEG, metadatos, geometría, consistencia local)
se combinan aparte y quedan acotadas por `weak_cap`: una sola señal débil, o todas juntas, nunca pasan
de un score medio. La confianza baja cuando faltan señales (barcode ilegible, pocos glifos, imagen pequeña).
"""
from dataclasses import dataclass, field

import numpy as np

from . import settings
from .types import Detection

# Fiabilidad base de los detectores de píxeles según el tipo de entrada.
INPUT_RELIABILITY = {"jpeg": 0.85, "png": 0.45, "pdf": 0.5}

LIMITATIONS = [
    "Este análisis estima una probabilidad y muestra evidencia; no afirma que un documento sea falso ni auténtico.",
    "Las señales son heurísticas con umbrales provisionales, no calibrados con una población de documentos reales.",
    "Una probabilidad baja no descarta una alteración: ediciones pequeñas, re-fotografiadas o re-comprimidas pueden pasar.",
    "Las reglas de vigencia son supuestos por verificar y un incumplimiento puede tener explicaciones legítimas.",
]


@dataclass
class Fusion:
    probability: float
    confidence: float
    contributions: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verdict: str = "inconclusive"  # evidence | no_evidence | inconclusive


def _noisy_or(values: list[float]) -> float:
    p = 1.0
    for v in values:
        p *= (1.0 - float(np.clip(v, 0, 1)))
    return 1.0 - p


def fuse(detections: list[Detection], pixel_scores: dict[str, float], meta_weight: float, input_kind: str,
         resolution: float, availability: dict) -> Fusion:
    """pixel_scores: {'ela','noise','jpeg_ghost'} máx. por documento. availability: barcode, glyphs, copy_move_applicable."""
    cfg = settings.thresholds()["fusion"]
    w = cfg["weights"]
    weak_types = set(cfg["weak_types"])
    contribs: list[dict] = []

    best: dict[str, float] = {}
    for d in detections:
        if d.type in w:
            best[d.type] = max(best.get(d.type, 0.0), d.score)
    for t, s in pixel_scores.items():
        if t in w:
            best[t] = max(best.get(t, 0.0), s)
    best["metadata"] = float(np.clip(meta_weight, 0, 0.4) / 0.4)

    strong, weak = [], []
    for t, s in best.items():
        if s <= 0:
            continue
        pts = w[t] * s
        contribs.append({"signal": t, "score": round(s, 3), "weight": w[t], "points": round(pts, 3),
                         "group": "weak" if t in weak_types else "strong"})
        (weak if t in weak_types else strong).append(pts)
    weak_p = min(_noisy_or(weak), cfg["weak_cap"])
    prob = float(np.clip(_noisy_or(strong + [weak_p]), 0, 1))

    reliability = INPUT_RELIABILITY.get(input_kind, 0.5) * (0.5 + 0.5 * resolution)
    conf = 0.15 + 0.6 * reliability
    warnings: list[str] = []
    lim = list(LIMITATIONS)
    if availability.get("barcode") in ("match", "mismatch", "partial"):
        conf += 0.12
    else:
        # El barcode es opcional: no todos los documentos lo tienen. Su ausencia no resta fiabilidad ni suma
        # sospecha; solo se informa que esa comprobación independiente no estuvo disponible.
        lim.append("No se leyó un código de barras: no hubo validación cruzada con el texto impreso (comprobación opcional).")
    if availability.get("glyphs", 0) < 10:
        conf -= 0.12
        warnings.append("Pocos caracteres segmentados: las señales locales son poco fiables.")
    if not availability.get("copy_move_applicable", True):
        conf -= 0.08
        lim.append("Copy-move entre glifos no aplicable: render digital sin ruido de sensor, recompresión fuerte o texto de "
                   "resolución demasiado baja (< 20 px de alto) para comparar el ruido de fondo.")
    if len({c["signal"] for c in contribs if c["group"] == "strong"}) >= 2:
        conf += 0.08
    if input_kind == "png":
        warnings.append("Formato sin historial de compresión JPEG (PNG/captura): la detección de ediciones es poco fiable.")
    elif input_kind == "pdf":
        warnings.append("PDF con páginas rasterizadas: el análisis de píxeles pierde las estadísticas de la imagen original.")
    if resolution < 0.5:
        warnings.append("Resolución baja: el análisis es menos fiable.")
    if prob < cfg["medium_probability"] and conf < 0.6:
        warnings.append("Una probabilidad baja no descarta una alteración con esta fiabilidad.")
    # Veredicto de tres estados
    verdict = "inconclusive"
    if prob >= cfg["high_probability"] and conf >= 0.5:
        verdict = "evidence"
    elif prob < cfg.get("inconclusive_threshold", 0.15) and len(contribs) < 2:
        verdict = "inconclusive"
    elif prob < cfg["medium_probability"]:
        verdict = "no_evidence"
    else:
        verdict = "inconclusive"
    return Fusion(round(prob, 3), round(float(np.clip(conf, 0.1, 0.9)), 3), contribs, lim, warnings, verdict)
