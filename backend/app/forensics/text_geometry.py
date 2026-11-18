"""Geometría del texto: línea base, altura y paso entre caracteres por línea (z robusto)."""
from dataclasses import dataclass, field

import numpy as np

from . import settings
from .glyphs import Glyph, lines_of


def robust_z(values, min_scale: float = 1.0):
    """z robusto en píxeles: escala mínima absoluta (en px) para que un MAD ~0 no dispare falsos positivos."""
    v = np.asarray(values, np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return (v - med) / max(mad, min_scale)
from .types import Detection


@dataclass
class GeometryResult:
    pitch_median: float = 0.0
    detections: list[Detection] = field(default_factory=list)


def analyze(glyphs: list[Glyph]) -> GeometryResult:
    cfg = settings.thresholds()["text_geometry"]
    out = GeometryResult()
    pitches_all = []
    for _, gl in lines_of(glyphs).items():
        if len(gl) < cfg["min_glyphs_per_line"]:
            continue
        gl = sorted(gl, key=lambda g: g.x)
        bottoms = np.array([g.y + g.h for g in gl], float)
        heights = np.array([g.h for g in gl], float)
        centers = np.array([g.x + g.w / 2 for g in gl], float)
        pitch = np.diff(centers)
        pitches_all += list(pitch)
        zb, zh = robust_z(bottoms), robust_z(heights)
        # los saltos entre palabras no son anomalías: solo se evalúa el paso dentro de la mediana*1.6
        mp = np.median(pitch)
        inner = pitch < 1.6 * mp
        # el paso solo es comparable en líneas de paso casi fijo (dígitos); en fuentes proporcionales
        # (letras) varía de forma natural y no se evalúa
        uniform = inner.sum() >= 4 and np.median(np.abs(pitch[inner] - mp)) / max(mp, 1) < 0.08
        zp = robust_z(pitch[inner]) if uniform else None
        pidx = np.where(inner)[0]
        for i, g in enumerate(gl):
            reasons = []
            if abs(zb[i]) > cfg["baseline_z"] and abs(bottoms[i] - np.median(bottoms)) >= 3:
                reasons.append("línea base")
            if abs(zh[i]) > cfg["height_z"] and abs(heights[i] - np.median(heights)) >= 3:
                reasons.append("altura")
            if zp is not None and i > 0 and (i - 1) in pidx:
                k = int(np.where(pidx == i - 1)[0][0])
                if abs(zp[k]) > cfg["pitch_z"] and abs(pitch[i - 1] - mp) >= 4:
                    reasons.append("espaciado")
            if reasons:
                out.detections.append(Detection(
                    "geometry_anomaly", f"Carácter fuera de la geometría de su línea ({', '.join(reasons)}).",
                    "low", 0.4, g.bbox, {"reasons": reasons}))
    if pitches_all:
        out.pitch_median = float(np.median(pitches_all))
    out.detections = out.detections[:3]
    return out
