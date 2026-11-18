"""Consistencia local por carácter: ruido del fondo adyacente, nitidez, contraste del trazo e iluminación.

Cada métrica se compara con la mediana de la MISMA línea de texto (z robusto con MAD). Un carácter es
atípico si al menos `min_metrics_flagged` métricas superan el umbral.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import settings
from .glyphs import Glyph, lines_of, robust_z, to_gray
from .types import Detection

METRICS = ("bg_noise", "sharpness", "stroke_contrast", "illumination")


@dataclass
class LocalResult:
    outliers: int = 0
    detections: list[Detection] = field(default_factory=list)


def glyph_metrics(gray: np.ndarray, ink: np.ndarray, g: Glyph, m: int = 5) -> dict | None:
    H, W = gray.shape
    x0, y0, x1, y1 = g.x - m, g.y - m, g.x + g.w + m, g.y + g.h + m
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    box = gray[y0:y1, x0:x1].astype(np.float32)
    im = cv2.dilate(ink[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0
    bg = box[~im]
    fg = box[ink[y0:y1, x0:x1] > 0]
    if bg.size < 30 or fg.size < 10:
        return None
    resid = box - cv2.GaussianBlur(box, (0, 0), 1.5)
    return {
        "bg_noise": float(resid[~im].std()),
        "sharpness": float(cv2.Laplacian(box[m:-m, m:-m], cv2.CV_32F).var()),
        "stroke_contrast": float(bg.mean() - fg.mean()),
        "illumination": float(bg.mean()),
    }


def metrics_by_glyph(img: np.ndarray, glyphs: list[Glyph], ink: np.ndarray) -> list[tuple[Glyph, dict]]:
    gray = to_gray(img)
    out = []
    for g in glyphs:
        m = glyph_metrics(gray, ink, g)
        if m:
            out.append((g, m))
    return out


def analyze(img: np.ndarray, glyphs: list[Glyph], ink: np.ndarray) -> LocalResult:
    cfg = settings.thresholds()["local_consistency"]
    result = LocalResult()
    by_line: dict[int, list[tuple[Glyph, dict]]] = {}
    for g, m in metrics_by_glyph(img, glyphs, ink):
        by_line.setdefault(g.line, []).append((g, m))
    for line, items in by_line.items():
        if len(items) < cfg["min_glyphs_per_line"]:
            continue
        zs = {k: robust_z(np.array([m[k] for _, m in items])) for k in METRICS}
        for idx, (g, m) in enumerate(items):
            flagged = [k for k in METRICS if abs(zs[k][idx]) > cfg["z_threshold"]]
            if len(flagged) >= cfg["min_metrics_flagged"]:
                result.outliers += 1
                score = float(np.clip(np.mean([abs(zs[k][idx]) for k in flagged]) / (cfg["z_threshold"] * 3), 0.2, 1))
                result.detections.append(Detection(
                    "local_inconsistency",
                    f"El carácter difiere del resto de su línea en: {', '.join(flagged)}. "
                    "Señal débil: la iluminación, los reflejos y el desgaste también causan esto.",
                    "low", round(score, 3), g.bbox, {"metrics": flagged}))
    result.detections.sort(key=lambda d: -d.score)
    result.detections = result.detections[:3]
    return result
