"""Costuras: líneas rectas muy finas (1-3 px) en el residuo de alta frecuencia, dentro de campos de texto.

Se evalúan solo píxeles SIN tinta (la costura cruza fondo) para no confundirla con los trazos de los
caracteres, y se compara cada columna/fila contra sus vecinos laterales.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import settings
from .glyphs import Glyph, lines_of, to_gray
from .types import Detection


@dataclass
class SeamResult:
    detected: bool = False
    detections: list[Detection] = field(default_factory=list)


def _scan(res: np.ndarray, free: np.ndarray, raw: np.ndarray, vertical: bool, med_h: float, cfg: dict,
          origin: tuple[int, int]):
    """Busca columnas (vertical=True) con diferencia sostenida respecto a las vecinas. res/free ya recortados."""
    if not vertical:
        res, free, raw = res.T, free.T, raw.T
    H, W = res.shape
    win = max(int(cfg["min_run_fraction"] * med_h), 12)
    out = []
    if H < win or W < 9:
        return out
    for x in range(4, W - 4):
        left = np.minimum(res[:, x - 4], res[:, x - 3])
        right = np.minimum(res[:, x + 3], res[:, x + 4])
        neigh = (left + right) / 2.0
        for sign in (1, -1):
            diff = sign * (res[:, x] - neigh)
            # pico de UN píxel en la imagen sin suavizar: un hueco claro entre trazos oscuros no es un pico
            peak = sign * raw[:, x] - np.maximum(sign * raw[:, x - 1], sign * raw[:, x + 1])
            ok = (diff > cfg["min_contrast"]) & free[:, x] & (peak > 0.5 * cfg["min_contrast"])
            # ancho: la columna contigua (x+-2) no debe cumplir el mismo contraste (descarta trazos anchos)
            wide = ((sign * (res[:, x - 2] - neigh)) > cfg["min_contrast"]) & ((sign * (res[:, x + 2] - neigh)) > cfg["min_contrast"])
            # corrida contigua más larga
            run = best = start = 0
            best_start = 0
            for i in range(H):
                if ok[i]:
                    if run == 0:
                        start = i
                    run += 1
                    if run > best:
                        best, best_start = run, start
                else:
                    run = 0
            if best >= win:
                seg = ok[best_start:best_start + best]
                # una línea fina no tiene a ambos lados una franja igual de brillante (eso sería un trazo ancho)
                rows = slice(best_start, best_start + best)
                if seg.mean() >= cfg["consistency"] and wide[rows].mean() < 0.3:
                    out.append((x, best_start, best, float(diff[best_start:best_start + best].mean()), sign))
    return out


def analyze(img: np.ndarray, glyphs: list[Glyph], ink: np.ndarray) -> SeamResult:
    cfg = settings.thresholds()["seam"]
    result = SeamResult()
    if len(glyphs) < 4:
        return result
    gray = to_gray(img).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 3.0)
    ink_d = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    free_all = ~ink_d
    med_h = float(np.median([g.h for g in glyphs]))
    H, W = gray.shape
    raw = []
    for _, gl in lines_of(glyphs).items():
        x0 = max(min(g.x for g in gl) - 40, 0)
        x1 = min(max(g.x + g.w for g in gl) + 40, W)
        y0 = max(int(np.median([g.y for g in gl]) - 0.8 * med_h), 0)
        y1 = min(int(np.median([g.y + g.h for g in gl]) + 0.8 * med_h), H)
        sub, free, subraw = residual[y0:y1, x0:x1], free_all[y0:y1, x0:x1], gray[y0:y1, x0:x1]
        for x, ys, ln, contrast, sg in _scan(sub, free, subraw, True, med_h, cfg, (x0, y0)):
            raw.append((x0 + x, y0 + ys, 3, ln, contrast, "vertical", sg))
        for y, xs, ln, contrast, sg in _scan(sub, free, subraw, False, med_h, cfg, (x0, y0)):
            raw.append((x0 + xs, y0 + y, ln, 3, contrast, "horizontal", sg))
    # fusiona detecciones adyacentes (misma línea, columnas contiguas)
    raw.sort()
    merged = []
    for r in raw:
        if merged and r[5] == merged[-1][5] and abs(r[0] - merged[-1][0]) <= 3 and abs(r[1] - merged[-1][1]) <= 6:
            continue
        merged.append(r)
    # Líneas paralelas repetidas (código de barras, rayas, tramas) no son una costura aislada
    def parallel(r):
        coord, lo, hi = (r[0], r[1], r[1] + r[3]) if r[5] == "vertical" else (r[1], r[0], r[0] + r[2])
        n = 0
        for o in merged:
            if o is r or o[5] != r[5]:
                continue
            c2, lo2, hi2 = (o[0], o[1], o[1] + o[3]) if o[5] == "vertical" else (o[1], o[0], o[0] + o[2])
            if abs(c2 - coord) <= 40 and lo < hi2 and lo2 < hi:
                n += 1
        return n
    merged = [r for r in merged if parallel(r) < 1]
    merged.sort(key=lambda r: -r[4])
    for x, y, w, h, contrast, orient, sg in merged[:5]:
        score = float(np.clip(contrast / (cfg["min_contrast"] * 3), 0.3, 1.0))
        result.detections.append(Detection(
            "seam_artifact",
            f"Línea {orient} muy fina ({h if orient == 'vertical' else w}px de largo) con contraste {contrast:.0f} sobre "
            "el fondo, fuera de los trazos: posible costura de edición. Puede ser también un defecto de la foto.",
            "medium", round(score, 3), (int(x - 2), int(y), int(w if orient == 'horizontal' else 5), int(h if orient == 'vertical' else 5)),
            {"orientation": orient, "contrast": round(contrast, 1), "bright": sg > 0}))
    result.detected = bool(result.detections)
    return result
