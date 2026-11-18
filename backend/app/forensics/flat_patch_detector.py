"""Detector de parches planos (white-out / overpaint).

Busca componentes de píxeles con baja varianza local (planos) y compara sus estadísticas con
las del anillo circundante. Un parche genuino (blanqueado sobre guilloché) es:
- interior plano (baja desviación estándar)
- anillo texturado (alta desviación)
- interior más brillante que el anillo
- rectangular y alineado a ejes

Esto evita falsos positivos en fondos lisos y márgenes (anillo también plano), y en espacios
entre dígitos de campos de texto (interior y anillo tienen mismo brillo).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import Detection

RING_MARGIN = 10
FLAT_MIN_PX = 8


@dataclass
class FlatPatchResult:
    detections: list[Detection] = field(default_factory=list)
    max_score: float = 0.0
    patches_evaluated: int = 0


def _local_variance(gray: np.ndarray, k: int = 5) -> np.ndarray:
    mean = cv2.blur(gray.astype(np.float32), (k, k))
    sq = cv2.blur(gray.astype(np.float32) ** 2, (k, k))
    return np.maximum(sq - mean * mean, 0)


def _hf_noise(gray: np.ndarray) -> np.ndarray:
    return np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0))


def _ring_mask(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin: int) -> np.ndarray | None:
    x, y, w, h = bbox
    H, W = shape[:2]
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, W), min(y + h + margin, H)
    outer = np.zeros(shape[:2], bool)
    outer[y0:y1, x0:x1] = True
    ix0, iy0 = max(x, 0), max(y, 0)
    ix1, iy1 = min(x + w, W), min(y + h, H)
    outer[iy0:iy1, ix0:ix1] = False
    if outer.sum() < 50:
        return None
    return outer


def _rectangularity(mask: np.ndarray) -> float:
    nz = np.where(mask)
    if len(nz[0]) < 5:
        return 0.0
    h = nz[0].max() - nz[0].min() + 1
    w = nz[1].max() - nz[1].min() + 1
    return mask.sum() / float(max(w * h, 1))


def _score_candidate(inside_std: float, ring_std: float, inside_hf: float, ring_hf: float,
                     brightness_ratio: float, rect: float) -> float:
    std_ratio = ring_std / max(inside_std, 0.01)
    hf_ratio = ring_hf / max(inside_hf, 0.01)
    s_std = float(np.clip((std_ratio - 5) / (40 - 5), 0, 1))
    s_hf = float(np.clip((hf_ratio - 5) / (40 - 5), 0, 1))
    s_rect = float(np.clip((rect - 0.6) / 0.3, 0, 1))
    s_br = float(np.clip((brightness_ratio - 1.02) / 0.05, 0, 1))
    return float(np.clip(0.25 * s_std + 0.2 * s_hf + 0.3 * s_rect + 0.25 * s_br, 0, 1))


def _expand_to_fill(gray: np.ndarray, bbox: tuple[int, int, int, int], level: float, max_grow: int = 40):
    """Crece el rectángulo lado a lado mientras la fila/columna vecina siga siendo (casi) del mismo nivel plano:
    el mapa de varianza recorta el borde del parche; así se recupera el rectángulo completo."""
    H, W = gray.shape
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    tol = 6.0
    for _ in range(max_grow):
        grew = False
        if x0 > 0 and (np.abs(gray[y0:y1, x0 - 1].astype(float) - level) <= tol).mean() >= 0.9:
            x0 -= 1; grew = True
        if x1 < W and (np.abs(gray[y0:y1, x1].astype(float) - level) <= tol).mean() >= 0.9:
            x1 += 1; grew = True
        if y0 > 0 and (np.abs(gray[y0 - 1, x0:x1].astype(float) - level) <= tol).mean() >= 0.9:
            y0 -= 1; grew = True
        if y1 < H and (np.abs(gray[y1, x0:x1].astype(float) - level) <= tol).mean() >= 0.9:
            y1 += 1; grew = True
        if not grew:
            break
    return x0, y0, x1 - x0, y1 - y0


def analyze(gray_or_bgr: np.ndarray, exclude_mask: np.ndarray | None = None) -> FlatPatchResult:
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    gray = gray.astype(np.uint8)
    result = FlatPatchResult()
    H, W = gray.shape

    # Máscara de exclusión
    if exclude_mask is not None:
        excl = cv2.GaussianBlur(exclude_mask.astype(np.float32), (0, 0), 5) > 0.5
    else:
        excl = np.zeros((H, W), bool)

    # Mapa de varianza local
    var = _local_variance(gray, 5)

    # Píxeles planos: varianza local muy baja
    low_var = (var < 1.5).astype(np.uint8)

    # Procesar a múltiples escalas
    best: list[tuple[float, tuple, dict]] = []

    for scale in [1.0, 0.5, 0.25]:
        if min(H, W) * scale < FLAT_MIN_PX * 2:
            continue
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        sv = _local_variance(small, 5)
        sm = (sv < 1.5).astype(np.uint8)

        # Cerrar para unir regiones planas cercanas
        k = max(5, int(11 * scale) | 1)
        sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

        # Contornos externos rellenos: un parche con texto encima (la "A") es plano salvo los trazos; el contorno
        # externo del conjunto plano es el rectángulo completo y los trazos quedan como agujeros.
        cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if min(w, h) < FLAT_MIN_PX * scale or w * h < 200 * scale * scale:
                continue
            filled = np.zeros(sm.shape, np.uint8)
            cv2.drawContours(filled, [cnt], -1, 1, thickness=cv2.FILLED)
            comp = filled.astype(bool)
            rect = _rectangularity(comp)
            if rect < 0.75:
                continue
            flat_px = comp & (sv < 1.5)
            flat_frac = flat_px.sum() / max(comp.sum(), 1)
            if flat_frac < 0.6:      # demasiada "tinta" dentro: no es un relleno
                continue

            factor = 1.0 / scale
            bx, by, bw, bh = int(x * factor), int(y * factor), int(w * factor), int(h * factor)
            if bx + bw > W or by + bh > H or bw < 1 or bh < 1:
                continue
            if excl[by:by + bh, bx:bx + bw].any():
                continue

            ring = _ring_mask((bx, by, bw, bh), gray.shape, int(RING_MARGIN))
            if ring is None or ring.sum() < 50:
                continue
            ring_pixels = gray[ring]
            inner = gray[by:by + bh, bx:bx + bw]
            # estadísticas del interior solo sobre los píxeles planos (excluye los trazos del texto nuevo)
            flat_full = cv2.resize(flat_px.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) \
                if scale != 1.0 else flat_px
            sel = flat_full[by:by + bh, bx:bx + bw]
            vals = inner[sel] if sel.sum() >= 20 else inner.ravel()
            in_std = float(vals.std())
            ring_std = float(ring_pixels.std())
            hf = _hf_noise(gray)
            in_hf = float(hf[by:by + bh, bx:bx + bw][sel].mean()) if sel.sum() >= 20 else float(hf[by:by + bh, bx:bx + bw].mean())
            ring_hf = float(hf[ring].mean())

            if ring_std < 5.0:
                continue
            std_ratio = ring_std / max(in_std, 0.01)
            if std_ratio < 5.0 or in_std > 2.0:
                continue
            brightness_ratio = float(vals.mean()) / max(float(ring_pixels.mean()), 1.0)
            if brightness_ratio < 1.02:
                continue

            bx, by, bw, bh = _expand_to_fill(gray, (bx, by, bw, bh), float(vals.mean()))
            score = _score_candidate(in_std, ring_std, in_hf, ring_hf, brightness_ratio, rect)
            if score < 0.15:
                continue
            result.patches_evaluated += 1
            best.append((score, (bx, by, bw, bh), {
                "inside_std": round(in_std, 2), "ring_std": round(ring_std, 2),
                "hf_ratio": round(ring_hf / max(in_hf, 0.01), 1),
                "std_ratio": round(std_ratio, 1),
                "brightness_ratio": round(brightness_ratio, 3),
            }))

    # Fusionar candidatos solapados
    best.sort(key=lambda e: -e[0])
    merged = []
    for score, bbox, ev in best:
        overlap = False
        for m in merged:
            x0 = max(bbox[0], m[1][0]); y0 = max(bbox[1], m[1][1])
            x1 = min(bbox[0] + bbox[2], m[1][0] + m[1][2])
            y1 = min(bbox[1] + bbox[3], m[1][1] + m[1][3])
            if x0 < x1 and y0 < y1:
                overlap = True
                if score > m[0]:
                    merged.remove(m); merged.append((score, bbox, ev))
                break
        if not overlap and len(merged) < 3:
            merged.append((score, bbox, ev))

    for score, bbox, ev in merged:
        sev = "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low")
        result.detections.append(Detection(
            "flat_patch",
            f"Parche plano (desv. interior {ev['inside_std']:.2f} vs anillo {ev['ring_std']:.2f}, brillo {ev['brightness_ratio']:.2f}x) en zona con textura: posible blanqueado o relleno.",
            sev, round(score, 3), bbox, ev
        ))
    result.max_score = max((d.score for d in result.detections), default=0.0)
    return result