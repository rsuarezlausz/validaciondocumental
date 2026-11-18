"""Segmentación de caracteres por línea de texto (umbral adaptativo + componentes conexas)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Glyph:
    x: int
    y: int
    w: int
    h: int
    line: int = -1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Máscara binaria (255 = tinta) con umbral adaptativo; el tamaño de ventana escala con la imagen."""
    k = int(max(15, (min(gray.shape) // 25) | 1))
    k += (k % 2 == 0)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, 10)


def segment_glyphs(img: np.ndarray, exclude: np.ndarray | None = None,
                   min_h_frac: float = 0.012, max_h_frac: float = 0.06) -> tuple[list[Glyph], np.ndarray]:
    """Devuelve (glifos ordenados por línea y x, máscara de tinta). `exclude`: máscara bool de zonas a ignorar."""
    gray = to_gray(img)
    H, W = gray.shape
    mask = ink_mask(gray)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cand: list[Glyph] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_h_frac * H <= h <= max_h_frac * H):
            continue
        if not (0.15 * h <= w <= 1.3 * h):
            continue
        fill = area / float(w * h)
        if not (0.12 <= fill <= 0.9):
            continue
        if exclude is not None and exclude[int(cents[i][1]), int(cents[i][0])]:
            continue
        cand.append(Glyph(int(x), int(y), int(w), int(h)))
    if not cand:
        return [], mask
    # Agrupa por línea con tolerancia relativa a la altura del propio glifo (el DNI mezcla tamaños de letra:
    # una mediana global descartaría los dígitos grandes) y filtra ruido contra la mediana de SU línea.
    cand.sort(key=lambda g: g.cy)
    groups: list[list[Glyph]] = []
    for g in cand:
        if groups:
            cy = float(np.mean([q.cy for q in groups[-1]]))
            mh = float(np.median([q.h for q in groups[-1]]))
            if abs(g.cy - cy) <= 0.5 * max(g.h, mh):
                groups[-1].append(g)
                continue
        groups.append([g])
    cand = []
    for k, grp in enumerate(groups):
        mh = float(np.median([q.h for q in grp]))
        for q in grp:
            if 0.6 * mh <= q.h <= 1.6 * mh or len(grp) < 3:
                q.line = k
                cand.append(q)
    cand.sort(key=lambda g: (g.line, g.x))
    return cand, mask


def lines_of(glyphs: list[Glyph]) -> dict[int, list[Glyph]]:
    out: dict[int, list[Glyph]] = {}
    for g in glyphs:
        out.setdefault(g.line, []).append(g)
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    scale = max(mad, 0.05 * abs(med), 1e-6)
    return (v - med) / scale
