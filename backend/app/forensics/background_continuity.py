"""Continuidad del fondo: detecta si el patrón (guilloché, líneas, tramas) se interrumpe en el borde del parche.

Usa Gabor o FFT local en el anillo del parche y comprueba que la orientación/frecuencia dominante
desaparece al cruzar el borde hacia el interior. Esto sube la severidad de un `flat_patch` pero
no es un detector independiente de alta confianza.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import settings
from .types import Detection


@dataclass
class BackgroundResult:
    detections: list[Detection] = field(default_factory=list)


def _gabor_energy(patch: np.ndarray, theta: float) -> float:
    """Energía de Gabor en una orientación theta."""
    kernel = cv2.getGaborKernel((9, 9), 4.0, theta, 6.0, 0.5, 0, cv2.CV_32F)
    filtered = cv2.filter2D(patch.astype(np.float32), cv2.CV_32F, kernel)
    return float(np.abs(filtered).mean())


def analyze(gray: np.ndarray, flat_detections: list[Detection]) -> BackgroundResult:
    """Evalúa la continuidad del fondo para cada flat_patch. gray: imagen en grises."""
    result = BackgroundResult()
    for d in flat_detections:
        if d.bbox is None:
            continue
        x, y, w, h = d.bbox
        if w < 10 or h < 10:
            continue
        # Anillo exterior más amplio (12 px)
        margin = 12
        x0, y0 = max(x - margin, 0), max(y - margin, 0)
        x1, y1 = min(x + w + margin, gray.shape[1]), min(y + h + margin, gray.shape[0])
        outer = gray[y0:y1, x0:x1].copy()

        # Interior
        inner = gray[y:y + h, x:x + w].copy()
        if inner.size < 10 or outer.size < 10:
            continue

        # Medir energía de Gabor en 4 orientaciones en el anillo
        ring = outer.copy()
        # Crear máscara de anillo
        ring_mask = np.ones_like(ring, bool)
        iy0, ix0 = y - y0, x - x0
        iy1, ix1 = iy0 + h, ix0 + w
        if iy0 >= 0 and ix0 >= 0 and iy1 <= ring.shape[0] and ix1 <= ring.shape[1]:
            ring_mask[iy0:iy1, ix0:ix1] = False

        ring_patch = ring[ring_mask]
        if ring_patch.size < 50:
            continue

        inner_patch = inner
        if inner_patch.size < 10:
            continue

        # Energía de Gabor: máx sobre 4 orientaciones
        outer_energies = [_gabor_energy(ring, t) for t in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]]
        inner_energies = [_gabor_energy(inner_patch, t) for t in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]]
        max_outer = max(outer_energies)
        max_inner = max(inner_energies)
        energy_ratio = max_outer / max(max_inner, 0.01)

        # Salto de luminancia: diferencia de media entre interior y cada lado
        sides = []
        hw, hh = w // 2, h // 2
        # izquierda, derecha, arriba, abajo
        box_im = gray[y:y + h, x:x + w].astype(float)
        for sx, sy, sw, sh in [
            (max(x - margin, 0), y, min(margin, x), h),
            (x + w, y, min(margin, gray.shape[1] - x - w), h),
            (x, max(y - margin, 0), w, min(margin, y)),
            (x, y + h, w, min(margin, gray.shape[0] - y - h)),
        ]:
            if sx > 0 and sy > 0 and sw > 2 and sh > 2:
                side = gray[sy:sy + sh, sx:sx + sw].astype(float)
                sides.append(float(abs(side.mean() - box_im.mean())))

        lum_jump = max(sides) if sides else 0.0

        # Solo subir severidad si la energía del anillo es clara y la del interior no
        if energy_ratio > 5 and max_outer > 1.0:
            boost = min(energy_ratio / 30, 1.0) * 0.5
            result.detections.append(Detection(
                "background_discontinuity",
                f"El patrón de fondo del anillo (energía Gabor {max_outer:.1f}) desaparece abruptamente en "
                f"el interior ({max_inner:.1f}, ratio {energy_ratio:.1f}x). Salto de luminancia borde: {lum_jump:.0f} niveles. "
                "Refuerza la sospecha de parche.", "info", round(boost, 3), d.bbox,
                {"energy_ratio": round(energy_ratio, 1), "lum_jump": round(lum_jump, 1),
                 "max_outer_energy": round(max_outer, 2), "max_inner_energy": round(max_inner, 2)}
            ))

    return result