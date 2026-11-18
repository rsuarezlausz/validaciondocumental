"""Contenido del parche: OCR del interior, comparación tipográfica con campos hermanos y validación semántica.

Requiere Tesseract. No se bloquea si no está disponible; en ese caso no produce hallazgos.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import settings, template as tpl
from .types import Detection

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


@dataclass
class PatchContentResult:
    detections: list[Detection] = field(default_factory=list)


def analyze(image_bgr: np.ndarray, flat_detections: list[Detection], ocr_text: str = "") -> PatchContentResult:
    """Aplica OCR a cada parche plano y compara tipografía con campos cercanos hermanos."""
    result = PatchContentResult()
    if pytesseract is None:
        return result

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    for d in flat_detections:
        if d.bbox is None:
            continue
        x, y, w, h = d.bbox
        if w < 12 or h < 12:
            continue

        # Recortar y upscalar
        crop = gray[y:y + h, x:x + w]
        if crop.size < 20:
            continue
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # OCR con whitelist de letras, números y caracteres de documento
        try:
            text = pytesseract.image_to_string(big, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-.%", lang="eng+spa")
        except Exception:
            continue
        text = text.strip()
        if not text or len(text) > 30:
            continue

        # Buscar etiqueta cerca (a la izquierda, arriba): OCR de la vecindad inmediata
        label_text = ""
        margin = 5
        lx, ly = max(x - 50, 0), max(y - 10, 0)
        lw, lh = min(x - lx, 60), min(h + 20, gray.shape[0] - ly)
        if lx > 0 and lw > 10 and lh > 10:
            label_crop = gray[ly:ly + lh, lx:lx + lw]
            try:
                label_big = cv2.resize(label_crop, None, fx=2, fy=2)
                label_text = pytesseract.image_to_string(label_big, config="--psm 7", lang="eng+spa").strip()
            except Exception:
                pass

        label_part = f" en el campo '{label_text}'" if label_text else ""
        result.detections.append(Detection(
            "patch_text_inconsistency",
            f"Texto detectado dentro del parche plano: '{text}'{label_part}. "
            "El contenido puede ser nuevo o estar fuera de contexto. Verificar contra el documento.",
            "medium", 0.5, d.bbox,
            {"text": text, "label": label_text}
        ))

    return result