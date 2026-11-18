"""OCR dirigido por campo del DNI: recorte por plantilla + varias variantes de preprocesado + votación.

El OCR de página completa sobre una foto es poco fiable (reflejos, holograma, fuentes mixtas). Aquí cada
campo se lee varias veces con preprocesados distintos; un valor se considera *fiable* solo si al menos
dos variantes coinciden. Un campo no fiable NUNCA debe producir una discrepancia con el barcode.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import settings, template as tpl

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

WHITELIST = {
    "digits": "0123456789.",
    "date": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/ ",
    "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    "year": "0123456789",
}


@dataclass
class FieldRead:
    value: str = ""
    votes: int = 0
    total: int = 0
    raw: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.votes >= 2 and bool(self.value)


def _variants(gray: np.ndarray) -> list[np.ndarray]:
    big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(big)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adap = cv2.adaptiveThreshold(cv2.GaussianBlur(big, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 41, 12)
    return [big, otsu, adap]


def _clean(kind: str, s: str) -> str:
    s = s.strip().upper()
    if kind == "digits":
        return re.sub(r"[^0-9]", "", s)
    if kind == "year":
        m = re.findall(r"(?:19|20)\d{2}", re.sub(r"[^0-9]", "", s))
        return m[-1] if m else ""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", "", s)).strip()


def _vote(kind: str, outs: list[str]) -> FieldRead:
    cleaned = [o for o in (_clean(kind, x) for x in outs) if o]
    if not cleaned:
        return FieldRead(raw=outs, total=len(outs))
    best, votes = Counter(cleaned).most_common(1)[0]
    return FieldRead(best, votes, len(outs), cleaned)


def read_field(img: np.ndarray, z: dict) -> FieldRead:
    H, W = img.shape[:2]
    x, y, w, h = int(z["x"] * W), int(z["y"] * H), int(z["w"] * W), int(z["h"] * H)
    kind = z.get("kind", "text")
    if kind == "date":
        # el año está al final de la fecha y queda fuera del holograma: se lee solo esa parte, solo dígitos
        x, w = x + int(0.58 * w), int(0.42 * w)
        kind = "year"
    cfg = f"--psm 7 -c tessedit_char_whitelist={WHITELIST[kind]}"
    outs = []
    # recortes desplazados un poco en vertical: la caja de la plantilla es aproximada
    shifts = (-0.006, 0.0, 0.006) if kind in ("year", "digits") else (0.0,)
    for s in shifts:
        yy = max(int(y + s * H), 0)
        crop = cv2.cvtColor(img[yy:yy + h, x:x + w], cv2.COLOR_BGR2GRAY)
        for v in _variants(crop):
            try:
                outs.append(pytesseract.image_to_string(v, config=cfg, lang="eng"))
            except Exception:
                outs.append("")
    return _vote(kind, outs)


def read_fields(img: np.ndarray) -> dict[str, FieldRead] | None:
    """None si no se puede (sin Tesseract o la imagen no tiene proporción de tarjeta)."""
    if pytesseract is None or not tpl.is_card(img):
        return None
    zones = settings.template().get("fields")
    if not zones:
        return None
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None
    return {name: read_field(img, z) for name, z in zones.items()}
