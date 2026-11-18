"""OCR con Tesseract (es+en). Para PDFs con capa de texto se usa ésta directamente."""
import logging
import shutil

import cv2
import numpy as np
import pytesseract

from .config import OCR_LANGS
from .models import Paragraph

log = logging.getLogger("ocr")


def _locate_tesseract() -> None:
    """En Windows el instalador no siempre agrega Tesseract al PATH: se prueban las rutas habituales."""
    import os
    if shutil.which("tesseract"):
        return
    for base in (os.environ.get("TESSERACT_CMD", ""),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
                 r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if base and os.path.isfile(base):
            pytesseract.pytesseract.tesseract_cmd = base
            return


_locate_tesseract()


def _lang() -> str:
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    wanted = [l for l in OCR_LANGS.split("+") if l in available]
    if len(wanted) < len(OCR_LANGS.split("+")):
        log.warning("Idiomas OCR faltantes; usando %s (disponibles: %s)", wanted, available)
    return "+".join(wanted) or "eng"


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None or bool(getattr(pytesseract.pytesseract, "tesseract_cmd", None)
                                                          and shutil.which(pytesseract.pytesseract.tesseract_cmd))


def ocr_page(image_bgr: np.ndarray, page: int) -> list[Paragraph]:
    """Devuelve párrafos (agrupados por block/par de Tesseract)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Tesseract rinde mal con texto de < ~25 px de alto (p.ej. una foto de 720 px de ancho dentro de un PDF):
    # se amplía a ~2200 px de lado largo (máx. x3) antes de leer.
    longest = max(gray.shape)
    if longest < 2200:
        s = min(2200 / longest, 3.0)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(gray, lang=_lang(), output_type=pytesseract.Output.DICT)
    groups: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        g = groups.setdefault(key, {"lines": {}, "conf": []})
        g["lines"].setdefault(data["line_num"][i], []).append(word)
        try:
            c = float(data["conf"][i])
            if c >= 0:
                g["conf"].append(c)
        except ValueError:
            pass
    out = []
    for key in sorted(groups):
        g = groups[key]
        text = "\n".join(" ".join(w) for _, w in sorted(g["lines"].items()))
        conf = round(sum(g["conf"]) / len(g["conf"]), 1) if g["conf"] else None
        out.append(Paragraph(page=page, text=text, confidence=conf))
    return out


def text_to_paragraphs(text: str, page: int) -> list[Paragraph]:
    return [Paragraph(page=page, text=p.strip()) for p in text.split("\n\n") if p.strip()]
