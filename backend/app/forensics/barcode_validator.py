"""Validación cruzada del PDF417 del DNI contra el texto OCR.

El formato del contenido NO se asume: los campos se clasifican por patrón (fechas dd/mm/aaaa,
número de 7-8 dígitos, trámite de 11 dígitos, letras sueltas M/F/X y ejemplar, textos). Se informa
qué campos se encontraron; si el barcode trae solo 2 fechas (nacimiento, emisión) el vencimiento
NO se valida aquí sino con las reglas de vigencia (rules.py).

Privacidad: el contenido crudo del barcode nunca se devuelve ni se registra; los valores sensibles
salen enmascarados.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np

from .types import Detection, mask_value

try:  # zxing-cpp es opcional en tiempo de import para que el resto del paquete funcione sin él
    import zxingcpp
except Exception:  # pragma: no cover
    zxingcpp = None

MONTHS = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AGO": 8,
          "SEP": 9, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12,
          "JAN": 1, "APR": 4, "AUG": 8, "DEC": 12}
SENSITIVE = {"doc_number", "tramite", "surname", "name"}
# Sinónimos por campo (ES/EN; "caducidad" en el DNI peruano)
LABELS = {"birth": ("NACIMIENTO", "BIRTH"), "issue": ("EMISION", "ISSUE"),
          "expiry": ("VENCIMIENTO", "CADUCIDAD", "EXPIRY")}
_DATE_PATTERNS = (
    re.compile(r"(\d{1,2}) ?([A-Z]{3}) ?/? ?(?:[A-Z]{3})? ?(\d{4})"),      # 18 DIC/ DEC 2015
    re.compile(r"(\d{1,2})[ /.\-](\d{1,2})[ /.\-](\d{4})"),               # 19 07 2021, 19/07/2021
    re.compile(r"(\d{2})(\d{2}) (\d{4})"),                                  # 1907 2029 (OCR pegó día y mes)
)


def _find_date(segment: str) -> Optional[date]:
    """Primera fecha válida del segmento (se prueba cada posición: un intento inválido no tapa el siguiente)."""
    for pos in range(len(segment)):
        for pat in _DATE_PATTERNS:
            m = pat.match(segment, pos)
            if not m:
                continue
            a, b, y = m.group(1), m.group(2), int(m.group(3))
            mon = MONTHS.get(b) if not b.isdigit() else int(b)
            if not mon or not (1900 <= y <= 2100):
                continue
            try:
                return date(y, mon, int(a))
            except ValueError:
                continue
    return None


@dataclass
class BarcodeResult:
    status: str                        # match | mismatch | unreadable | partial
    fields: list[dict] = field(default_factory=list)
    fields_found: list[str] = field(default_factory=list)
    has_expiry: bool = False
    detections: list[Detection] = field(default_factory=list)
    barcode_dates: dict[str, date] = field(default_factory=dict)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9/ ]", " ", s)).strip()


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _parse_date(s: str) -> Optional[date]:
    m = re.fullmatch(r"\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\s*", s)
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def parse_barcode_payload(payload: str) -> dict:
    """Clasifica por patrón los campos de un payload tipo 'a@b@c@...'. Devuelve dict de campos hallados."""
    tokens = [t.strip() for t in re.split(r"[@\n\r]", payload) if t.strip()]
    out: dict = {}
    dates, texts, singles = [], [], []
    for t in tokens:
        d = _parse_date(t)
        if d:
            dates.append(d)
        elif re.fullmatch(r"\d{11}", t):
            out.setdefault("tramite", t)
        elif re.fullmatch(r"\d{7,8}", t):
            out.setdefault("doc_number", t)
        elif re.fullmatch(r"[A-Za-z]", t):
            singles.append(t.upper())
        elif re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ' .\-]{2,}", t):
            texts.append(t)
    if texts:
        out["surname"] = texts[0]
    if len(texts) > 1:
        out["name"] = texts[1]
    for s in singles:           # sexo primero (M/F/X), luego ejemplar
        if "sex" not in out and s in "MFX":
            out["sex"] = s
        elif "copy" not in out:
            out["copy"] = s
    if dates:
        dates.sort()
        out["birth"] = dates[0]
        rest = dates[1:]
        if rest:
            out["issue"] = rest[0]
        if len(rest) > 1:
            out["expiry"] = rest[1]
    return out


def extract_ocr_dates(ocr_text: str) -> dict[str, date]:
    """Fechas del OCR asociadas a su etiqueta (nacimiento/emisión/vencimiento).

    Cada etiqueta solo "posee" el texto que hay entre ella y la SIGUIENTE etiqueta: si la fecha de emisión
    no se lee (p.ej. bajo un holograma), no se debe tomar la del vencimiento que viene después. Si dos
    etiquetas terminan con la misma fecha, o el orden nacimiento < emisión < vencimiento no se cumple,
    la asociación es ambigua y esas fechas se descartan (resultado no concluyente, no discrepancia).
    """
    text = norm_text(ocr_text)
    pos = sorted((text.find(label), key, label) for key, labels in LABELS.items()
                 for label in labels if text.find(label) >= 0)
    found: dict[str, date] = {}
    for n, (i, key, label) in enumerate(pos):
        end = pos[n + 1][0] if n + 1 < len(pos) else i + 80
        segment = text[i + len(label):min(end, i + len(label) + 80)]
        d = _find_date(segment)
        if d is not None:
            found[key] = d
    values = list(found.values())
    for k in list(found):          # la misma fecha en dos campos: ambigua
        if values.count(found[k]) > 1:
            found.pop(k, None)
    b, i_, e = found.get("birth"), found.get("issue"), found.get("expiry")
    if i_ and e and i_ >= e:
        found.pop("issue", None)
        found.pop("expiry", None)
    if b and found.get("issue") and b >= found["issue"]:
        found.pop("issue", None)
    return found


def _decode(img: np.ndarray) -> Optional[str]:
    if zxingcpp is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    variants = []
    for scale in (1.0, 0.75, 0.5, 1.5):
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale,
                                                  interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        variants.append(g)
        variants.append(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        variants.append(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10))
    fmt = getattr(zxingcpp.BarcodeFormat, "PDF417", None)
    for v in variants:
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            vv = v if rot is None else cv2.rotate(v, rot)
            try:
                res = zxingcpp.read_barcodes(vv, formats=fmt) if fmt else zxingcpp.read_barcodes(vv)
            except Exception:
                continue
            for r in res:
                if r.text:
                    return r.text
    return None


def _best_ratio(word: str, text: str) -> float:
    if word in text:
        return 1.0
    return max((SequenceMatcher(None, word, tok).ratio() for tok in text.split()), default=0.0)


def compare_fields(bc: dict, ocr_text: str, field_reads: dict | None = None) -> list[dict]:
    """Compara cada campo del barcode con lo impreso. status: match | mismatch | not_found (no concluyente).

    Con `field_reads` (OCR dirigido con votación) solo un campo leído de forma FIABLE puede dar `mismatch`.
    Sin él (OCR de página), un campo ausente o dudoso es siempre `not_found`: el OCR de una foto falla
    con frecuencia y eso no es evidencia de alteración. Fechas: con la etiqueta localizada se compara la
    fecha completa; con OCR por campo se compara el año (el día/mes suele quedar bajo el holograma).
    """
    text = norm_text(ocr_text)
    digits_text = _digits(ocr_text.replace(".", ""))
    ocr_dates = extract_ocr_dates(ocr_text)
    rows = []

    def row(name: str, ocr_val: str, bc_val: str, status: str, source: str = "page_ocr"):
        sens = name in SENSITIVE
        rows.append({"field": name,
                     "ocr_value": mask_value(ocr_val) if sens and ocr_val else ocr_val,
                     "barcode_value": mask_value(bc_val) if sens else bc_val,
                     "match": status == "match", "status": status, "source": source,
                     # solo una lectura dirigida y con votos puede sustentar una discrepancia fuerte
                     "reliable": source == "targeted_ocr"})

    def read(name):
        fr = (field_reads or {}).get(name)
        return fr if fr is not None and fr.reliable else None

    for name in ("doc_number", "tramite"):
        if name not in bc:
            continue
        v, fr = bc[name], read(name)
        if fr is not None:
            ok = fr.value == v or v in fr.value
            row(name, fr.value, v, "match" if ok else "mismatch", "targeted_ocr")
        elif v in digits_text:
            row(name, v, v, "match")
        else:
            row(name, "", v, "not_found")
    for key in ("birth", "issue", "expiry"):
        if key not in bc:
            continue
        d, fr, o = bc[key], read(key), ocr_dates.get(key)
        if fr is not None:   # la lectura dirigida (recorte + votos) prima sobre el OCR de página
            row(key, fr.value, str(d.year), "match" if fr.value == str(d.year) else "mismatch", "targeted_ocr")
        elif o is not None:
            row(key, o.isoformat(), d.isoformat(), "match" if o == d else "mismatch", "page_ocr")
        else:
            row(key, "", d.isoformat(), "not_found")
    for name in ("surname", "name"):
        if name not in bc:
            continue
        words = norm_text(bc[name]).split()
        fr = read(name)
        src = norm_text(fr.value) if fr is not None else text
        worst = min((_best_ratio(w, src) for w in words), default=0.0)
        if worst >= 0.8:
            row(name, bc[name], bc[name], "match")
        elif fr is not None and worst < 0.5:
            row(name, fr.value, bc[name], "mismatch", "targeted_ocr")
        else:
            row(name, "", bc[name], "not_found")
    return rows


def validate(img: np.ndarray, ocr_text: str, payload: Optional[str] = None,
             field_reads: Optional[dict] = None) -> BarcodeResult:
    """`payload` permite inyectar el contenido ya decodificado (tests)."""
    if payload is None:
        payload = _decode(img)
    if not payload:
        return BarcodeResult(
            status="unreadable",
            detections=[Detection("barcode_unreadable", "No se pudo decodificar el PDF417 (dato informativo, "
                                  "no es evidencia de fraude).", "info", 0.0)])
    bc = parse_barcode_payload(payload)
    found = sorted(bc.keys())
    rows = compare_fields(bc, ocr_text, field_reads)
    dets: list[Detection] = []
    reliable_mismatch = False
    for r in rows:
        if r["status"] == "mismatch":
            if r["reliable"]:
                reliable_mismatch = True
                dets.append(Detection(
                    "barcode_mismatch",
                    f"El campo '{r['field']}' del código de barras no coincide con el texto impreso "
                    f"(impreso: {r['ocr_value'] or 'n/d'}, barcode: {r['barcode_value']}).",
                    "high", 1.0, None, {"field": r["field"], "source": r["source"]}))
            else:
                # lectura de página completa: puede ser una mala asociación de etiqueta; indicio débil
                dets.append(Detection(
                    "barcode_mismatch",
                    f"Posible discrepancia en '{r['field']}' (impreso: {r['ocr_value'] or 'n/d'}, barcode: "
                    f"{r['barcode_value']}) leída con OCR de página completa: NO concluyente, verificar manualmente.",
                    "low", 0.2, None, {"field": r["field"], "source": r["source"]}))
    status = "mismatch" if reliable_mismatch else ("match" if any(r["match"] for r in rows) else "partial")
    return BarcodeResult(status, rows, found, "expiry" in bc, dets,
                         {k: bc[k] for k in ("birth", "issue", "expiry") if k in bc})
