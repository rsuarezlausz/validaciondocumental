"""Verificación de metadatos de imágenes (EXIF) y PDFs (info, versiones incrementales, hashes por página)."""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
from PIL.ExifTags import TAGS

EDITOR_HINTS = ("photoshop", "gimp", "acrobat", "illustrator", "canva", "pixlr", "paint",
                "foxit", "nitro", "sejda", "smallpdf", "ilovepdf", "pdf-xchange", "affinity", "lightroom")


_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]

# Peso de cada hallazgo de metadatos en la probabilidad (fuerte / medio / débil)
_STRONG = ("posterior", "edición", "incrementales")
_MEDIUM = ("no contiene metadatos", "re-exportad", "historial de edición", "cuantización")


def issue_weight(issue: str) -> float:
    if any(k in issue for k in _STRONG):
        return 0.25
    if any(k in issue for k in _MEDIUM):
        return 0.12
    return 0.05


def _is_libjpeg_standard(table: list[int]) -> bool:
    """True si la tabla coincide con la escala estándar libjpeg de algún quality (1..100)."""
    for q in range(1, 101):
        s = 5000 / q if q < 50 else 200 - 2 * q
        if all(max(1, min(255, (b * s + 50) // 100)) == t for b, t in zip(_STD_LUMA, table)):
            return True
    return False


def _parse_pdf_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", s)
    if not m:
        return None
    parts = [int(g) if g else d for g, d in zip(m.groups(), (1970, 1, 1, 0, 0, 0))]
    try:
        return datetime(*parts, tzinfo=timezone.utc)
    except ValueError:
        return None


def _editor(text: str) -> Optional[str]:
    t = (text or "").lower()
    return next((h for h in EDITOR_HINTS if h in t), None)


def merkle_root(leaves: list[str]) -> str:
    """Raíz Merkle SHA-256 de hashes hex (hashlib, sin dependencias externas)."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(h) for h in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def check_pdf(path: Path) -> tuple[list[str], str]:
    issues: list[str] = []
    raw = path.read_bytes()
    doc = fitz.open(stream=raw, filetype="pdf")
    md = doc.metadata or {}
    created, modified = _parse_pdf_date(md.get("creationDate")), _parse_pdf_date(md.get("modDate"))
    if created and modified and modified > created:
        issues.append(f"Fecha de modificación posterior a creación ({created:%Y-%m-%d %H:%M} -> {modified:%Y-%m-%d %H:%M})")
    if not created:
        issues.append("El PDF no contiene fecha de creación")
    for field in ("producer", "creator"):
        ed = _editor(md.get(field, ""))
        if ed:
            issues.append(f"Software de edición detectado en '{field}': {md.get(field)}")
    prod, crea = md.get("producer", ""), md.get("creator", "")
    if prod and crea and prod != crea and not _editor(prod + crea):
        issues.append(f"Productor ({prod}) distinto del creador ({crea})")
    if raw.count(b"%%EOF") > 1:
        issues.append(f"Actualizaciones incrementales detectadas ({raw.count(b'%%EOF')} marcadores %%EOF): el documento fue guardado varias veces")
    if doc.is_encrypted:
        issues.append("El PDF está cifrado")
    leaves = []
    for page in doc:
        h = hashlib.sha256()
        for xref in page.get_contents():
            h.update(doc.xref_stream(xref) or b"")
        leaves.append(h.hexdigest())
    return issues, merkle_root(leaves)


def check_image(path: Path) -> tuple[list[str], str]:
    issues: list[str] = []
    quant = None
    with Image.open(path) as im:
        quant = getattr(im, "quantization", None) if im.format == "JPEG" else None
        quant = quant.get(0) if quant else None
        exif = {TAGS.get(k, k): v for k, v in (im.getexif() or {}).items()}
        try:
            exif.update({TAGS.get(k, k): v for k, v in (im.getexif().get_ifd(0x8769) or {}).items()})
        except Exception:
            pass

    sw = str(exif.get("Software", ""))
    if _editor(sw):
        issues.append(f"Software de edición detectado en EXIF: {sw}")
    elif sw:
        issues.append(f"Software registrado en EXIF: {sw}")
    dt, orig = exif.get("DateTime"), exif.get("DateTimeOriginal")
    if dt and orig and str(dt) != str(orig):
        issues.append(f"Fecha de modificación ({dt}) distinta de la captura original ({orig})")
    if not exif:
        issues.append("La imagen no contiene metadatos EXIF (posible eliminación o re-exportación)")
    raw = path.read_bytes()
    low = raw.lower()
    if b"photoshop" in low or b"xmpmm:history" in low:
        issues.append("Historial de edición (XMP/Photoshop) presente en el archivo")
    if quant and not exif.get("Make") and _is_libjpeg_standard(list(quant)[:64]):
        issues.append("Tablas de cuantización estándar sin datos de cámara: imagen re-exportada por software")
    return issues, hashlib.sha256(raw).hexdigest()
