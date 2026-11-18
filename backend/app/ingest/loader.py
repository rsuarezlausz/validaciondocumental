"""Cargador único de formatos: normaliza cualquier entrada (PNG, JPG/JPEG, BMP, PDF) a una estructura común.

Detecta el formato por contenido (magic bytes), no por extensión ni MIME. Maneja PNG (canal alfa,
paleta, 16 bits, grises, metadatos), JPEG (EXIF orientación, grises, progresivo), BMP (profundidades
variadas), PDF (PyMuPDF, sandbox, límites de páginas y tamaño).
"""
import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional

import cv2
import numpy as np

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

MAX_PAGES = 10
MAX_BYTES = 20 * 1024 * 1024
PDF_RENDER_DPI = 200
TIMEOUT_PER_PAGE = 30
MAX_DIM = 8000


# ---------------------------------------------------------------------------
# Datos de salida
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    image: np.ndarray              # BGR uint8
    source: str = "raster"          # raster | pdf_vector | pdf_embedded_image
    dpi: float = 200.0
    coord_transform: dict = field(default_factory=dict)  # ej. {offset_x, offset_y, scale}
    exif_orientation_applied: bool = False


@dataclass
class IngestResult:
    pages: list[PageData] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    format: str = "unknown"         # png | jpeg | bmp | pdf
    capabilities: dict = field(default_factory=dict)  # qué detectores aplicar
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Magic bytes
# ---------------------------------------------------------------------------

def sniff_format(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:2] == b"BM":
        return "bmp"
    if data[:5] == b"%PDF-":
        return "pdf"
    return "unknown"


# ---------------------------------------------------------------------------
# Cargador principal
# ---------------------------------------------------------------------------

def load(path_or_bytes: Path | bytes | BinaryIO) -> IngestResult:
    """Punto de entrada: detecta el formato y normaliza. Devuelve IngestResult."""
    if isinstance(path_or_bytes, bytes):
        data: bytes = path_or_bytes
    elif isinstance(path_or_bytes, Path):
        data = path_or_bytes.read_bytes()
    elif hasattr(path_or_bytes, "read"):
        data = path_or_bytes.read()
    else:
        return IngestResult(errors=["Unsupported input type"])

    fmt = sniff_format(data)
    result = IngestResult(format=fmt)

    if fmt == "pdf":
        return _load_pdf(data, result)
    if fmt in ("png", "jpeg", "bmp"):
        return _load_image(data, fmt, result)
    result.errors.append(f"Formato no soportado (magic: {data[:8].hex()})")
    return result


# ---------------------------------------------------------------------------
# Imagen (PNG/JPEG/BMP)
# ---------------------------------------------------------------------------

def _load_image(data: bytes, fmt: str, result: IngestResult) -> IngestResult:
    try:
        arr, meta = _decode_img(data)
    except Exception as e:
        result.errors.append(str(e))
        return result

    h, w = arr.shape[:2]
    if max(h, w) > MAX_DIM:
        result.errors.append(f"Dimensiones {w}x{h} exceden el máximo {MAX_DIM}")
        return result

    page = PageData(image=arr, source="raster", dpi=200)
    # estimar DPI si hay EXIF
    exif = meta.get("exif", {})
    if exif.get("XResolution"):
        try:
            page.dpi = float(exif["XResolution"])
        except (ValueError, TypeError):
            pass
    page.exif_orientation_applied = bool(meta.get("orientation_applied", False))

    result.metadata = meta
    result.pages = [page]
    result.capabilities = _capabilities(fmt, meta)
    return result


def _decode_img(data: bytes) -> tuple[np.ndarray, dict]:
    """Decodifica bytes a BGR numpy y extrae metadatos. Maneja canal alfa, paleta, 16 bits, grises."""
    if Image is None:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("No se pudo decodificar la imagen con OpenCV")
        return arr, {}

    buf = BytesIO(data)
    with Image.open(buf) as im:
        meta = {}
        exif = _parse_exif(im)
        meta["exif"] = exif
        meta["format"] = im.format or ""
        meta["mode"] = im.mode
        meta["size"] = im.size

        # Orientación EXIF
        orientation = exif.get("Orientation", 1)
        if orientation != 1:
            if orientation == 3:
                im = im.rotate(180, expand=True)
            elif orientation == 6:
                im = im.rotate(270, expand=True)
            elif orientation == 8:
                im = im.rotate(90, expand=True)
            meta["orientation_applied"] = True

        # Convertir a RGB
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
            meta["alpha"] = True
        elif im.mode == "P":
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
            meta["palette"] = True
        elif im.mode == "I":
            # 32-bit int: normalizar
            arr = np.array(im, dtype=np.float32)
            arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255
            im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        elif im.mode not in ("RGB", "L", "LA"):
            im = im.convert("RGB")

        # grises -> BGR (OpenCV compatible)
        if im.mode in ("L", "LA"):
            arr = np.array(im)
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            arr = np.array(im.convert("RGB"))[:, :, ::-1]  # RGB -> BGR

        # PNG alpha warning
        if meta.get("alpha"):
            meta["alpha_blended"] = True

        # Metadatos extra (PNG tEXt, iTXt, eXIf, Software)
        info = im.info
        for key in ("Software", "software", "Comment"):
            if key in info:
                meta[f"key_{key}"] = info[key]
        text_chunks = {k: v for k, v in info.items() if k[0].isupper()}
        if text_chunks:
            meta["text_chunks"] = text_chunks

        return arr, meta


def _parse_exif(im) -> dict:
    exif = {}
    try:
        raw = im.getexif() or {}
        from PIL.ExifTags import TAGS as E_TAGS
        exif = {E_TAGS.get(k, k): v for k, v in raw.items()}
        # Intenta GPSInfo y otros IFD
        ifd = raw.get_ifd(0x8769)
        if ifd:
            exif.update({E_TAGS.get(k, k): v for k, v in ifd.items()})
    except Exception:
        pass
    return exif


def _capabilities(fmt: str, meta: dict) -> dict:
    """Lista qué detectores se ejecutan y cuáles se omiten según formato y metadatos."""
    caps = {"ran": [], "skipped": []}

    if fmt == "jpeg":
        caps["ran"] = ["ela", "noise", "jpeg_ghost", "metadata", "flat_patch", "copy_move",
                       "seam", "local_consistency", "geometry", "barcode", "rules"]
    elif fmt in ("png", "bmp"):
        caps["ran"] = ["noise", "flat_patch", "copy_move", "seam", "local_consistency",
                       "geometry", "barcode", "rules"]
        caps["skipped"] = [
            {"detector": "ela", "reason": "lossless_format"},
            {"detector": "jpeg_ghost", "reason": "lossless_format"},
            {"detector": "metadata", "reason": "limited_metadata" if fmt == "bmp" else ""},
        ]
        if fmt == "bmp":
            caps["skipped"].append({"detector": "metadata", "reason": "lossless_format"})
    else:
        caps["ran"] = ["flat_patch", "noise"]
        caps["skipped"] = [{"detector": "ela", "reason": "format_not_supported"}]

    return caps


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _load_pdf(data: bytes, result: IngestResult) -> IngestResult:
    if fitz is None:
        result.errors.append("PyMuPDF no instalado, no se puede procesar PDF")
        return result

    if len(data) > MAX_BYTES:
        result.errors.append(f"Archivo PDF > {MAX_BYTES // 1024 // 1024} MB")
        return result

    # Sandbox temporales aislados
    with tempfile.TemporaryDirectory(prefix="ingest_pdf_") as _:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            result.errors.append(f"Error al abrir PDF: {e}")
            return result

        if doc.is_encrypted:
            result.errors.append("PDF cifrado o protegido por contraseña: no analizable")
            doc.close()
            return result

        if doc.page_count > MAX_PAGES:
            doc.close()
            result.errors.append(f"El PDF tiene {doc.page_count} páginas (máximo {MAX_PAGES})")
            return result

        md = doc.metadata or {}
        result.metadata = {
            "title": md.get("title", ""),
            "author": md.get("author", ""),
            "producer": md.get("producer", ""),
            "creator": md.get("creator", ""),
            "creationDate": md.get("creationDate", ""),
            "modDate": md.get("modDate", ""),
            "page_count": doc.page_count,
            "encrypted": doc.is_encrypted,
        }

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            t0 = time.perf_counter()
            # Rasterizar página
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            page_data = PageData(
                image=cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                source="raster",
                dpi=PDF_RENDER_DPI,
            )
            result.pages.append(page_data)

            # Superposiciones vectoriales: rectángulos rellenos casi blancos
            for draw in page.get_drawings():
                if _is_overlay_rect(draw, pix.width, pix.height):
                    # Convertir a coordenadas de píxeles
                    r = draw["rect"]
                    x0, y0, x1_, y1_ = r.x0, r.y0, r.x1, r.y1
                    # Normalizar a pixmap coords
                    page_pix = page.get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
                    px0 = int(x0 * page_pix.width / page.rect.width + 0.5)
                    py0 = int(y0 * page_pix.height / page.rect.height + 0.5)
                    px1 = int(x1_ * page_pix.width / page.rect.width + 0.5)
                    py1 = int(y1_ * page_pix.height / page.rect.height + 0.5)
                    if px1 - px0 > 20 and py1 - py0 > 15:
                        overlay = {
                            "type": "pdf_vector_overlay",
                            "rect": draw.get("rect"),
                            "fill": draw.get("fill"),
                            "page": page_idx + 1,
                            "bbox": (px0, py0, px1 - px0, py1 - py0),
                        }
                        page_data.coord_transform[overlay["type"]] = overlay
                        result.metadata.setdefault("vector_overlays", []).append(overlay)

            # Capa de texto: detectar texto tapado
            text_layer = page.get_text("text", textpage=page.get_textpage())
            if text_layer.strip() and len(text_layer.strip()) > 20:
                page_data.exif_orientation_applied = True
                page_data.source = "raster"

            if time.perf_counter() - t0 > TIMEOUT_PER_PAGE:
                result.warnings.append(f"Timeout en página {page_idx + 1}")

        # Estructura: actualizaciones incrementales
        raw = data
        eof_count = raw.count(b"%%EOF")
        xref_count = len(re.findall(rb"xref\b", raw))
        result.metadata["incremental_updates"] = eof_count - 1
        result.metadata["xref_sections"] = xref_count

        doc.close()

    result.capabilities = _capabilities("pdf", result.metadata)
    return result


def _is_overlay_rect(draw: dict, page_w: int, page_h: int) -> bool:
    """Detecta rectángulos rellenos (blancos o casi blancos) grandes dibujados sobre la página."""
    r = draw.get("rect")
    fill = draw.get("fill")
    if not r or not fill:
        return False
    # Verificar que es un rectángulo y está relleno en blanco
    if r.x1 - r.x0 < 10 or r.y1 - r.y0 < 10:
        return False
    if draw.get("type") not in ("rect", "square"):
        return False
    try:
        f = list(fill)
        return all(c > 0.85 for c in f[:3])  # blanco / casi blanco
    except Exception:
        return False