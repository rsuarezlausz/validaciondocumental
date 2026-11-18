"""Pipeline: cargar -> OCR -> forense -> persistir -> borrar original."""
import json
import logging
import time
from pathlib import Path

import cv2
import fitz
import numpy as np

from .. import db, forensic_analyzer as fa, metadata_checker as mc, ocr_engine
from ..config import LOG_DIR, MAX_PAGES, PAGES_DIR, PDF_RENDER_DPI, UPLOAD_DIR
from ..ingest import loader as ingest
from ..forensics import analyzer as local_analyzer, profile as doc_profile, score_fusion, template as fx_template
from ..forensics.types import Detection
from ..models import (AnalysisResult, Finding, ForensicAnalysis, PageInfo, Paragraph, Region)

audit = logging.getLogger("audit")
if not audit.handlers:
    h = logging.FileHandler(LOG_DIR / "analysis.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    audit.addHandler(h)
    audit.setLevel(logging.INFO)

log = logging.getLogger("pipeline")

MAX_DIM = 8000
MIN_IMAGE_FRACTION = 0.05   # una imagen embebida cuenta como "documento" si ocupa >= 5% de la página
MIN_IMAGE_SIDE = 200
FULL_PAGE_COVERAGE = 0.85   # fracción de la página que debe cubrir una imagen embebida para usarla como página


def find_upload(doc_id: str) -> Path | None:
    return next(iter(UPLOAD_DIR.glob(f"{doc_id}.*")), None)


class PageSource:
    """Cómo se obtuvo la imagen de cada página, para decidir qué detectores son válidos."""
    PHOTO = "raster"                    # imagen subida directamente (JPG/PNG/BMP)
    PDF_IMAGE = "pdf_embedded_image"    # imagen embebida a resolución nativa dentro de un PDF escaneado
    PDF_VECTOR = "pdf_vector"           # página rasterizada de un PDF con texto/gráficos vectoriales


def _decode_embedded(doc: "fitz.Document", xref: int) -> tuple[np.ndarray, bool] | None:
    """Imagen embebida -> (BGR, con_pérdida). None si no se puede decodificar de forma fiable."""
    try:
        info = doc.extract_image(xref)
    except Exception:
        return None
    arr = None
    ext = (info.get("ext") or "").lower()
    if info.get("image"):
        arr = cv2.imdecode(np.frombuffer(info["image"], np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            arr = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if pix.n == 1 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    if arr is None or max(arr.shape[:2]) > MAX_DIM:
        return None
    return arr, ext in ("jpeg", "jpg")


def _full_page_image(doc: "fitz.Document", page: "fitz.Page") -> tuple[np.ndarray, bool] | None:
    """Si la página es esencialmente una imagen (PDF escaneado / foto convertida a PDF), devuelve esa imagen en su
    resolución nativa: conserva las estadísticas de compresión y el ruido del original."""
    page_area = max(page.rect.get_area(), 1.0)
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        if not rects or sum(r.get_area() for r in rects) < FULL_PAGE_COVERAGE * page_area:
            continue
        got = _decode_embedded(doc, xref)
        if got is None:
            continue
        img, lossy = got
        r = rects[0]
        # proporciones incompatibles = imagen rotada o recortada por la página: se usa la rasterización
        if abs((img.shape[1] / img.shape[0]) / max(r.width / max(r.height, 1e-6), 1e-6) - 1.0) > 0.15:
            continue
        return img, lossy
    return None


def _meaningful_text(txt: str) -> bool:
    """Capa de texto útil: >= 20 caracteres alfanuméricos. (Un PDF de Word con solo imágenes trae una capa de
    espacios y saltos de línea que NO es texto: contarla hacía que se omitiera el OCR.)"""
    return sum(ch.isalnum() for ch in txt or "") >= 20


def _large_images(doc: "fitz.Document", page: "fitz.Page") -> list[tuple[np.ndarray, bool]]:
    """Imágenes embebidas grandes de una página (p.ej. frente y dorso de un documento), de arriba abajo."""
    page_area = max(page.rect.get_area(), 1.0)
    found, seen = [], set()
    for info in page.get_images(full=True):
        xref = info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        if not rects or rects[0].get_area() < MIN_IMAGE_FRACTION * page_area:
            continue
        got = _decode_embedded(doc, xref)
        if got is None or min(got[0].shape[:2]) < MIN_IMAGE_SIDE:
            continue
        found.append((rects[0].y0, rects[0].x0, got))
    return [g for _, _, g in sorted(found, key=lambda t: (t[0], t[1]))]


def _render_pdf(path: Path) -> tuple[list[np.ndarray], list[str], list[tuple[str, bool]]]:
    """Devuelve imágenes BGR, texto de la capa de texto y (origen, con_pérdida) por página."""
    images, texts, kinds = [], [], []
    doc = fitz.open(path)
    try:
        if doc.is_encrypted:
            raise ValueError("PDF cifrado o protegido por contraseña: no analizable")
        if doc.page_count > MAX_PAGES:
            raise ValueError(f"El PDF tiene {doc.page_count} páginas (máximo {MAX_PAGES})")
        for page in doc:
            native = _full_page_image(doc, page)
            text = page.get_text().strip()
            collage = [] if (native is not None or _meaningful_text(text)) else _large_images(doc, page)
            if native is not None:
                images.append(native[0])
                kinds.append((PageSource.PDF_IMAGE, native[1]))
                texts.append(text)
                continue
            if collage:
                # página de solo imágenes (p.ej. Word con frente y dorso del documento): cada imagen se analiza
                # por separado y a su resolución nativa, no la página A4 rasterizada con el documento diminuto
                for img, lossy in collage:
                    images.append(img)
                    kinds.append((PageSource.PDF_IMAGE, lossy))
                    texts.append("")
                continue
            else:
                pix = page.get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                kinds.append((PageSource.PDF_VECTOR, False))
            texts.append(text)
        if len(images) > MAX_PAGES:
            raise ValueError(f"El PDF contiene {len(images)} imágenes/páginas a analizar (máximo {MAX_PAGES})")
    finally:
        doc.close()
    return images, texts, kinds


def _save_pages(images: list[np.ndarray], doc_id: str) -> None:
    out = PAGES_DIR / doc_id
    out.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images, 1):
        if not cv2.imwrite(str(out / f"page-{i}.png"), im):
            raise IOError(f"No se pudo guardar la vista previa de la página {i}")


def _to_finding(d: Detection, page_num: int = 1) -> Finding:
    region = None
    if d.bbox:
        x, y, w, h = d.bbox
        region = Region(x=int(x), y=int(y), width=int(w), height=int(h), page=page_num)
    extra = [Region(x=r[0], y=r[1], width=r[2], height=r[3], page=page_num)
             for r in (d.extra.get("region_a"), d.extra.get("region_b")) if r]
    return Finding(type=d.type, region=region, regions=extra, description=d.description,
                   score=round(float(d.score), 3), severity=d.severity)


def _capabilities(any_lossy: bool, photo_pages: bool, copy_move_ok: bool, vector_only: bool) -> dict:
    """Capacidades REALES de este análisis (qué detectores corrieron y por qué se omitieron los demás)."""
    ran, skipped = ["noise", "metadata"], []
    if any_lossy:
        ran += ["ela", "jpeg_ghost"]
    else:
        skipped += [{"detector": "ela", "reason": "lossless_or_rasterized_page"},
                    {"detector": "jpeg_ghost", "reason": "lossless_or_rasterized_page"}]
    local = ["flat_patch", "seam", "local_consistency", "geometry", "barcode", "rules"]
    if photo_pages:
        ran += local
        if copy_move_ok:
            ran.append("copy_move")
        else:
            skipped.append({"detector": "copy_move", "reason": "no_sensor_noise_or_low_resolution_text"})
    else:
        skipped += [{"detector": d, "reason": "digital_pdf_page"} for d in local + ["copy_move"]]
    return {"ran": ran, "skipped": skipped}


def run_analysis(doc_id: str) -> None:
    t0 = time.perf_counter()
    path = find_upload(doc_id)
    base = db.get_result(doc_id)
    try:
        if path is None or base is None:
            raise FileNotFoundError("Archivo no encontrado (ya procesado o eliminado)")
        data = path.read_bytes()
        is_pdf = ingest.sniff_format(data) == "pdf"
        input_result = None if is_pdf else ingest.load(data)   # el PDF se procesa en _render_pdf (evita rasterizar dos veces)
        if input_result is not None and input_result.errors:
            raise ValueError(input_result.errors[0])

        if is_pdf:
            images, texts, kinds = _render_pdf(path)
            _save_pages(images, doc_id)
        else:
            images = [p.image for p in input_result.pages]
            texts = [""] * len(images)
            lossy_fmt = input_result.format == "jpeg"
            kinds = [(PageSource.PHOTO, lossy_fmt)] * len(images)
            _save_pages(images, doc_id)   # la vista previa se sirve desde estos PNG (mismas coordenadas que el análisis)

        paragraphs: list[Paragraph] = []
        ocr_failed = False
        for i, (img, txt) in enumerate(zip(images, texts), 1):
            if _meaningful_text(txt):
                paragraphs += ocr_engine.text_to_paragraphs(txt, i)
            else:
                try:
                    paragraphs += ocr_engine.ocr_page(img, i)
                except Exception as exc:  # noqa: BLE001  - sin OCR el análisis forense sigue siendo válido
                    log.warning("OCR no disponible en la página %s: %s", i, type(exc).__name__)
                    ocr_failed = True

        ocr_text = "\n\n".join(p.text for p in paragraphs)
        prof = doc_profile.detect(ocr_text)      # plantilla y reglas específicas solo si el OCR identifica el documento

        findings: list[Finding] = []
        scores, signals = [], []
        for i, (img, (src, lossy)) in enumerate(zip(images, kinds), 1):
            s, f, sig = fa.analyze_image(img, i, fx_template.excluded_mask(img) if prof.uses_template else None,
                                         use_compression=lossy)
            scores.append(s); signals.append(sig); findings += f
        for f in findings:
            f.severity = "medium" if f.score >= 0.7 else "low"

        any_lossy = any(lossy for _, lossy in kinds)
        photo_idx = [i for i, (src, _) in enumerate(kinds) if src in (PageSource.PHOTO, PageSource.PDF_IMAGE)]
        if is_pdf:
            kind = ("jpeg" if any_lossy else "png") if photo_idx and len(photo_idx) == len(kinds) else "pdf"
        else:
            kind = "jpeg" if input_result.format == "jpeg" else "png"

        issues, integrity = mc.check_pdf(path) if is_pdf else mc.check_image(path)
        meta_weight = sum(mc.issue_weight(m) for m in issues)
        findings += [Finding(type="metadata", description=m, score=0.5, severity="low") for m in issues]

        # Señales locales de fotografía (glifos, costuras, barcode, parches planos): solo páginas que SON una imagen.
        # Una página vectorial renderizada tiene glifos idénticos por construcción y fondos digitales perfectos.
        all_local_detections: list[Detection] = []
        local = local_analyzer.LocalReport()
        first = True
        for page_idx in photo_idx:
            page_local = local_analyzer.run(images[page_idx], ocr_text, profile=prof.name)
            all_local_detections += page_local.detections
            findings += [_to_finding(d, page_idx + 1) for d in page_local.detections]
            if first:
                local, first = page_local, False

        pixel_scores: dict[str, float] = {}
        for f in findings:
            key = {"ela_inconsistency": "ela", "noise_inconsistency": "noise", "jpeg_ghost": "jpeg_ghost"}.get(f.type)
            if key:
                pixel_scores[key] = max(pixel_scores.get(key, 0.0), f.score)

        fusion = score_fusion.fuse(
            all_local_detections, pixel_scores, meta_weight, kind, float(np.mean(signals)) if signals else 0.2,
            {"barcode": local.barcode.status if local.barcode else "unreadable", "glyphs": local.glyph_count,
             "copy_move_applicable": local.copy_move_applicable if photo_idx else False})
        prob, conf, warnings, verdict = fusion.probability, fusion.confidence, fusion.warnings, fusion.verdict
        if is_pdf and not photo_idx:
            fusion.limitations.append("Página de PDF digital (vectorial): no se aplican señales de fotografía "
                                      "(glifos, costuras, código de barras, parches planos).")
        if is_pdf and photo_idx:
            fusion.limitations.append("PDF con imagen embebida: se analizó la imagen a su resolución nativa.")
        if ocr_failed:
            warnings = list(warnings) + ["OCR no disponible: no hay texto extraído ni validación cruzada con el código de barras."]
            fusion.limitations.append("Tesseract no está disponible: se omitió el OCR y las señales que dependen de él.")

        caps = _capabilities(any_lossy, bool(photo_idx), local.copy_move_applicable, not photo_idx)
        pages = [PageInfo(page=i + 1, width=im.shape[1], height=im.shape[0],
                          url=f"/api/v1/documents/{doc_id}/pages/{i + 1}") for i, im in enumerate(images)]
        elapsed = round(time.perf_counter() - t0, 2)
        result = AnalysisResult(
            document_id=doc_id, filename=base.filename, status="completed", uploaded_at=base.uploaded_at,
            ocr_text=ocr_text, paragraphs=paragraphs,
            forensic_analysis=ForensicAnalysis(
                tampering_probability=prob, confidence=conf, findings=findings,
                metadata_issues=issues, warnings=warnings, integrity_hash=integrity,
                signals={**local.signals, "page_sources": [k[0] for k in kinds], "document_profile": prof.name,
                         "timings": local.timings}, limitations=fusion.limitations,
                contributions=fusion.contributions, analysis_capabilities=caps, verdict=verdict),
            pages=pages, processing_seconds=elapsed,
        )
        db.save_result(result)
        audit.info(json.dumps({"document_id": doc_id, "status": "completed", "verdict": verdict,
                               "tampering_probability": prob, "confidence": conf, "seconds": elapsed}))
    except Exception as exc:  # noqa: BLE001
        log.exception("Fallo el análisis de %s", doc_id)
        db.set_status(doc_id, "failed", str(exc))
        audit.info(json.dumps({"document_id": doc_id, "status": "failed", "error": str(exc)}))
    finally:
        if path is not None:
            path.unlink(missing_ok=True)  # el original no se conserva
