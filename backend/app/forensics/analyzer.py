"""Orquesta las señales locales sobre una imagen fotografiada (no PDF) y devuelve detecciones + señales."""
import time
from dataclasses import dataclass, field
from datetime import date

import cv2
import numpy as np

from . import background_continuity, barcode_validator, field_ocr, flat_patch_detector, glyph_copy_move
from . import local_consistency, patch_content, rules, seam_detector, settings, text_geometry
from . import profile as doc_profile
from . import template as tpl
from .glyphs import segment_glyphs
from .types import Detection

MIN_GLYPH_HEIGHT = 20          # texto más pequeño no tiene píxeles de fondo suficientes para comparar ruido
MIN_NOISE_FOR_COPY_MOVE = 0.8   # sigma del residuo en zonas planas por debajo: render digital o recompresión fuerte


@dataclass
class LocalReport:
    detections: list[Detection] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    barcode: barcode_validator.BarcodeResult | None = None
    glyph_count: int = 0
    copy_move_applicable: bool = True
    timings: dict = field(default_factory=dict)
    profile: str = "generic"


def run(img: np.ndarray, ocr_text: str = "", payload: str | None = None,
        profile: str | None = None) -> LocalReport:
    """`profile`: nombre explícito ("ar_dni", "generic") o None para detectarlo del texto OCR."""
    prof = doc_profile.get(profile) if profile else doc_profile.detect(ocr_text)
    rep = LocalReport()
    rep.profile = prof.name

    def mask(names):    # zonas de plantilla solo si el documento fue identificado con ese perfil
        return tpl.excluded_mask(img, names) if prof.uses_template else None

    t = time.perf_counter()
    reads = field_ocr.read_fields(img) if prof.uses_template else None
    rep.timings["field_ocr"] = round(time.perf_counter() - t, 2)
    bc = barcode_validator.validate(img, ocr_text, payload, reads)
    rep.barcode = bc
    rep.timings["barcode"] = round(time.perf_counter() - t, 2)
    rep.detections += bc.detections

    t = time.perf_counter()
    hologram_photo = mask(("photo", "hologram", "barcode"))
    glyphs, ink = segment_glyphs(img, exclude=hologram_photo)
    rep.glyph_count = len(glyphs)
    rep.timings["segment"] = round(time.perf_counter() - t, 2)

    noise = glyph_copy_move.estimate_noise(img)
    med_h = float(np.median([g.h for g in glyphs])) if glyphs else 0.0
    rep.copy_move_applicable = noise >= MIN_NOISE_FOR_COPY_MOVE and med_h >= MIN_GLYPH_HEIGHT

    t = time.perf_counter()
    cm = glyph_copy_move.analyze(img, glyphs, ink) if rep.copy_move_applicable else glyph_copy_move.CopyMoveResult()
    rep.timings["copy_move"] = round(time.perf_counter() - t, 2)
    rep.detections += cm.detections

    t = time.perf_counter()
    # Parches planos: se aplican siempre (funciona sobre cualquier imagen), sin necesidad de glifos
    excl = mask(("photo", "hologram", "barcode", "guilloche_band", "border"))
    flat = flat_patch_detector.analyze(img, excl)
    rep.detections += flat.detections
    rep.timings["flat_patch"] = round(time.perf_counter() - t, 2)
    rep.signals["flat_patch"] = {"patches_evaluated": flat.patches_evaluated, "max_score": round(flat.max_score, 3)}

    # Continuidad del fondo: refuerza flat_patch
    gray_f = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    bg = background_continuity.analyze(gray_f, flat.detections)
    rep.detections += bg.detections

    # Contenido del parche: OCR interno (requiere Tesseract)
    pc = patch_content.analyze(img, flat.detections)
    rep.detections += pc.detections

    t = time.perf_counter()
    seam = seam_detector.analyze(img, glyphs, ink) if rep.copy_move_applicable else seam_detector.SeamResult()
    rep.detections += seam.detections
    loc = local_consistency.analyze(img, glyphs, ink)
    rep.detections += loc.detections
    geo = text_geometry.analyze(glyphs)
    rep.detections += geo.detections
    rep.timings["seam_local_geometry"] = round(time.perf_counter() - t, 2)

    # reglas: fechas del OCR; el barcode complementa (y prima en nacimiento/emisión)
    dates: dict[str, date] = dict(barcode_validator.extract_ocr_dates(ocr_text))
    dates.update(bc.barcode_dates)
    exp_year = None
    if reads and "expiry" not in dates and reads.get("expiry") is not None and reads["expiry"].reliable:
        exp_year = int(reads["expiry"].value)
    rep.detections += rules.evaluate(dates, exp_year, check_validity=prof.check_validity)

    rep.signals = {
        "barcode_crosscheck": {"status": bc.status, "fields": bc.fields, "fields_found": bc.fields_found,
                               "has_expiry": bc.has_expiry},
        "glyph_copy_move": {"max_shape_ncc": cm.max_shape_ncc, "max_residual_ncc": cm.max_residual_ncc,
                            "pairs_evaluated": cm.pairs_evaluated, "applicable": rep.copy_move_applicable},
        "seam": {"detected": seam.detected},
        "local_consistency": {"outliers": loc.outliers},
        "text_geometry": {"pitch_median_px": round(geo.pitch_median, 1), "anomalies": len(geo.detections)},
        "glyphs_segmented": rep.glyph_count,
        "document_profile": prof.name,
    }
    return rep
