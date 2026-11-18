import numpy as np
import cv2
import pytest

import synth
from app.forensics import (barcode_validator as bv, glyph_copy_move, glyphs as G, local_consistency, rules,
                           score_fusion, seam_detector, template, text_geometry)
from app.forensics import analyzer
from app.forensics.types import Detection, mask_value

PAYLOAD = "00412747395@TESTER SURNAME@DEMO NAME@M@12345678@B@28/12/1977@18/12/2015"
OCR = ("REPUBLICA ARGENTINA Apellido TESTER SURNAME Nombre DEMO NAME Sexo M Ejemplar B "
       "Fecha de nacimiento 28 DIC/ DEC 1977 Fecha de emisión 18 DIC/ DEC 2015 "
       "Fecha de vencimiento 18 DIC/ DEC 2030 Documento 12.345.678 Trámite 00412747395 7006")


@pytest.fixture(scope="module")
def card():
    img, _ = synth.make_card()
    glyphs, ink = G.segment_glyphs(img)
    return img, glyphs, ink


# ---------- barcode ----------
def test_barcode_payload_is_parsed_by_pattern_not_by_position():
    f = bv.parse_barcode_payload(PAYLOAD)
    assert f["doc_number"] == "12345678" and f["tramite"] == "00412747395"
    assert f["sex"] == "M" and f["copy"] == "B"
    assert str(f["birth"]) == "1977-12-28" and str(f["issue"]) == "2015-12-18"
    assert "expiry" not in f     # sin vencimiento en el barcode: NO se inventa


def test_barcode_matches_ocr_and_masks_sensitive_values():
    r = bv.validate(np.zeros((50, 50, 3), np.uint8), OCR, payload=PAYLOAD)
    assert r.status == "match" and not r.has_expiry and not r.detections
    vals = {x["field"]: x for x in r.fields}
    assert vals["doc_number"]["barcode_value"] == "***678"
    assert "12345678" not in str(r.fields)


def test_barcode_mismatch_is_high_severity():
    bad = OCR.replace("1977", "1987")
    r = bv.validate(np.zeros((50, 50, 3), np.uint8), bad, payload=PAYLOAD)
    # OCR de página completa: puede ser una mala asociación de etiqueta -> indicio débil, no concluyente
    d = r.detections[0]
    assert d.type == "barcode_mismatch" and d.severity == "low" and r.status != "mismatch"

    class Read:     # lectura dirigida y fiable: sí sustenta una discrepancia fuerte
        reliable = True
        value = "1987"
    r = bv.validate(np.zeros((50, 50, 3), np.uint8), "", payload=PAYLOAD, field_reads={"birth": Read()})
    d = r.detections[0]
    assert r.status == "mismatch" and d.severity == "high"


def test_unreadable_barcode_is_informative_only():
    r = bv.validate(np.full((200, 300, 3), 255, np.uint8), OCR)
    assert r.status == "unreadable" and r.detections[0].severity == "info" and r.detections[0].score == 0


def test_not_found_ocr_field_is_not_a_mismatch():
    r = bv.validate(np.zeros((50, 50, 3), np.uint8), "texto ilegible", payload=PAYLOAD)
    assert r.status != "mismatch"


# ---------- copy-move ----------
def test_clean_document_has_no_copy_move(card):
    img, glyphs, ink = card
    res = glyph_copy_move.analyze(img, glyphs, ink)
    assert res.detections == []          # dígitos naturales repetidos NO deben marcarse
    assert res.max_shape_ncc > 0.95      # ...aunque su forma sea casi idéntica


def test_pasted_glyph_is_detected_between_the_two_glyphs(card):
    img, glyphs, ink = card
    line = sorted(G.lines_of(glyphs)[4], key=lambda g: g.x)
    src, dst = line[-2], line[-1]
    edited = synth.paste_region(img, src.bbox, (dst.x, dst.y))
    g2, ink2 = G.segment_glyphs(edited)
    res = glyph_copy_move.analyze(edited, g2, ink2)
    assert res.detections and res.detections[0].type == "glyph_copy_move"
    d = res.detections[0]
    assert d.severity == "high"
    ax, bx = d.extra["region_a"][0], d.extra["region_b"][0]
    assert {ax, bx} == {src.x, dst.x}


def test_copy_move_not_applicable_without_sensor_noise():
    img, _ = synth.make_card(noise=0.0, jpeg_q=98)
    assert glyph_copy_move.estimate_noise(img) < analyzer.MIN_NOISE_FOR_COPY_MOVE
    rep = analyzer.run(img, "")
    assert rep.copy_move_applicable is False and rep.signals["glyph_copy_move"]["applicable"] is False


# ---------- seam ----------
def test_seam_detected_in_text_field_but_not_on_clean(card):
    img, glyphs, ink = card
    assert not seam_detector.analyze(img, glyphs, ink).detected
    l = sorted(G.lines_of(glyphs)[0], key=lambda g: g.x)
    seamed = img.copy()
    sx = l[-1].x + l[-1].w + 20
    y0, y1 = l[0].y - 5, l[0].y + l[0].h + 30
    col = seamed[y0:y1, sx:sx + 1].astype(int) + 80
    seamed[y0:y1, sx:sx + 1] = np.clip(col, 0, 255).astype(np.uint8)
    res = seam_detector.analyze(seamed, glyphs, ink)
    assert res.detected and res.detections[0].type == "seam_artifact"
    x, y, w, h = res.detections[0].bbox
    assert x - 6 <= sx <= x + w + 6


# ---------- consistencia local y geometría ----------
def test_local_consistency_flags_blurred_character_only(card):
    img, glyphs, ink = card
    assert local_consistency.analyze(img, glyphs, ink).outliers == 0
    line = sorted(G.lines_of(glyphs)[4], key=lambda g: g.x)
    g = line[5]
    edited = img.copy()
    x0, y0, x1, y1 = g.x - 6, g.y - 6, g.x + g.w + 6, g.y + g.h + 6
    edited[y0:y1, x0:x1] = cv2.GaussianBlur(edited[y0:y1, x0:x1], (0, 0), 2.5)   # trazo borroso + sin ruido
    g2, ink2 = G.segment_glyphs(edited)
    res = local_consistency.analyze(edited, g2, ink2)
    assert res.outliers >= 1


def test_geometry_quiet_on_clean_and_flags_shifted_glyph(card):
    img, glyphs, ink = card
    assert text_geometry.analyze(glyphs).detections == []
    line = sorted(G.lines_of(glyphs)[4], key=lambda g: g.x)
    g = line[6]
    g.y += 9     # carácter desplazado respecto a la línea base
    res = text_geometry.analyze(glyphs)
    g.y -= 9
    assert any(d.bbox[0] == g.x for d in res.detections)


# ---------- reglas ----------
def test_rules_flag_expiry_inconsistent_with_assumed_validity():
    from datetime import date
    ok = rules.evaluate({"birth": date(1977, 12, 28), "issue": date(2015, 12, 18), "expiry": date(2030, 12, 18)})
    assert ok == []
    bad = rules.evaluate({"birth": date(1977, 12, 28), "issue": date(2015, 12, 18), "expiry": date(2033, 12, 18)})
    assert bad and bad[0].type == "rule_violation" and bad[0].extra["verified"] is False
    assert "por verificar" in bad[0].description.lower()


def test_rules_order_checks():
    from datetime import date
    out = rules.evaluate({"birth": date(2020, 1, 1), "issue": date(2015, 1, 1), "expiry": date(2010, 1, 1)})
    assert {d.extra["rule"] for d in out} >= {"issue_before_expiry", "birth_before_issue"}


def test_ocr_dates_extracted_by_label():
    from app.forensics.barcode_validator import extract_ocr_dates
    d = extract_ocr_dates(OCR)
    assert (d["birth"].year, d["issue"].year, d["expiry"].year) == (1977, 2015, 2030)


# ---------- plantilla ----------
def test_template_excludes_hologram_and_photo_for_cards_only():
    card_img = np.zeros((1000, 1600, 3), np.uint8)
    m = template.excluded_mask(card_img)
    assert m is not None and m[500, 250] and not m[500, 1400]       # foto sí, texto derecho no
    assert template.excluded_mask(np.zeros((1000, 800, 3), np.uint8)) is None


# ---------- fusión ----------
def test_barcode_mismatch_dominates():
    det = [Detection("barcode_mismatch", "x", "high", 1.0)]
    f = score_fusion.fuse(det, {}, 0.0, "jpeg", 1.0, {"barcode": "mismatch", "glyphs": 50})
    assert f.probability >= 0.85


def test_single_weak_signal_cannot_exceed_medium():
    f = score_fusion.fuse([], {"ela": 1.0, "noise": 1.0, "jpeg_ghost": 1.0}, 0.6, "jpeg", 1.0,
                          {"barcode": "match", "glyphs": 50})
    assert f.probability <= 0.45 + 1e-9


def test_missing_exif_or_resave_alone_never_raises_score_much():
    f = score_fusion.fuse([], {}, 0.37, "jpeg", 1.0, {"barcode": "match", "glyphs": 50})
    assert f.probability < 0.2


def test_confidence_drops_with_few_signals_and_limitations_always_present():
    full = score_fusion.fuse([], {}, 0.0, "jpeg", 1.0, {"barcode": "match", "glyphs": 60})
    poor = score_fusion.fuse([], {}, 0.0, "jpeg", 1.0, {"barcode": "unreadable", "glyphs": 3})
    assert poor.confidence < full.confidence and poor.limitations and full.limitations


def test_mask_value():
    assert mask_value("26329713") == "***713"
