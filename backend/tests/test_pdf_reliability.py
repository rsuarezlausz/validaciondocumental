"""Fiabilidad en PDF: asociación de fechas, barcode opcional, imagen embebida nativa y capacidades reales."""
import glob
import os
import tempfile
import time
from pathlib import Path

import cv2
import fitz
import pytest

import synth
from app.forensics import barcode_validator as bv, score_fusion
from app.forensics.types import Detection
from app.services import pipeline

# OCR real de una página de DNI en PDF: la fecha de emisión quedó bajo el holograma y no se leyó
OCR_DNI_PDF = """REPUBLICA ARGENTINA - MERCOSUR
Sexo / Sex Nacionalidad / Nationality Ejemplar
M ARGENTINA
Fecha de nacimiento / Date of birth
28 DIC/ DEC 1977
«Echa de emisión / Date of issue
Fecha de vencimiento / Date of expiry
"18 DIC/ DEC 2030 ee
Documento / Document (Frame Ne. OÍ ident.
26.329.743"""

PAYLOAD = "00412747395@TESTER SURNAME@DEMO NAME@M@12345678@B@28/12/1977@18/12/2015"


def test_issue_date_is_not_taken_from_the_expiry_that_follows():
    dates = bv.extract_ocr_dates(OCR_DNI_PDF)
    assert "issue" not in dates                       # ilegible: no se inventa
    assert dates["expiry"].year == 2030 and dates["birth"].year == 1977


def test_unreadable_issue_is_inconclusive_not_a_mismatch():
    bc = bv.parse_barcode_payload(PAYLOAD)
    rows = {r["field"]: r for r in bv.compare_fields(bc, OCR_DNI_PDF)}
    assert rows["issue"]["status"] == "not_found"
    assert rows["birth"]["status"] == "match"


def test_same_date_under_two_labels_is_ambiguous():
    txt = "Fecha de emision 18 DIC/ DEC 2030 Fecha de vencimiento 18 DIC/ DEC 2030"
    assert bv.extract_ocr_dates(txt) == {}


def test_page_ocr_mismatch_is_weak_and_targeted_mismatch_is_strong():
    txt = "Fecha de nacimiento 28 DIC/ DEC 1977 Fecha de emision 18 DIC/ DEC 2016 Fecha de vencimiento 18 DIC/ DEC 2031"
    import numpy as np
    blank = np.zeros((50, 50, 3), np.uint8)
    weak = bv.validate(blank, txt, payload=PAYLOAD)
    d = [x for x in weak.detections if x.type == "barcode_mismatch"]
    assert d and d[0].severity == "low" and d[0].score <= 0.25 and weak.status != "mismatch"

    class Read:       # lectura dirigida y fiable (>= 2 votos)
        reliable = True
        value = "2016"
    strong = bv.validate(blank, "", payload=PAYLOAD, field_reads={"issue": Read()})
    d = [x for x in strong.detections if x.type == "barcode_mismatch"]
    assert d and d[0].severity == "high" and strong.status == "mismatch"


def test_missing_barcode_does_not_lower_confidence_or_raise_score():
    base = dict(pixel_scores={}, meta_weight=0.0, input_kind="jpeg", resolution=1.0)
    none = score_fusion.fuse([], availability={"barcode": "unreadable", "glyphs": 60}, **base)
    ok = score_fusion.fuse([], availability={"barcode": "match", "glyphs": 60}, **base)
    assert none.probability == ok.probability == 0.0
    assert not any("código de barras" in w for w in none.warnings)
    assert none.confidence >= 0.6


def test_one_weak_page_ocr_mismatch_cannot_make_evidence():
    weak = Detection("barcode_mismatch", "x", "low", 0.2)
    f = score_fusion.fuse([weak], {}, 0.0, "jpeg", 1.0, {"barcode": "partial", "glyphs": 60})
    assert f.probability < 0.25 and f.verdict != "evidence"


def _pdf_with_image(tmp_path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    doc = fitz.open()
    page = doc.new_page(width=595, height=595 * img.shape[0] / img.shape[1])
    page.insert_image(page.rect, stream=buf.tobytes())
    p = tmp_path / "scan.pdf"
    doc.save(p)
    return p


def test_pdf_with_full_page_image_uses_native_resolution(tmp_path):
    card, _ = synth.make_card()
    images, texts, kinds = pipeline._render_pdf(_pdf_with_image(tmp_path, card))
    assert kinds == [(pipeline.PageSource.PDF_IMAGE, True)]          # JPEG: ELA/ghost siguen siendo válidos
    assert images[0].shape[:2] == card.shape[:2]                     # no se rasterizó a otra resolución


def test_vector_pdf_is_rasterized_and_marked_lossless(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Contrato de prueba con texto suficiente.")
    p = tmp_path / "v.pdf"
    doc.save(p)
    images, texts, kinds = pipeline._render_pdf(p)
    assert kinds == [(pipeline.PageSource.PDF_VECTOR, False)]


def test_capabilities_match_what_really_runs():
    vec = pipeline._capabilities(any_lossy=False, photo_pages=False, copy_move_ok=False, vector_only=True)
    assert "ela" not in vec["ran"] and {"detector": "ela", "reason": "lossless_or_rasterized_page"} in vec["skipped"]
    assert "flat_patch" not in vec["ran"]
    photo = pipeline._capabilities(any_lossy=True, photo_pages=True, copy_move_ok=True, vector_only=False)
    assert {"ela", "flat_patch", "copy_move"} <= set(photo["ran"]) and photo["skipped"] == []


def test_lossless_page_skips_ela_and_ghost():
    from app import forensic_analyzer as fa
    card, _ = synth.make_card()
    _, findings, _ = fa.analyze_image(card, 1, None, use_compression=False)
    assert not [f for f in findings if f.type in ("ela_inconsistency", "jpeg_ghost")]


PDF_FIX = sorted(glob.glob(str(Path(__file__).parent / "fixtures" / "private" / "DNI_PDF*.pdf")))


@pytest.mark.skipif(not PDF_FIX, reason="falta DNI_PDF.pdf privado")
def test_genuine_dni_converted_to_pdf_is_not_flagged():
    os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        doc = c.post("/api/v1/documents/upload",
                     files={"file": ("dni.pdf", Path(PDF_FIX[0]).read_bytes(), "application/pdf")}).json()["document_id"]
        c.post(f"/api/v1/documents/analyze/{doc}")
        for _ in range(240):
            r = c.get(f"/api/v1/documents/results/{doc}").json()
            if r["status"] != "processing":
                break
            time.sleep(0.5)
        assert r["status"] == "completed", r
        f = r["forensic_analysis"]
        assert f["signals"]["page_sources"] == ["pdf_embedded_image"]
        assert f["verdict"] != "evidence" and f["tampering_probability"] < 0.4
        assert not [x for x in f["findings"] if x["type"] == "barcode_mismatch" and x["severity"] == "high"]
        assert c.get(r["pages"][0]["url"]).status_code == 200


# ---------------------------------------------------------------- PDF de Word con imágenes (frente y dorso), capa de texto vacía
OCR_PE = """REPUBLICA DEL PERU
Primer Apellido PEREZ Segundo Apellido TEJADA Prenombres CESAR AUGUSTO
Sexo Nacionalidad Fecha de Nacimiento
882364 M PER 27 05 1975
Estado Civil Fecha de Emision
CASADO 19 07 2021
N de Tarjeta Fecha de Caducidad
0200340972 1907 2029"""


def test_whitespace_text_layer_is_not_text():
    assert not pipeline._meaningful_text(" \n \n" * 30)
    assert not pipeline._meaningful_text("")
    assert pipeline._meaningful_text("Contrato de prueba con texto suficiente.")


def test_numeric_dates_and_glued_day_month_are_read_by_label():
    d = bv.extract_ocr_dates(OCR_PE)
    assert (d["birth"].year, d["issue"].year, d["expiry"].year) == (1975, 2021, 2029)
    assert (d["expiry"].month, d["expiry"].day) == (7, 19)      # "1907 2029" -> 19/07/2029


def test_profile_detection_and_no_validity_rule_for_unknown_documents():
    from datetime import date
    from app.forensics import profile, rules
    assert profile.detect("REPUBLICA ARGENTINA - MERCOSUR").name == "ar_dni"
    assert profile.detect(OCR_PE).name == "generic"
    assert profile.detect("").name == "generic"
    # DNI peruano: emisión 2021 -> caducidad 2029 (8 años) NO debe violar la vigencia supuesta del DNI argentino (15)
    dates = {"birth": date(1975, 5, 27), "issue": date(2021, 7, 19), "expiry": date(2029, 7, 19)}
    assert rules.evaluate(dates, check_validity=profile.detect(OCR_PE).check_validity) == []
    assert rules.evaluate(dates, check_validity=True)          # con el perfil AR sí (supuesto por verificar)


def test_pdf_with_two_images_and_empty_text_layer_analyzes_each_image_natively(tmp_path):
    front, _ = synth.make_card(seed=1, w=720, h=441)
    back, _ = synth.make_card(seed=2, w=720, h=438)
    doc = fitz.open()
    page = doc.new_page()          # A4
    for img, y in ((front, 100), (back, 450)):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        page.insert_image(fitz.Rect(130, y, 465, y + 205), stream=buf.tobytes())
    page.insert_text((10, 10), " ")      # capa de texto "vacía"
    p = tmp_path / "word_like.pdf"
    doc.save(p)
    images, texts, kinds = pipeline._render_pdf(p)
    assert len(images) == 2 and all(k == (pipeline.PageSource.PDF_IMAGE, True) for k in kinds)
    assert [im.shape[:2] for im in images] == [(441, 720), (438, 720)]       # resolución nativa, no el A4 rasterizado


def test_dense_repeated_glyphs_and_barcode_lines_do_not_raise_copy_move_or_seam():
    """Texto periódico tipo MRZ ("<<<<") y rayas de código de barras no son una edición."""
    import numpy as np
    from app.forensics import analyzer
    rng = np.random.default_rng(3)
    img = np.clip(np.full((441, 720, 3), 235) + rng.normal(0, 4, (441, 720, 3)), 0, 255).astype(np.uint8)
    for x in range(40, 680, 20):
        cv2.putText(img, "<", (x, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    for y in range(60, 140, 14):
        img[y:y + 2, 600:690] = 20
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], 1)
    rep = analyzer.run(img, "", profile="generic")
    assert not [d for d in rep.detections if d.type in ("glyph_copy_move", "seam_artifact")]


PERU = sorted(glob.glob(str(Path(__file__).parent / "fixtures" / "private" / "DNI Cesar*.pdf")))


@pytest.mark.skipif(not PERU, reason="falta el PDF peruano privado")
def test_peruvian_dni_pdf_gets_ocr_and_is_not_flagged():
    os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        doc = c.post("/api/v1/documents/upload",
                     files={"file": ("dni.pdf", Path(PERU[0]).read_bytes(), "application/pdf")}).json()["document_id"]
        c.post(f"/api/v1/documents/analyze/{doc}")
        for _ in range(240):
            r = c.get(f"/api/v1/documents/results/{doc}").json()
            if r["status"] != "processing":
                break
            time.sleep(0.5)
        assert r["status"] == "completed", r
        f = r["forensic_analysis"]
        assert len(r["pages"]) == 2 and f["signals"]["page_sources"] == ["pdf_embedded_image"] * 2
        assert f["signals"]["document_profile"] == "generic"
        import shutil
        if shutil.which("tesseract") or os.path.isfile(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe")):
            assert len(r["ocr_text"]) > 300 and "PERU" in r["ocr_text"].upper()      # antes: 0 (capa de texto en blanco)
        assert f["verdict"] != "evidence" and f["tampering_probability"] < 0.5
        assert not [x for x in f["findings"] if x["type"] in ("glyph_copy_move", "seam_artifact", "rule_violation")
                    and x["severity"] == "high"]
