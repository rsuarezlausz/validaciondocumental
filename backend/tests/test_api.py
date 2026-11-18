import os
import tempfile
import time

os.environ["DATA_DIR"] = tempfile.mkdtemp()

import cv2
import fitz
import numpy as np
from fastapi.testclient import TestClient

from app import forensic_analyzer as fa, ocr_engine
from app.main import app
from app.models import Paragraph


def _doc_image():
    rng = np.random.default_rng(0)
    img = np.full((600, 800, 3), 245, np.uint8)
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    cv2.putText(img, "FACTURA 1234", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    cv2.imwrite("_clean.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imread("_clean.jpg")


def test_forensics_flags_pasted_region():
    clean = _doc_image()
    tampered = clean.copy()
    patch = np.full((60, 200, 3), 255, np.uint8)
    cv2.putText(patch, "9999", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    tampered[300:360, 400:600] = patch  # pegado sin ruido ni compresión previa
    s_clean, _, _ = fa.analyze_image(clean, 1)
    s_bad, findings, _ = fa.analyze_image(tampered, 1)
    assert s_bad > s_clean
    assert any(f.region and 380 <= f.region.x <= 620 and 256 <= f.region.y <= 380 for f in findings)


def test_api_flow_pdf_and_history():
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 100), "Contrato de prueba con texto suficiente para la capa de texto.")
    pdf.set_metadata({"producer": "Adobe Acrobat", "creationDate": "D:20240101000000", "modDate": "D:20240301000000"})
    data = pdf.tobytes()
    with TestClient(app) as c:
        r = c.post("/api/v1/documents/upload", files={"file": ("a.pdf", data, "application/pdf")})
        assert r.status_code == 201
        doc_id = r.json()["document_id"]
        assert c.post(f"/api/v1/documents/analyze/{doc_id}").status_code == 202
        for _ in range(60):
            res = c.get(f"/api/v1/documents/results/{doc_id}").json()
            if res["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        assert res["status"] == "completed", res
        fa_ = res["forensic_analysis"]
        assert "Contrato" in res["ocr_text"]
        assert 0 <= fa_["tampering_probability"] <= 1
        assert any("posterior" in m for m in fa_["metadata_issues"])
        assert c.get(res["pages"][0]["url"]).status_code == 200
        assert c.get("/api/v1/documents").json()[0]["document_id"] == doc_id


def test_rejects_bad_type_and_spoofed_content():
    with TestClient(app) as c:
        assert c.post("/api/v1/documents/upload", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
        assert c.post("/api/v1/documents/upload", files={"file": ("a.png", b"not png", "image/png")}).status_code == 415


def test_image_flow_with_mock_ocr(monkeypatch):
    monkeypatch.setattr(ocr_engine, "ocr_page", lambda img, p: [Paragraph(page=p, text="hola")])
    ok, buf = cv2.imencode(".png", _doc_image())
    with TestClient(app) as c:
        doc_id = c.post("/api/v1/documents/upload", files={"file": ("a.png", buf.tobytes(), "image/png")}).json()["document_id"]
        c.post(f"/api/v1/documents/analyze/{doc_id}")
        for _ in range(60):
            res = c.get(f"/api/v1/documents/results/{doc_id}").json()
            if res["status"] != "processing":
                break
            time.sleep(0.5)
        assert res["status"] == "completed", res
        assert res["ocr_text"] == "hola"
        assert c.get(res["pages"][0]["url"]).status_code == 200   # la vista previa de imágenes debe existir

def test_card_photo_end_to_end_reports_copy_move_signals_and_limitations(monkeypatch):
    import synth
    from app.forensics import glyphs as G
    img, _ = synth.make_card()
    glyphs, _ink = G.segment_glyphs(img)
    line = sorted(G.lines_of(glyphs)[4], key=lambda g: g.x)
    edited = synth.paste_region(img, line[-2].bbox, (line[-1].x, line[-1].y))
    ocr = ("REPUBLICA ARGENTINA Fecha de nacimiento 28 DIC/ DEC 1977 Fecha de emision 18 DIC/ DEC 2015 "
           "Fecha de vencimiento 18 DIC/ DEC 2033")
    monkeypatch.setattr(ocr_engine, "ocr_page", lambda im, p: [Paragraph(page=p, text=ocr)])
    ok, buf = cv2.imencode(".jpg", edited, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with TestClient(app) as c:
        doc_id = c.post("/api/v1/documents/upload",
                        files={"file": ("dni.jpg", buf.tobytes(), "image/jpeg")}).json()["document_id"]
        c.post(f"/api/v1/documents/analyze/{doc_id}")
        for _ in range(120):
            res = c.get(f"/api/v1/documents/results/{doc_id}").json()
            if res["status"] != "processing":
                break
            time.sleep(0.5)
        assert res["status"] == "completed", res
        f = res["forensic_analysis"]
        types = {x["type"] for x in f["findings"]}
        assert "glyph_copy_move" in types and "rule_violation" in types
        assert f["tampering_probability"] >= 0.4
        assert f["limitations"] and "glyph_copy_move" in f["signals"] and "barcode_crosscheck" in f["signals"]
        cm = next(x for x in f["findings"] if x["type"] == "glyph_copy_move")
        assert cm["severity"] == "high" and len(cm["regions"]) == 2
        assert f["signals"]["timings"]
