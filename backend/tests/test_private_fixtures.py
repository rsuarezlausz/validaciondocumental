"""Regresión con imágenes reales del equipo (tests/fixtures/private/, NO versionadas). Se omiten si faltan.

Convención de nombres: `real*` = documento sin alterar; `falso*` = alterado (aquí: dígito 3 copiado sobre el 0 del
vencimiento 2030 -> 2033; `falso1` además trae una línea vertical fina de 1 px junto al último dígito).
"""
import glob
import shutil
from pathlib import Path

import cv2
import pytest

from app import forensic_analyzer as fa
from app.forensics import analyzer, score_fusion, template
import app.metadata_checker as mc

FIX = Path(__file__).parent / "fixtures" / "private"
REAL = sorted(glob.glob(str(FIX / "real*.j*")))
FALSO = sorted(glob.glob(str(FIX / "falso*.j*")))
pytestmark = pytest.mark.skipif(not (REAL and FALSO), reason="faltan imágenes privadas real*/falso*")

EDIT_BOX = (1660, 1380, 120, 100)   # x, y, w, h del último "3" pegado en la imagen de 2902x1927


def _score(path, profile="ar_dni"):
    img = cv2.imread(path)
    rep = analyzer.run(img, "", profile=profile)
    s, fnd, _ = fa.analyze_image(img, 1, template.excluded_mask(img))
    ps = {}
    for f in fnd:
        k = {"ela_inconsistency": "ela", "noise_inconsistency": "noise", "jpeg_ghost": "jpeg_ghost"}[f.type]
        ps[k] = max(ps.get(k, 0.0), f.score)
    issues, _ = mc.check_image(Path(path))
    fu = score_fusion.fuse(rep.detections, ps, sum(mc.issue_weight(m) for m in issues), "jpeg", 1.0,
                           {"barcode": rep.barcode.status, "glyphs": rep.glyph_count,
                            "copy_move_applicable": rep.copy_move_applicable})
    return rep, fu


@pytest.mark.parametrize("path", REAL)
def test_real_document_is_not_flagged(path):
    rep, fu = _score(path)
    assert not [d for d in rep.detections if d.type in ("glyph_copy_move", "seam_artifact", "barcode_mismatch")]
    assert not [d for d in rep.detections if d.severity == "high"]
    assert fu.verdict != "evidence"
    assert rep.barcode.status != "mismatch"


@pytest.mark.parametrize("path", FALSO)
def test_edited_document_gets_copy_move_on_last_digits_and_high_score(path):
    rep, fu = _score(path)
    cm = [d for d in rep.detections if d.type == "glyph_copy_move"]
    assert cm and cm[0].severity == "high"
    ax, bx = cm[0].extra["region_a"][0], cm[0].extra["region_b"][0]
    ex, ey, ew, eh = EDIT_BOX
    assert ex <= min(ax, bx) and max(ax, bx) <= ex + ew
    assert fu.probability >= 0.6


def test_seam_only_in_the_image_that_has_it():
    seams = {Path(p).name: [d for d in _score(p, "ar_dni")[0].detections if d.type == "seam_artifact"] for p in FALSO}
    with_seam = [n for n, v in seams.items() if v]
    assert with_seam == ["falso1.jpg"]
    d = seams["falso1.jpg"][0]
    x, y, w, h = d.bbox
    assert abs((x + 2) - 1757) <= 6 and 1395 <= y <= 1420 and h >= 40


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract no disponible")
def test_targeted_ocr_reads_expiry_year_and_never_invents_barcode_mismatch():
    from app.forensics import field_ocr
    for p in REAL + FALSO:
        reads = field_ocr.read_fields(cv2.imread(p))
        assert reads and reads["doc_number"].value == "26329713"
        exp = reads["expiry"].value
        assert exp == ("2030" if "real" in p else "2033") or not reads["expiry"].reliable
        rep = analyzer.run(cv2.imread(p), "")
        assert rep.barcode.status != "mismatch"


ID3 = sorted(glob.glob(str(FIX / "id_3*.j*")))


@pytest.mark.skipif(not ID3, reason="falta id_3.jpeg")
def test_id3_white_box_with_letter_is_high_severity_flat_patch():
    """Cuadro blanco con la letra "A" en el campo 9a END, sobre fondo de seguridad."""
    rep, fu = _score(ID3[0], profile="generic")
    fp = [d for d in rep.detections if d.type == "flat_patch"]
    assert fp and fp[0].severity == "high"
    x, y, w, h = fp[0].bbox
    assert abs(x - 450) <= 20 and abs(y - 265) <= 12 and w >= 28 and h >= 16
    assert fu.verdict != "no_evidence" and fu.probability >= 0.5
