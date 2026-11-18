import cv2
import numpy as np

from app import forensic_analyzer as fa, metadata_checker as mc


def _base():
    rng = np.random.default_rng(0)
    img = np.clip(np.full((600, 800, 3), 245) + rng.normal(0, 2, (600, 800, 3)), 0, 255).astype(np.uint8)
    for k, t in enumerate(["FACTURA 1234", "TOTAL 1500.00", "CLIENTE ACME SA"]):
        cv2.putText(img, t, (50, 100 + k * 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    return img


def _jpeg(img, q):
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _edited_resaved(q_orig, q_save):
    edited = _jpeg(_base(), q_orig)
    edited[210:270, 300:560] = 245
    cv2.putText(edited, "9500.00", (310, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    return _jpeg(edited, q_save)


def test_resaved_edit_scores_above_clean():
    clean = _jpeg(_base(), 70)
    s_clean, _, _ = fa.analyze_image(clean, 1)
    for q_save in (70, 92):
        s_edit, _, _ = fa.analyze_image(_edited_resaved(70, q_save), 1)
        assert s_edit > s_clean + 0.2


def test_ghost_localizes_unrecompressed_edit():
    img = _jpeg(_base(), 70)
    img[210:270, 300:560] = 245
    cv2.putText(img, "9500.00", (310, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    res = fa.ghost_analysis(img, 1)
    assert res.score > 0.5
    assert any(f.region and 250 <= f.region.x <= 560 and 180 <= f.region.y <= 280 for f in res.findings)


def test_clean_jpeg_has_no_ghost():
    assert fa.ghost_analysis(_jpeg(_base(), 70), 1).score == 0.0


def test_confidence_is_reliability_not_certainty():
    _, conf_jpeg, w_jpeg = fa.combine([0.0], [1.0], 0.0, "jpeg")
    _, conf_png, w_png = fa.combine([0.0], [1.0], 0.0, "png")
    assert conf_png < conf_jpeg < 0.9          # 0% de probabilidad ya no implica ~95% de confianza
    assert w_png and not any("PNG" in w for w in w_jpeg)


def test_metadata_weights_and_missing_exif_is_medium():
    assert mc.issue_weight("Software de edición detectado en EXIF: Photoshop") == 0.25
    assert mc.issue_weight("La imagen no contiene metadatos EXIF (posible eliminación o re-exportación)") == 0.12
    assert mc.issue_weight("Software registrado en EXIF: x") == 0.05


def test_standard_libjpeg_tables_detected(tmp_path):
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), _base(), [cv2.IMWRITE_JPEG_QUALITY, 85])
    issues, _ = mc.check_image(p)
    assert any("cuantización" in m for m in issues)
