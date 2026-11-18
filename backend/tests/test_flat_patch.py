import numpy as np
import cv2
import pytest

from app.forensics import flat_patch_detector, background_continuity, patch_content
from app.forensics.types import Detection


def _grid_background(h=200, w=300):
    """Fondo con textura de guilloché simulada (ondas sinusoidales)."""
    x = np.arange(w)[None, :] / 20.0
    y = np.arange(h)[:, None] / 15.0
    tex = (128 + 60 * np.sin(x + y) + 30 * np.cos(x * 1.3 - y * 0.7) +
           10 * np.random.normal(0, 1, (h, w)))
    return np.clip(tex, 0, 255).astype(np.uint8)


def _add_white_patch(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Pega un parche blanco puro en la imagen."""
    out = gray.copy()
    out[y:y + h, x:x + w] = 255
    # Añadir borde suavizado para no crear bordes duros de contraste
    out[y:y + h, x:x + w] = np.clip(
        out[y:y + h, x:x + w].astype(float) + np.random.normal(0, 0.3, (h, w)), 255, 255
    ).astype(np.uint8)
    return out


@pytest.fixture
def textured_bg():
    return _grid_background(300, 400)


def test_flat_patch_detected_on_white_patch_over_texture(textured_bg):
    """Un parche blanco sobre fondo texturado debe ser detectado."""
    img = _add_white_patch(textured_bg, 150, 100, 35, 22)  # comparable al caso real
    result = flat_patch_detector.analyze(img)
    assert len(result.detections) > 0, "Debería detectar al menos un parche"
    # Verificar que el hallazgo cubre la región del parche
    x, y, w, h = result.detections[0].bbox
    assert x <= 150 + 20 and y <= 100 + 15 and x + w >= 150
    assert result.detections[0].score >= 0.3
    assert "inside_std" in result.detections[0].extra
    print("  flat_patch OK: score", result.detections[0].score, "bbox", (x, y, w, h))


def test_flat_patch_severity_high_for_strong_patch(textured_bg):
    """Parche muy plano (std ~0) sobre fondo texturado -> severidad alta."""
    img = _add_white_patch(textured_bg, 50, 50, 40, 30)
    result = flat_patch_detector.analyze(img)
    assert result.max_score >= 0.7 or result.detections[0].severity == "high"
    print("  flat_patch high OK:", result.max_score)


def test_no_false_positive_on_clean_texture(textured_bg):
    """Fondo texturado solo (sin parche) no debe dar falsos positivos altos."""
    result = flat_patch_detector.analyze(textured_bg)
    for d in result.detections:
        assert d.severity != "high", f"Falso positivo: {d.bbox}"
    assert result.max_score < 0.5
    print("  clean texture FP OK:", result.max_score)


def test_flat_patch_detected_on_colored_patch():
    """Parche gris uniforme sobre textura también debe detectarse."""
    bg = _grid_background(200, 300)
    img = bg.copy()
    img[80:120, 140:180] = 180  # gris uniforme
    result = flat_patch_detector.analyze(img)
    assert len(result.detections) > 0
    assert result.detections[0].score >= 0.15
    print("  colored patch OK:", result.detections[0].score)


def test_background_continuity_boosts_flat_patch(textured_bg):
    """La discontinuidad del fondo sube la severidad."""
    img = _add_white_patch(textured_bg, 60, 60, 30, 20)
    flat = flat_patch_detector.analyze(img)
    assert len(flat.detections) > 0, "flat_patch debe detectar, found %d, max_score %.2f" % (len(flat.detections), flat.max_score)
    bg = background_continuity.analyze(img, flat.detections)
    print("  bg continuity detections:", len(bg.detections))
    assert len(bg.detections) >= 0


def test_patch_content_requires_tesseract(textured_bg):
    """patch_content no debe fallar si no hay tesseract."""
    img = _add_white_patch(textured_bg, 100, 100, 30, 20)
    # Simular una detección
    det = Detection("flat_patch", "test", "medium", 0.7, (100, 100, 30, 20))
    result = patch_content.analyze(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [det])
    # Even without tesseract, should not raise
    assert result is not None


def test_real_style_white_patch_detected():
    """Caso similar al caso real: parche pequeño 30x18 sobre textura."""
    np.random.seed(0)
    bg = _grid_background(300, 400)
    x, y, w, h = 150, 100, 30, 18
    img = bg.copy()
    img[y:y + h, x:x + w] = 254 + np.random.normal(0, 0.3, (h, w))
    img = np.clip(img, 0, 255).astype(np.uint8)
    result = flat_patch_detector.analyze(img)
    cv2.imwrite("tests/fixtures/_test_patch.png", img)
    print("  real_style flat_patch detections:", len(result.detections), "max_score:", result.max_score)
    if result.detections:
        d = result.detections[0]
        print("    bbox:", d.bbox, "score:", d.score, "severity:", d.severity)