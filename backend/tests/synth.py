"""Generador de documentos sintéticos tipo DNI (sin datos personales) para tests y evaluación."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
LINES = [
    ("SURNAME TESTER", 0.0), ("GIVEN NAME DEMO", 0.0), ("DOB 28 DEC 1977", 0.0),
    ("ISSUE 18 DEC 2015", 0.0), ("EXPIRY 18 DEC 2030", 0.0), ("NO 12 345 678", 0.0),
]


def make_card(seed: int = 0, w: int = 1600, h: int = 1000, noise: float = 3.0, jpeg_q: int = 90,
              lines=None, text_scale: float = 1.4, pitch: int = 38) -> tuple[np.ndarray, dict]:
    """Devuelve (imagen BGR ya pasada por JPEG, info con posiciones de cada línea)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 200 + 12 * np.sin(xx / 260.0) + 8 * np.cos(yy / 190.0)          # iluminación suave
    base = base + 10 * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 9) * 4  # guilloché suave
    img = np.clip(np.stack([base * 0.97, base, base * 0.95], axis=2), 0, 255).astype(np.uint8)
    pos = {}
    for k, text in enumerate(lines or [t for t, _ in LINES]):
        org = (900, 220 + k * 110)
        for i, ch in enumerate(text):          # paso fijo (~38 px) como los dígitos del DNI
            if ch != " ":
                cv2.putText(img, ch, (org[0] + i * pitch, org[1]), FONT, text_scale, (50, 50, 50), 3, cv2.LINE_AA)
        pos[k] = org
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), {"lines": pos}


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def paste_region(img: np.ndarray, src: tuple[int, int, int, int], dst_xy: tuple[int, int], margin: int = 6) -> np.ndarray:
    """Copia src (x,y,w,h) + margen sobre dst: simula copiar y pegar un carácter."""
    x, y, w, h = src
    patch = img[y - margin:y + h + margin, x - margin:x + w + margin].copy()
    out = img.copy()
    dx, dy = dst_xy
    out[dy - margin:dy + h + margin, dx - margin:dx + w + margin] = patch
    return out
