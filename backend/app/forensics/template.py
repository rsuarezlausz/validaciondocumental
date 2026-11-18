"""Plantilla del documento: detección de tipo (aspecto de tarjeta) y máscara de zonas excluidas."""
import numpy as np

from . import settings


def is_card(img: np.ndarray) -> bool:
    t = settings.template()
    if not t.get("enabled", False):
        return False
    h, w = img.shape[:2]
    return abs(max(w, h) / max(min(w, h), 1) - t["aspect_ratio"]) <= t["aspect_tolerance"] and w >= h


def zones_px(shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    H, W = shape[:2]
    out = {}
    for name, z in settings.template().get("excluded_zones", {}).items():
        out[name] = (int(z["x"] * W), int(z["y"] * H), int(z["w"] * W), int(z["h"] * H))
    return out


def excluded_mask(img: np.ndarray, names: tuple[str, ...] | None = None) -> np.ndarray | None:
    """Máscara bool (True = zona excluida de ELA/ruido). None si la plantilla no aplica."""
    if not is_card(img):
        return None
    m = np.zeros(img.shape[:2], bool)
    for name, (x, y, w, h) in zones_px(img.shape).items():
        if names is None or name in names:
            m[y:y + h, x:x + w] = True
    mx = float(settings.template().get("border_margin_x", 0))
    my = float(settings.template().get("border_margin_y", 0))
    if (mx > 0 or my > 0) and (names is None or "border" in names):
        H, W = img.shape[:2]
        by, bx = int(my * H), int(mx * W)
        if by:
            m[:by, :] = True
            m[H - by:, :] = True
        if bx:
            m[:, :bx] = True
            m[:, W - bx:] = True
    return m


def zone_of(bbox: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> str | None:
    if bbox is None:
        return None
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    for name, (x, y, w, h) in zones_px(shape).items():
        if x <= cx <= x + w and y <= cy <= y + h:
            return name
    return None
