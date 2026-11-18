"""Copy-move entre glifos: detecta un carácter copiado de otro lugar del mismo documento.

Para cada par de glifos de tamaño compatible se mide (a) similitud de forma (matchTemplate NCC con
búsqueda de desplazamiento) y (b) correlación del residuo de ruido de alta frecuencia dentro de la
caja con margen (incluye fondo). Un dígito repetido de forma natural tiene forma casi idéntica pero
residuos independientes; un pegado reproduce también el residuo (mismos píxeles).

La separación medida es pequeña, por eso: puntuación gradual, umbrales en YAML y requisitos de apoyo
(fondo correlacionado, misma iluminación, atipicidad frente a los pares naturales del documento).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import settings
from .glyphs import Glyph, robust_z, to_gray
from .types import Detection, union_bbox

MARGIN = 10
SHIFT = 3


@dataclass
class CopyMoveResult:
    max_shape_ncc: float = 0.0
    max_residual_ncc: float = 0.0
    pairs_evaluated: int = 0
    detections: list[Detection] = field(default_factory=list)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def _crop(arr: np.ndarray, x: int, y: int, w: int, h: int, m: int) -> np.ndarray | None:
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = x - m, y - m, x + w + m, y + h + m
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    return arr[y0:y1, x0:x1]


def _best_shift(gray: np.ndarray, a: Glyph, b: Glyph) -> tuple[float, tuple[int, int]]:
    """Mejor NCC de forma de A (plantilla) dentro de B (ventana con margen SHIFT)."""
    tpl = _crop(gray, a.x, a.y, a.w, a.h, 2)
    win = _crop(gray, b.x, b.y, b.w, b.h, 2 + SHIFT)
    if tpl is None or win is None or win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return 0.0, (0, 0)
    if tpl.std() < 1e-3:
        return 0.0, (0, 0)
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), (loc[0] - SHIFT, loc[1] - SHIFT)


def estimate_noise(img: np.ndarray) -> float:
    """Desvío del residuo de alta frecuencia en zonas planas: mide cuánto ruido de sensor queda.
    Un render digital o una recompresión fuerte lo eliminan y el copy-move deja de ser fiable."""
    g = to_gray(img).astype(np.float32)
    sm = cv2.GaussianBlur(g, (0, 0), 2.0)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1)) / 8.0
    flat = grad < max(2.5 * float(np.median(grad)), 0.05)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    return float(res[flat].std()) if flat.sum() > 500 else 0.0


def _ramp(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1))


def analyze(img: np.ndarray, glyphs: list[Glyph], ink: np.ndarray) -> CopyMoveResult:
    cfg = settings.thresholds()["glyph_copy_move"]
    res = CopyMoveResult()
    if len(glyphs) < 4:
        return res
    gray = to_gray(img)
    gf = gray.astype(np.float32)
    residual = gf - cv2.GaussianBlur(gf, (0, 0), 1.0)
    tol = cfg["size_tolerance_px"]
    # Píxeles de fondo "planos": gradiente suavizado bajo (relativo a la mediana de la imagen). Así el ruido
    # del fondo no se mezcla con halos de trazos propios o vecinos que se repiten por contexto (p.ej. "TE").
    sm = cv2.GaussianBlur(gf, (0, 0), 2.0)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1)) / 8.0
    flat = (grad < max(2.5 * float(np.median(grad)), 0.05)).astype(np.uint8)
    ink_d = cv2.dilate(ink, np.ones((5, 5), np.uint8)) > 0   # tinta (propia y de glifos vecinos) nunca es 'fondo'

    pairs = []
    for i in range(len(glyphs)):
        for j in range(i + 1, len(glyphs)):
            a, b = glyphs[i], glyphs[j]
            if abs(a.h - b.h) > tol or abs(a.w - b.w) > tol:
                continue
            if a.w < 10 or a.h < 20:   # glifos finos (1, I, l) son casi iguales por construcción
                continue
            # glifos pegados uno al lado del otro (mismo trazo partido) no son pares válidos
            if abs(a.x - b.x) < max(a.w, b.w) and abs(a.y - b.y) < max(a.h, b.h):
                continue
            pairs.append((a, b))
    if not pairs:
        return res

    rows = []
    for a, b in pairs:
        shape, (dx, dy) = _best_shift(gray, a, b)
        if shape < 0.6:                       # no se parecen: no hace falta el resto
            rows.append((a, b, shape, 0.0, 0.0, 99.0))
            continue
        bx, by = b.x + dx, b.y + dy
        ca = _crop(residual, a.x, a.y, a.w, a.h, MARGIN)
        cb = _crop(residual, bx, by, a.w, a.h, MARGIN)
        if ca is None or cb is None or ca.shape != cb.shape:
            rows.append((a, b, shape, 0.0, 0.0, 99.0))
            continue
        res_ncc = _ncc(ca, cb)
        # solo fondo plano en ambos recortes; se prueba un margen corto (pegado ajustado al glifo)
        # y uno largo (pegado con fondo generoso) y se toma el mayor
        bg_ncc, bgm = -1.0, None
        for m in (2, 4, MARGIN):
            # recortes que se solapan comparten fondo (y su textura): correlación espuria, se descartan
            if abs(a.x - bx) < a.w + 2 * m and abs(a.y - by) < a.h + 2 * m:
                continue
            ra, rb = _crop(residual, a.x, a.y, a.w, a.h, m), _crop(residual, bx, by, a.w, a.h, m)
            fa_, fb_ = _crop(flat, a.x, a.y, a.w, a.h, m), _crop(flat, bx, by, a.w, a.h, m)
            if ra is None or rb is None or fa_ is None or fb_ is None or ra.shape != rb.shape:
                continue
            ia_, ib_ = _crop(ink_d, a.x, a.y, a.w, a.h, m), _crop(ink_d, bx, by, a.w, a.h, m)
            if ia_ is None or ib_ is None or ia_.shape != ra.shape or ib_.shape != ra.shape:
                continue
            # Texto periódico (MRZ "<<<<", puntos, rayas): el contexto vecino se repite idéntico. Por eso el fondo
            # excluye la tinta vecina, y si lo que queda tiene estructura (no ruido) no se puede afirmar nada.
            mm = (fa_ > 0) & (fb_ > 0) & ~ia_ & ~ib_
            if mm.sum() > 40 and float(ra[mm].std()) <= 8.0 and float(rb[mm].std()) <= 8.0:
                v = _ncc(ra[mm], rb[mm])
                if v > bg_ncc:
                    bg_ncc, bgm = v, (m, mm)
        illum = 255.0
        if bgm is not None:
            m, mm = bgm
            ga, gb = _crop(gf, a.x, a.y, a.w, a.h, m), _crop(gf, bx, by, a.w, a.h, m)
            illum = abs(float(ga[mm].mean()) - float(gb[mm].mean()))
        rows.append((a, b, shape, res_ncc, bg_ncc, illum))

    res.pairs_evaluated = len(rows)
    res.max_shape_ncc = round(max(r[2] for r in rows), 3)
    res.max_residual_ncc = round(max((r[3] for r in rows if r[2] >= cfg["shape_ncc_floor"]), default=0.0), 3)

    # distribución natural de residuos: pares con forma parecida (>=0.85) del mismo documento
    nat = np.array([r[4] for r in rows if r[2] >= 0.85 and r[4] >= -0.5])

    cands = []
    for k, (a, b, shape, rn, bgn, illum) in enumerate(rows):
        if shape < cfg["shape_ncc_floor"]:
            continue
        s_shape = _ramp(shape, cfg["shape_ncc_floor"], cfg["shape_ncc_full"])
        s_res = _ramp(rn, cfg["residual_ncc_floor"], cfg["residual_ncc_full"])
        s_bg = _ramp(max(bgn, 0.0), cfg["background_ncc_floor"], cfg["background_ncc_full"])
        same_light = illum <= cfg["illumination_delta_max"]
        z_ok = 1.0
        if len(nat) >= 3:
            med = np.median(nat)
            mad = max(np.median(np.abs(nat - med)) * 1.4826, 0.05)
            z_ok = _ramp((bgn - med) / mad, 0.0, cfg["distribution_z_min"])
        # En un render con glifos idénticos (o dígitos repetidos con forma casi igual) el residuo con trazos
        # se correlaciona por la propia forma; lo que delata una copia es el ruido del FONDO. Por eso el
        # residuo del fondo pesa más; luz parecida y atipicidad frente a los pares naturales refuerzan.
        resid_term = 0.85 * s_bg + 0.15 * s_res if bgn >= -0.5 else 0.3 * s_res
        support = 0.2 + 0.2 * (1.0 if same_light else 0.0) + 0.6 * z_ok
        score = float(np.clip(s_shape * resid_term * support, 0, 1))
        if score >= cfg["min_score_report"]:
            cands.append((score, a, b, shape, rn, bgn, illum))
    for score, a, b, shape, rn, bgn, illum in sorted(cands, key=lambda c: -c[0])[:5]:
        sev = "high" if score >= cfg["high_score"] else "medium"
        res.detections.append(Detection(
            "glyph_copy_move",
            f"Dos caracteres casi idénticos: forma {shape:.2f}, residuo de ruido {rn:.2f}, fondo {bgn:.2f}. "
            "Un dígito repetido natural suele tener residuo independiente; esta correlación sugiere copia. "
            "Indicio, no prueba.", sev, round(score, 3), union_bbox(a.bbox, b.bbox),
            {"region_a": list(a.bbox), "region_b": list(b.bbox), "shape_ncc": round(shape, 3),
             "residual_ncc": round(rn, 3), "background_ncc": round(bgn, 3)}))
    return res
