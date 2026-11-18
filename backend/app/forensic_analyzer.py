"""Análisis forense de píxeles: ELA + residual de ruido, con localización por bloques.

Heurístico (POC): los scores son indicadores, no una prueba pericial.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .models import Finding, Region

BLOCK = 16
Z_THRESHOLD = 4.0
MAX_REGIONS_PER_MAP = 5


@dataclass
class MapResult:
    score: float
    findings: list[Finding]


def ela_map(img: np.ndarray, quality: int = 90, normalize: bool = True) -> np.ndarray:
    """ELA. Con normalize=True el error se divide por la textura local, para que las zonas de alta
    textura (guilloché, foto, holograma) no se marquen solo por ser texturadas."""
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img, recompressed).astype(np.float32).max(axis=2)
    if not normalize:
        return diff
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mean * mean, 0))
    return diff / (2.0 + 0.5 * std)


def noise_map(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    return np.abs(residual)


def _block_means(m: np.ndarray) -> np.ndarray:
    h, w = (m.shape[0] // BLOCK) * BLOCK, (m.shape[1] // BLOCK) * BLOCK
    if h == 0 or w == 0:
        return np.zeros((1, 1), np.float32)
    return m[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(1, 3))


def _outliers(blocks: np.ndarray, excl: np.ndarray | None = None):
    # En documentos la mayoría de los bloques es fondo plano. La estadística robusta se calcula
    # solo sobre los bloques con contenido (texto/bordes) para que el fondo no infle el z-score.
    floor = max(np.percentile(blocks, 70), 1e-3)
    active = blocks > floor
    if excl is not None:
        active &= ~excl
    if active.sum() < 20:
        active = np.ones_like(blocks, bool)
    ref = blocks[active]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    scale = max(mad, ref.std() * 0.5, 1e-3)
    z = np.where(active, (blocks - med) / scale, -np.inf)
    return z, z > Z_THRESHOLD


def _block_excl(excl, shape):
    if excl is None:
        return None
    return _block_means(excl.astype(np.float32)) > 0.5


def _analyze_map(m: np.ndarray, kind: str, page: int, label: str, excl=None) -> MapResult:
    blocks = _block_means(m)
    z, mask = _outliers(blocks, _block_excl(excl, m.shape))
    if not mask.any():
        return MapResult(0.0, [])
    mask_u8 = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:MAX_REGIONS_PER_MAP]
    findings = []
    for i in comps:
        x, y, w, h, _a = stats[i]
        peak = float(z[labels[: z.shape[0], : z.shape[1]] == i].max()) if (labels == i).any() else Z_THRESHOLD
        s = float(np.clip((peak - Z_THRESHOLD) / 10, 0, 1))
        findings.append(Finding(
            type=kind, score=round(s, 3),
            region=Region(x=int(x * BLOCK), y=int(y * BLOCK), width=int(w * BLOCK), height=int(h * BLOCK), page=page),
            description=f"{label} (z={peak:.1f})",
        ))
    peak_all = float(z[mask].max())
    area_ratio = float(mask.mean())
    score = 0.6 * np.clip((peak_all - Z_THRESHOLD) / 10, 0, 1) + 0.4 * np.clip(area_ratio * 20, 0, 1)
    return MapResult(float(score), findings)


GHOST_QUALITIES = list(range(50, 100, 5))
GHOST_MIN_STEPS = 2          # diferencia mínima (en pasos de 5 de quality) respecto al quality dominante
GHOST_MIN_BLOCKS = 3         # tamaño mínimo (bloques) de una región sospechosa


def ghost_analysis(img: np.ndarray, page: int, excl=None) -> MapResult:
    """JPEG ghost: recomprime a varios quality y localiza bloques cuyo quality "de mínimo error"
    difiere del dominante => región con historial de compresión distinto (pegada o re-guardada)."""
    diffs = []
    for q in GHOST_QUALITIES:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        d = ((img.astype(np.float32) - rec.astype(np.float32)) ** 2).mean(axis=2)
        diffs.append(_block_means(d))
    stack = np.stack(diffs)                      # (nq, bh, bw)
    span = stack.max(axis=0) - stack.min(axis=0)
    textured = span > np.percentile(span, 50) * 0.5 + 0.5   # bloques con información (no planos)
    bex = _block_excl(excl, img.shape[:2])
    if bex is not None:
        textured = textured & ~bex
    if textured.sum() < 20:
        return MapResult(0.0, [])
    best = stack.argmin(axis=0)
    mode = int(np.bincount(best[textured]).argmax())
    mask = textured & (np.abs(best - mode) >= GHOST_MIN_STEPS)
    mask_u8 = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) & textured.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= GHOST_MIN_BLOCKS]
    comps = sorted(comps, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:MAX_REGIONS_PER_MAP]
    if not comps:
        return MapResult(0.0, [])
    frac = float(sum(stats[i, cv2.CC_STAT_AREA] for i in comps) / textured.sum())
    score = float(np.clip(frac * 20, 0, 1))
    findings = []
    for i in comps:
        x, y, w, h, a = stats[i]
        qd = GHOST_QUALITIES[int(np.median(best[labels == i]))]
        findings.append(Finding(
            type="jpeg_ghost", score=round(float(np.clip(a / textured.sum() * 20, 0, 1)), 3),
            region=Region(x=int(x * BLOCK), y=int(y * BLOCK), width=int(w * BLOCK), height=int(h * BLOCK), page=page),
            description=f"Historial de compresión distinto (JPEG ghost, Q~{qd} vs Q~{GHOST_QUALITIES[mode]})",
        ))
    return MapResult(score, findings)


def analyze_image(img: np.ndarray, page: int, excl=None,
                  use_compression: bool = True) -> tuple[float, list[Finding], float]:
    """Devuelve (score_pixel 0..1, hallazgos, calidad_de_señal 0..1) para una página."""
    # ELA y JPEG ghost solo tienen sentido con historial de compresión con pérdida (JPEG); en PNG/BMP o en la
    # rasterización de un PDF vectorial reflejarían la conversión, no una edición.
    ela = (_analyze_map(ela_map(img), "ela_inconsistency", page, "Inconsistencia de compresión (ELA)", excl)
           if use_compression else MapResult(0.0, []))
    noi = _analyze_map(noise_map(img), "noise_inconsistency", page, "Inconsistencia del residual de ruido", excl)
    gho = ghost_analysis(img, page, excl) if use_compression else MapResult(0.0, [])
    # ELA y ruido sobre texto/fondo plano dan muchos falsos positivos: aportan como máximo 0.5.
    # El ghost JPEG (historial de compresión por región) es el detector de mayor peso.
    score = max(gho.score, 0.5 * max(ela.score, noi.score))
    # menos píxeles => señal menos fiable
    signal = float(np.clip(img.shape[0] * img.shape[1] / (1000 * 1000), 0.2, 1.0))
    return score, ela.findings + noi.findings + gho.findings, signal


# Fiabilidad de los detectores de píxeles según el tipo de entrada: sin historial JPEG
# (PNG/captura) o rasterizado desde PDF, ELA y ghost aportan mucho menos.
INPUT_RELIABILITY = {"jpeg": 0.85, "png": 0.45, "pdf": 0.5}


def combine(page_scores: list[float], signals: list[float], meta_weight: float,
            input_kind: str = "jpeg") -> tuple[float, float, list[str]]:
    """Combina píxeles y metadatos en (probabilidad, confianza, advertencias).

    La confianza es la FIABILIDAD del análisis para este tipo de archivo y resolución,
    no una función de la probabilidad: un 0% sobre un PNG no es "95% seguro de que está limpio".
    """
    pixel = max(page_scores) if page_scores else 0.0
    meta = float(np.clip(meta_weight, 0, 0.6))
    prob = float(np.clip(1 - (1 - pixel) * (1 - meta), 0, 1))
    res = float(np.mean(signals)) if signals else 0.2
    reliability = INPUT_RELIABILITY.get(input_kind, 0.5) * (0.5 + 0.5 * res)
    corroboration = 0.15 if (pixel > 0.15 and meta > 0) else 0.0   # píxeles y metadatos coinciden
    confidence = float(np.clip(0.15 + 0.6 * reliability + corroboration, 0.1, 0.9))
    warnings = []
    if input_kind == "png":
        warnings.append("Formato sin historial de compresión JPEG (PNG/captura): la detección de ediciones es poco fiable.")
    elif input_kind == "pdf":
        warnings.append("PDF rasterizado: el análisis de píxeles pierde las estadísticas de las imágenes originales.")
    if res < 0.5:
        warnings.append("Resolución baja: el análisis es menos fiable.")
    if prob < 0.4 and confidence < 0.6:
        warnings.append("Una probabilidad baja no descarta una alteración con esta fiabilidad.")
    return round(prob, 3), round(confidence, 3), warnings


