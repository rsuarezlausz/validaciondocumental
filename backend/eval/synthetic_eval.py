"""Evaluación sintética y calibración de umbrales.

Genera documentos sintéticos tipo DNI (sin datos personales), crea ediciones controladas con ground truth
(copiar un carácter del mismo documento sobre otro, y una costura fina), mide AUC / precisión / recall por
señal y fusionada, calcula umbrales óptimos (índice de Youden) y reporta falsos positivos por región.

Uso (desde backend/):
    python -m eval.synthetic_eval --n 30                 # solo reporta
    python -m eval.synthetic_eval --n 30 --write         # además escribe eval/calibrated_thresholds.yaml
LIMITACIÓN: documentos sintéticos != fotos reales. Los umbrales deben recalibrarse con documentos propios
del equipo (genuinos y editados) antes de usarlos como evidencia.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import synth  # noqa: E402
from app import forensic_analyzer as fa  # noqa: E402
from app.forensics import analyzer, glyphs as G, score_fusion, settings, template  # noqa: E402

OUT = Path(__file__).resolve().parent
SIGNALS = ["glyph_copy_move", "seam_artifact", "local_inconsistency", "geometry_anomaly", "ela", "noise", "jpeg_ghost", "fused"]


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def youden(pos: list[float], neg: list[float]) -> tuple[float, float, float, float]:
    best = (0.0, 0.5, 0.0, 0.0)  # (J, thr, tpr, fpr)
    for thr in sorted(set(pos + neg + [0.0])):
        tpr = np.mean([p > thr for p in pos]) if pos else 0.0
        fpr = np.mean([n > thr for n in neg]) if neg else 0.0
        if tpr - fpr > best[0]:
            best = (float(tpr - fpr), float(thr), float(tpr), float(fpr))
    return best


def score_doc(img: np.ndarray) -> dict[str, float]:
    cfg = settings.thresholds()["glyph_copy_move"]
    old = cfg["min_score_report"]
    cfg["min_score_report"] = 0.0          # puntuación gradual para la curva ROC
    try:
        rep = analyzer.run(img, "")
    finally:
        cfg["min_score_report"] = old
    s = {k: 0.0 for k in SIGNALS}
    for d in rep.detections:
        if d.type in s:
            s[d.type] = max(s[d.type], d.score)
    ps, findings, sig = fa.analyze_image(img, 1, template.excluded_mask(img))
    for f in findings:
        key = {"ela_inconsistency": "ela", "noise_inconsistency": "noise", "jpeg_ghost": "jpeg_ghost"}.get(f.type)
        if key:
            s[key] = max(s[key], f.score)
    pixel = {k: s[k] for k in ("ela", "noise", "jpeg_ghost")}
    fus = score_fusion.fuse([d for d in rep.detections if d.type != "glyph_copy_move" or d.score >= old],
                            pixel, 0.0, "jpeg", 1.0, {"barcode": "unreadable", "glyphs": rep.glyph_count,
                                                      "copy_move_applicable": rep.copy_move_applicable})
    s["fused"] = fus.probability
    return s, rep


def edit_copy_glyph(img, rng):
    glyphs, _ = G.segment_glyphs(img)
    lines = {k: v for k, v in G.lines_of(glyphs).items() if len(v) >= 8}
    gl = sorted(lines[rng.choice(list(lines))], key=lambda g: g.x)
    i = rng.randrange(1, len(gl))
    j = rng.choice([k for k in range(len(gl)) if abs(k - i) >= 1 and k != i])
    src, dst = gl[j], gl[i]
    dx = dst.x + (dst.w - src.w) // 2   # centrado: cubre el carácter original
    return synth.paste_region(img, src.bbox, (dx, dst.y), margin=max(6, (dst.w - src.w) // 2 + 6))


def edit_seam(img, rng):
    glyphs, _ = G.segment_glyphs(img)
    line = sorted(rng.choice([v for v in G.lines_of(glyphs).values() if len(v) >= 8]), key=lambda g: g.x)
    x = line[-1].x + line[-1].w + 20
    y0, y1 = line[0].y - 5, line[0].y + line[0].h + 30
    out = img.copy()
    out[y0:y1, x:x + 1] = np.clip(out[y0:y1, x:x + 1].astype(int) + 80, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = {"clean": [], "copy": [], "seam": []}
    times = []
    for k in range(a.n):
        q = rng.choice([90, 95])
        img, _ = synth.make_card(seed=k, jpeg_q=q, noise=rng.choice([2.5, 3.0, 3.5]))
        for kind, im in (("clean", img),
                         ("copy", synth.jpeg(edit_copy_glyph(img, rng), rng.choice([95, 98]))),
                         ("seam", edit_seam(img, rng))):
            t = time.perf_counter()
            s, rep = score_doc(im)
            times.append(time.perf_counter() - t)
            rows[kind].append(s)
    report = {"n_per_class": a.n, "mean_seconds_per_document": round(float(np.mean(times)), 2), "signals": {}}
    cal = {}
    target = {"glyph_copy_move": "copy", "seam_artifact": "seam", "fused": None}
    for sig in SIGNALS:
        if sig in ("fused",):
            pos = [r[sig] for r in rows["copy"] + rows["seam"]]
        elif sig in target:
            pos = [r[sig] for r in rows[target[sig]]]
        else:
            pos = [r[sig] for r in rows["copy"] + rows["seam"]]     # señales de apoyo: ¿reaccionan a cualquier edición?
        neg = [r[sig] for r in rows["clean"]]
        j, thr, tpr, fpr = youden(pos, neg)
        report["signals"][sig] = {"auc": round(auc(pos, neg), 3), "best_threshold": round(thr, 3),
                                  "recall": round(tpr, 3), "false_positive_rate": round(fpr, 3),
                                  "precision": round(tpr * len(pos) / max(tpr * len(pos) + fpr * len(neg), 1e-9), 3)}
        cal[sig] = round(thr, 3)
    # falsos positivos por región (en sintéticos no hay foto/holograma: se reporta solo el fondo de texto)
    report["false_positives_by_region"] = {"photo": "n/a (sintético)", "hologram": "n/a (sintético)",
                                           "guilloche": "n/a (sintético)",
                                           "text_field": sum(1 for r in rows["clean"] if r["glyph_copy_move"] > 0.35)}
    report["note"] = "Datos sintéticos: recalibrar con documentos reales del equipo."
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if a.write:
        base = yaml.safe_load((ROOT / "app/forensics/config/thresholds.yaml").read_text(encoding="utf-8"))
        base["version"] = int(base.get("version", 1)) + 1
        base["source"] = f"synthetic-eval n={a.n} seed={a.seed} ({time.strftime('%Y-%m-%d')})"
        thr = max(cal.get("glyph_copy_move", 0.35), 0.35)
        base["glyph_copy_move"]["min_score_report"] = round(thr, 3)
        (OUT / "calibrated_thresholds.yaml").write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
                                                        encoding="utf-8")
        print("Escrito eval/calibrated_thresholds.yaml (copiar a app/forensics/config/thresholds.yaml tras revisar)")


if __name__ == "__main__":
    main()
