import pathlib

# 1. models.py: add new finding types
p=pathlib.Path("models.py"); c=p.read_bytes().decode("utf-8"); nl="\r\n" if "\r\n" in c else "\n"; c=c.replace("\r\n","\n")
typ = '"barcode_mismatch", "barcode_unreadable", "rule_violation"'
assert typ in c
c=c.replace(typ, '"barcode_mismatch", "barcode_unreadable", "rule_violation",\n                  "flat_patch", "background_discontinuity", "patch_text_inconsistency",\n                  "pdf_incremental_update", "pdf_producer_anomaly", "pdf_overlay_shape", "pdf_text_layer_mismatch"')
c=c.replace('signals: dict = {}\n', 'signals: dict = {}\n    analysis_capabilities: dict = {}\n')
c=c.replace('contributions: list[dict] = []\n', 'contributions: list[dict] = []\n    verdict: Optional[str] = None\n')
p.write_bytes(c.replace("\n",nl).encode("utf-8"))

# 2. thresholds.yaml: add flat_patch section and weights
p=pathlib.Path("forensics/config/thresholds.yaml"); t=p.read_bytes().decode("utf-8"); nl="\r\n" if "\r\n" in t else "\n"; t=t.replace("\r\n","\n")
t=t.replace("medium_probability: 0.40\n  high_probability: 0.70","medium_probability: 0.40\n  high_probability: 0.70\n  inconclusive_threshold: 0.15   # por debajo: sin suficientes señales\n\nflat_patch:\n  min_px: 8\n  ring_width: 8\n  std_ratio_floor: 5.0      # ring_std / inside_std mínimo\n  hf_ratio_floor: 5.0\n  rectangularity_min: 0.65\n  scale_count: 3")
t=t.replace("    jpeg_ghost: 0.30\n    metadata: 0.20","    jpeg_ghost: 0.30\n    metadata: 0.20\n    flat_patch: 0.70\n    background_discontinuity: 0.35\n    patch_text_inconsistency: 0.45\n    pdf_incremental_update: 0.15\n    pdf_producer_anomaly: 0.10\n    pdf_overlay_shape: 0.60\n    pdf_text_layer_mismatch: 0.50")
t=t.replace("weak_types: [ela, noise, jpeg_ghost, metadata, geometry_anomaly, local_inconsistency]",
            "weak_types: [ela, noise, jpeg_ghost, metadata, geometry_anomaly, local_inconsistency, background_discontinuity]")
p.write_bytes(t.replace("\n",nl).encode("utf-8"))
print("models.yaml thresholds.yaml ok")
