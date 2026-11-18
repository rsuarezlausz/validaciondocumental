from dataclasses import dataclass, field
from typing import Optional

Severity = str  # "low" | "medium" | "high" | "info"


@dataclass
class Detection:
    """Hallazgo de una señal. bbox en píxeles de la imagen original: (x, y, w, h)."""
    type: str
    description: str
    severity: Severity = "medium"
    score: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None
    extra: dict = field(default_factory=dict)


def union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return x0, y0, x1 - x0, y1 - y0


def mask_value(value: str, keep: int = 3) -> str:
    """Enmascara datos personales: '26329713' -> '***713'."""
    v = str(value or "")
    return v if len(v) <= keep else "***" + v[-keep:]
