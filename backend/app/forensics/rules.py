"""Reglas de coherencia entre fechas. Configurables en config/rules.yaml y TODAS marcadas "por verificar"."""
from datetime import date, timedelta

from . import settings
from .types import Detection


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:      # 29 de febrero
        return d.replace(year=d.year + years, day=28)


def _age(birth: date, at: date) -> int:
    return at.year - birth.year - ((at.month, at.day) < (birth.month, birth.day))


def evaluate(dates: dict[str, date], expiry_year: int | None = None,
             check_validity: bool = True) -> list[Detection]:
    """`dates` puede traer birth/issue/expiry (de OCR o barcode). `expiry_year`: año leído por OCR dirigido cuando
    el día/mes no es legible (queda bajo el holograma). Devuelve incumplimientos."""
    rules = settings.rules()["rules"]
    out: list[Detection] = []
    birth, issue, expiry = dates.get("birth"), dates.get("issue"), dates.get("expiry")

    def det(name: str, text: str, sev: str, score: float):
        out.append(Detection("rule_violation", f"[Regla por verificar: {name}] {text}", sev, score,
                             None, {"rule": name, "verified": bool(rules[name].get("verified", False))}))

    if issue and expiry and issue >= expiry:
        det("issue_before_expiry", "La fecha de emisión no es anterior al vencimiento.", "medium", 0.8)
    if birth and issue and birth >= issue:
        det("birth_before_issue", "La fecha de nacimiento no es anterior a la emisión.", "medium", 0.8)
    ev = rules.get("expected_validity") if check_validity else None   # vigencia: solo perfiles con fuente
    if ev and birth and issue and expiry and issue < expiry:
        if _age(birth, issue) >= ev["adult_min_age"]:
            expected = _add_years(issue, ev["adult_validity_years"])
            if abs((expiry - expected).days) > ev["tolerance_days"]:
                det("expected_validity",
                    f"El vencimiento ({expiry.year}) no coincide con la vigencia supuesta de "
                    f"{ev['adult_validity_years']} años desde la emisión ({expected.year}). "
                    "Supuesto sin verificar con la fuente oficial.", "medium", 0.6)
    if ev and birth and issue and not expiry and expiry_year and _age(birth, issue) >= ev["adult_min_age"]:
        expected = _add_years(issue, ev["adult_validity_years"])
        if expiry_year != expected.year:
            det("expected_validity",
                f"El año de vencimiento leído ({expiry_year}) no coincide con la vigencia supuesta de "
                f"{ev['adult_validity_years']} años desde la emisión ({expected.year}). "
                "Supuesto sin verificar con la fuente oficial.", "medium", 0.6)
    return out
