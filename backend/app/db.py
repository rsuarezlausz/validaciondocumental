"""Persistencia SQLite (stdlib) de metadatos y resultados."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from .config import DB_PATH
from .models import AnalysisResult, HistoryItem


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, uploaded_at TEXT NOT NULL,
                status TEXT NOT NULL, tampering_probability REAL, result_json TEXT, error TEXT)"""
        )


def create_document(doc_id: str, filename: str) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as c:
        c.execute(
            "INSERT INTO documents (id, filename, uploaded_at, status) VALUES (?,?,?,?)",
            (doc_id, filename, now, "uploaded"),
        )
    return now


def set_status(doc_id: str, status: str, error: Optional[str] = None) -> None:
    with _conn() as c:
        c.execute("UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id))


def claim_for_processing(doc_id: str) -> bool:
    """Transición atómica uploaded/failed -> processing (evita doble análisis)."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE documents SET status='processing', error=NULL "
            "WHERE id=? AND status IN ('uploaded','failed')",
            (doc_id,),
        )
        return cur.rowcount == 1


def save_result(result: AnalysisResult) -> None:
    prob = result.forensic_analysis.tampering_probability if result.forensic_analysis else None
    with _conn() as c:
        c.execute(
            "UPDATE documents SET status=?, tampering_probability=?, result_json=?, error=NULL WHERE id=?",
            (result.status, prob, result.model_dump_json(), result.document_id),
        )


def get_result(doc_id: str) -> Optional[AnalysisResult]:
    with _conn() as c:
        r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not r:
        return None
    if r["result_json"] and r["status"] == "completed":
        return AnalysisResult.model_validate_json(r["result_json"])
    return AnalysisResult(
        document_id=r["id"], filename=r["filename"], status=r["status"],
        uploaded_at=r["uploaded_at"], error=r["error"],
    )


def list_history() -> list[HistoryItem]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return [
        HistoryItem(
            document_id=r["id"], filename=r["filename"], uploaded_at=r["uploaded_at"],
            status=r["status"], tampering_probability=r["tampering_probability"],
        )
        for r in rows
    ]
