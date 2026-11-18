import uuid
from pathlib import Path

import fitz
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import db
from ..config import ALLOWED_MIME, MAX_UPLOAD_BYTES, PAGES_DIR, UPLOAD_DIR
from ..models import AnalysisResult, HistoryItem, UploadResponse
from ..services.pipeline import find_upload, run_analysis

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

EXT = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png"}


def _sniff(head: bytes) -> str | None:
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload(file: UploadFile):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, "Tipo no permitido. Use PDF, JPG o PNG")
    doc_id = str(uuid.uuid4())
    tmp = UPLOAD_DIR / f"{doc_id}.part"
    size, head = 0, b""
    try:
        with tmp.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                if not head:
                    head = chunk[:8]
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"El archivo supera {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
                out.write(chunk)
        real = _sniff(head)
        if real is None or real != file.content_type:
            raise HTTPException(415, "El contenido no coincide con el tipo declarado")
        tmp.rename(UPLOAD_DIR / f"{doc_id}{EXT[real]}")
    finally:
        tmp.unlink(missing_ok=True)
    name = Path(file.filename or "documento").name[:200]
    db.create_document(doc_id, name)
    return UploadResponse(document_id=doc_id, filename=name, status="uploaded")


@router.post("/analyze/{document_id}", response_model=UploadResponse, status_code=202)
def analyze(document_id: str, background: BackgroundTasks):
    doc = db.get_result(document_id)
    if doc is None:
        raise HTTPException(404, "Documento no encontrado")
    if doc.status == "completed":
        return UploadResponse(document_id=document_id, filename=doc.filename, status="completed")
    if find_upload(document_id) is None:
        raise HTTPException(409, "El archivo original ya no está disponible")
    if not db.claim_for_processing(document_id):
        return UploadResponse(document_id=document_id, filename=doc.filename, status="processing")
    background.add_task(run_analysis, document_id)
    return UploadResponse(document_id=document_id, filename=doc.filename, status="processing")


@router.get("/results/{document_id}", response_model=AnalysisResult)
def results(document_id: str):
    r = db.get_result(document_id)
    if r is None:
        raise HTTPException(404, "Documento no encontrado")
    return r


@router.get("", response_model=list[HistoryItem])
def history():
    return db.list_history()


@router.get("/{document_id}/pages/{page}")
def page_image(document_id: str, page: int):
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(404, "No encontrado")
    p = PAGES_DIR / document_id / f"page-{page}.png"
    if not p.is_file():
        raise HTTPException(404, "Página no encontrada")
    return FileResponse(p, media_type="image/png")
