from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["uploaded", "processing", "completed", "failed"]


class Region(BaseModel):
    x: int
    y: int
    width: int
    height: int
    page: int = 1  # 1-indexed; coordenadas en píxeles de la imagen de preview


class Finding(BaseModel):
    type: Literal["ela_inconsistency", "noise_inconsistency", "jpeg_ghost", "metadata",
                  "glyph_copy_move", "seam_artifact", "local_inconsistency", "geometry_anomaly",
                  "barcode_mismatch", "barcode_unreadable", "rule_violation",
                  "flat_patch", "background_discontinuity", "patch_text_inconsistency",
                  "pdf_incremental_update", "pdf_producer_anomaly", "pdf_overlay_shape", "pdf_text_layer_mismatch"]
    region: Optional[Region] = None
    regions: list[Region] = []  # regiones adicionales (p.ej. los dos glifos de un copy-move)
    description: str
    score: float = Field(0.0, ge=0, le=1)
    severity: Optional[Literal["info", "low", "medium", "high"]] = None


class ForensicAnalysis(BaseModel):
    tampering_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    findings: list[Finding] = []
    metadata_issues: list[str] = []
    warnings: list[str] = []
    signals: dict = {}
    analysis_capabilities: dict = {}
    limitations: list[str] = []
    contributions: list[dict] = []
    verdict: Optional[str] = None
    integrity_hash: Optional[str] = None


class Paragraph(BaseModel):
    page: int
    text: str
    confidence: Optional[float] = None


class PageInfo(BaseModel):
    page: int
    width: int
    height: int
    url: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: Status


class AnalysisResult(BaseModel):
    document_id: str
    filename: str
    status: Status
    uploaded_at: str
    error: Optional[str] = None
    ocr_text: str = ""
    paragraphs: list[Paragraph] = []
    forensic_analysis: Optional[ForensicAnalysis] = None
    pages: list[PageInfo] = []
    processing_seconds: Optional[float] = None


class HistoryItem(BaseModel):
    document_id: str
    filename: str
    uploaded_at: str
    status: Status
    tampering_probability: Optional[float] = None

