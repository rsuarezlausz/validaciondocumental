import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"      # aislado; se elimina tras el análisis
PAGES_DIR = DATA_DIR / "pages"         # previews renderizadas (almacenamiento seguro)
LOG_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "documents.db"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "150"))
OCR_LANGS = os.getenv("OCR_LANGS", "spa+eng")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

ALLOWED_MIME = {"application/pdf", "image/jpeg", "image/png"}

for d in (UPLOAD_DIR, PAGES_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
