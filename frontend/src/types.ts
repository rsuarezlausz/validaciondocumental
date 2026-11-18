export type Status = "uploaded" | "processing" | "completed" | "failed";

export interface Region { x: number; y: number; width: number; height: number; page: number }
export interface Finding {
  type: "ela_inconsistency" | "noise_inconsistency" | "jpeg_ghost" | "metadata" | "glyph_copy_move"
    | "seam_artifact" | "local_inconsistency" | "geometry_anomaly" | "barcode_mismatch"
    | "barcode_unreadable" | "rule_violation" | "flat_patch" | "background_discontinuity"
    | "patch_text_inconsistency" | "pdf_incremental_update" | "pdf_producer_anomaly"
    | "pdf_overlay_shape" | "pdf_text_layer_mismatch";
  region: Region | null;
  regions?: Region[];
  description: string;
  score: number;
  severity?: "info" | "low" | "medium" | "high" | null;
}
export interface BarcodeField {
  field: string; ocr_value: string; barcode_value: string; match: boolean; status: string;
}
export interface Contribution { signal: string; score: number; weight: number; points: number; group: string }
export interface ForensicAnalysis {
  tampering_probability: number;
  confidence: number;
  findings: Finding[];
  metadata_issues: string[];
  warnings?: string[];
  signals?: {
    barcode_crosscheck?: { status: string; fields: BarcodeField[]; fields_found: string[]; has_expiry: boolean };
    glyph_copy_move?: { max_shape_ncc: number; max_residual_ncc: number; applicable?: boolean };
    seam?: { detected: boolean };
    local_consistency?: { outliers: number };
    timings?: Record<string, number>;
    [k: string]: unknown;
  };
  limitations?: string[];
  contributions?: Contribution[];
  analysis_capabilities?: { ran: string[]; skipped?: { detector: string; reason: string }[] };
  verdict?: string;
  integrity_hash: string | null;
}
export interface Paragraph { page: number; text: string; confidence: number | null }
export interface PageInfo { page: number; width: number; height: number; url: string }
export interface AnalysisResult {
  document_id: string;
  filename: string;
  status: Status;
  uploaded_at: string;
  error: string | null;
  ocr_text: string;
  paragraphs: Paragraph[];
  forensic_analysis: ForensicAnalysis | null;
  pages: PageInfo[];
  processing_seconds: number | null;
}
export interface HistoryItem {
  document_id: string;
  filename: string;
  uploaded_at: string;
  status: Status;
  tampering_probability: number | null;
}

