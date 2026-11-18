import axios from "axios";
import type { AnalysisResult, HistoryItem } from "../types";

const http = axios.create({ baseURL: "/api/v1/documents" });

export const MAX_BYTES = 20 * 1024 * 1024;
export const ACCEPTED = {
  "application/pdf": [".pdf"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
};

export function validateFile(file: File): string | null {
  if (!Object.keys(ACCEPTED).includes(file.type)) return "Formato no permitido. Use PDF, JPG o PNG.";
  if (file.size > MAX_BYTES) return "El archivo supera el máximo de 20 MB.";
  return null;
}

export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) return e.response?.data?.detail ?? e.message;
  return e instanceof Error ? e.message : "Error desconocido";
}

export const DocumentService = {
  async upload(file: File, onProgress: (pct: number) => void): Promise<string> {
    const form = new FormData();
    form.append("file", file);
    const { data } = await http.post("/upload", form, {
      onUploadProgress: (e) => e.total && onProgress(Math.round((e.loaded / e.total) * 100)),
    });
    return data.document_id;
  },
  analyze: (id: string) => http.post(`/analyze/${id}`).then((r) => r.data),
  results: (id: string) => http.get<AnalysisResult>(`/results/${id}`).then((r) => r.data),
  history: () => http.get<HistoryItem[]>("").then((r) => r.data),

  async waitForResult(id: string, intervalMs = 1000, timeoutMs = 120_000): Promise<AnalysisResult> {
    const start = Date.now();
    for (;;) {
      const r = await DocumentService.results(id);
      if (r.status === "completed") return r;
      if (r.status === "failed") throw new Error(r.error ?? "El análisis falló");
      if (Date.now() - start > timeoutMs) throw new Error("Tiempo de espera agotado");
      await new Promise((res) => setTimeout(res, intervalMs));
    }
  },
};
