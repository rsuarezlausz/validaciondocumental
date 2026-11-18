import { useState } from "react";
import { Box, Button, Chip, Divider, Drawer, IconButton, LinearProgress, List, ListItem, ListItemText, Paper, Stack, Tab, Tabs, Typography, Alert, Table, TableHead, TableBody, TableRow, TableCell } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { useStore } from "../store";
import type { AnalysisResult, Finding } from "../types";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const riskColor = (p: number) => (p >= 0.7 ? "error" : p >= 0.4 ? "warning" : "success");

function OcrTab({ r }: { r: AnalysisResult }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(r.ocr_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <Box>
      <Button startIcon={<ContentCopyIcon />} onClick={copy} sx={{ mb: 2 }}>
        {copied ? "¡Copiado!" : "Copiar texto"}
      </Button>
      {r.paragraphs.length === 0 && <Alert severity="info">No se detectó texto.</Alert>}
      {r.paragraphs.map((p, i) => (
        <Paper key={i} variant="outlined" sx={{ p: 1.5, mb: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Pág. {p.page}{p.confidence != null && ` · confianza OCR ${p.confidence}%`}
          </Typography>
          <Typography sx={{ whiteSpace: "pre-wrap" }}>{p.text}</Typography>
        </Paper>
      ))}
    </Box>
  );
}

const LABELS: Record<Finding["type"], string> = {
  ela_inconsistency: "ELA", noise_inconsistency: "Ruido", jpeg_ghost: "Doble JPEG", metadata: "Metadatos",
  glyph_copy_move: "Copia de carácter", seam_artifact: "Costura", local_inconsistency: "Consistencia local",
  geometry_anomaly: "Geometría", barcode_mismatch: "Barcode ≠ texto", barcode_unreadable: "Barcode ilegible",
  rule_violation: "Regla (por verificar)",
  flat_patch: "Parche plano", background_discontinuity: "Fondo discontinuo",
  patch_text_inconsistency: "Texto en parche",
  pdf_incremental_update: "PDF incremental", pdf_producer_anomaly: "Productor PDF",
  pdf_overlay_shape: "Superposicion PDF", pdf_text_layer_mismatch: "Capa de texto PDF",
};
const LABELS_BY_SIGNAL: Record<string, string> = {
  barcode_mismatch: "Barcode vs texto", glyph_copy_move: "Copia de caracter", seam_artifact: "Costura",
  rule_violation: "Reglas de coherencia", local_inconsistency: "Consistencia local", geometry_anomaly: "Geometria",
  ela: "ELA", noise: "Ruido", jpeg_ghost: "Doble JPEG", metadata: "Metadatos",
  flat_patch: "Parche plano", background_discontinuity: "Fondo", patch_text_inconsistency: "Texto",
  pdf_overlay_shape: "Superposicion PDF", pdf_text_layer_mismatch: "Capa texto PDF",
};
const STATUS_ES: Record<string, string> = {
  match: "coincide", mismatch: "discrepancia", unreadable: "ilegible", partial: "parcial",
};
const SEVERITY_COLOR = {
  high: { fill: "rgba(255,0,0,.25)", stroke: "red" },
  medium: { fill: "rgba(255,165,0,.25)", stroke: "orange" },
  low: { fill: "rgba(255,235,59,.25)", stroke: "#c9b400" },
  info: { fill: "rgba(120,120,120,.15)", stroke: "gray" },
};
const SEVERITY_CHIP = { high: "error", medium: "warning", low: "default", info: "default" } as const;
const DEFAULT_LIMITATION =
  "Este análisis estima una probabilidad y muestra evidencia; no afirma que un documento sea falso ni auténtico.";

function ForensicTab({ r }: { r: AnalysisResult }) {
  const f = r.forensic_analysis;
  if (!f) return <Alert severity="warning">Sin resultado forense.</Alert>;
  const label: Record<Finding["type"], string> = LABELS;
  const bc = f.signals?.barcode_crosscheck;
  const contributions = f.contributions ?? [];
  // Con veredicto no concluyente el porcentaje es un puntaje heurístico poco fiable: se atenúa y se explica
  const inconclusive = f.verdict === "inconclusive";
  const scoreColor = inconclusive ? "text.secondary" : `${riskColor(f.tampering_probability)}.main`;
  return (
    <Box>
      {f.verdict && (
        <Alert severity={f.verdict === "evidence" ? "error" : inconclusive ? "warning" : "success"} sx={{ mb: 1 }} data-testid="verdict">
          {f.verdict === "evidence" && "Hay evidencia de posible alteración. Revise los hallazgos antes de decidir."}
          {inconclusive && "Resultado no concluyente: se requiere revisión manual. El porcentaje es un puntaje heurístico y no una conclusión de falsedad."}
          {f.verdict === "no_evidence" && "No se encontró evidencia de alteración con las señales disponibles (no garantiza autenticidad)."}
        </Alert>
      )}
      <Typography variant="overline">Puntaje de sospecha (heurístico, no calibrado)</Typography>
      <Typography variant="h3" color={scoreColor}>
        {pct(f.tampering_probability)}
      </Typography>
      <LinearProgress variant="determinate" value={f.tampering_probability * 100}
        color={inconclusive ? "inherit" : riskColor(f.tampering_probability)} sx={{ height: 10, borderRadius: 5, my: 1 }} />
      <Typography variant="body2">Fiabilidad del análisis: <b>{pct(f.confidence)}</b>
        {r.processing_seconds != null && ` · procesado en ${r.processing_seconds}s`}</Typography>
      {(f.warnings ?? []).map((w, i) => <Alert key={i} severity="warning" sx={{ mt: 1 }}>{w}</Alert>)}
      {f.integrity_hash && (
        <Typography variant="caption" color="text.secondary" sx={{ wordBreak: "break-all" }}>
          Hash de integridad: {f.integrity_hash}
        </Typography>
      )}
      <Stack direction="row" spacing={1} alignItems="center" sx={{ my: 1 }}>
        {f.verdict && <Chip size="small" label={f.verdict} color={f.verdict === "evidence" ? "error" : f.verdict === "inconclusive" ? "warning" : "success"} />}
        {f.analysis_capabilities && <Typography variant="caption" color="text.secondary">
          Detectores: {f.analysis_capabilities.ran.join(", ")} &middot; Omitidos: {(f.analysis_capabilities.skipped ?? []).map(s => s.detector).join(", ")}
        </Typography>}
      </Stack>
      {contributions.length > 0 && (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1">Desglose por señal</Typography>
          <Table size="small" aria-label="desglose-senales">
            <TableHead><TableRow><TableCell>Señal</TableCell><TableCell align="right">Score</TableCell>
              <TableCell align="right">Peso</TableCell><TableCell align="right">Puntos</TableCell></TableRow></TableHead>
            <TableBody>
              {contributions.map((c) => (
                <TableRow key={c.signal}>
                  <TableCell>{LABELS_BY_SIGNAL[c.signal] ?? c.signal}{c.group === "weak" ? " (débil)" : ""}</TableCell>
                  <TableCell align="right">{c.score.toFixed(2)}</TableCell>
                  <TableCell align="right">{c.weight.toFixed(2)}</TableCell>
                  <TableCell align="right">{c.points.toFixed(2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </>
      )}
      {bc && (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1">Código de barras vs texto impreso ({STATUS_ES[bc.status] ?? bc.status})</Typography>
          {bc.status === "unreadable" && (
            <Alert severity="info">No se pudo leer el PDF417. Es un dato informativo, no evidencia de fraude.</Alert>
          )}
          {bc.fields.length > 0 && (
            <Table size="small" aria-label="barcode-vs-ocr">
              <TableHead><TableRow><TableCell>Campo</TableCell><TableCell>Valor leído (OCR)</TableCell>
                <TableCell>Valor del barcode</TableCell><TableCell>Coincide</TableCell></TableRow></TableHead>
              <TableBody>
                {bc.fields.map((r) => (
                  <TableRow key={r.field}>
                    <TableCell>{r.field}</TableCell><TableCell>{r.ocr_value || "—"}</TableCell>
                    <TableCell>{r.barcode_value}</TableCell>
                    <TableCell>{r.status === "match" ? "Sí" : r.status === "mismatch" ? "No" : "No concluyente"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
          {bc.fields_found.length > 0 && !bc.has_expiry && (
            <Typography variant="caption" color="text.secondary">
              El barcode no trae la fecha de vencimiento: se valida con la regla de vigencia (por verificar).
            </Typography>
          )}
        </>
      )}
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1">Hallazgos ({f.findings.length})</Typography>
      {f.findings.length === 0 && <Alert severity="success">No se detectaron anomalías.</Alert>}
      <List dense>
        {f.findings.map((x, i) => (
          <ListItem key={i} disableGutters>
            <ListItemText
              primary={<Stack direction="row" spacing={1} alignItems="center">
                <Chip size="small" label={label[x.type]} color={SEVERITY_CHIP[x.severity ?? "low"]} /><span>{x.description}</span></Stack>}
              secondary={x.region && `Pág. ${x.region.page} · x:${x.region.x} y:${x.region.y} ${x.region.width}×${x.region.height}`}
            />
          </ListItem>
        ))}
      </List>
      <Divider sx={{ my: 2 }} />
      <Alert severity="info" icon={false} data-testid="limitations">
        <Typography variant="subtitle2">Limitaciones de este análisis</Typography>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          {(f.limitations && f.limitations.length > 0 ? f.limitations : [DEFAULT_LIMITATION]).map((l, i) => (
            <li key={i}><Typography variant="caption">{l}</Typography></li>
          ))}
        </ul>
      </Alert>
    </Box>
  );
}

function PreviewTab({ r }: { r: AnalysisResult }) {
  const [page, setPage] = useState(1);
  const info = r.pages.find((p) => p.page === page);
  const regions = (r.forensic_analysis?.findings ?? []).filter((f) => f.region && f.region.page === page);
  if (!info) return <Alert severity="info">Sin vista previa.</Alert>;
  return (
    <Box>
      {r.pages.length > 1 && (
        <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
          {r.pages.map((p) => (
            <Chip key={p.page} label={p.page} color={p.page === page ? "primary" : "default"} onClick={() => setPage(p.page)} />
          ))}
        </Stack>
      )}
      <Box sx={{ position: "relative", lineHeight: 0, border: 1, borderColor: "divider" }}>
        <img src={info.url} alt={`Página ${page}`} style={{ width: "100%", display: "block" }} />
        <svg viewBox={`0 0 ${info.width} ${info.height}`}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} data-testid="overlay">
          {regions.flatMap((f, i) => {
            const col = SEVERITY_COLOR[f.severity ?? "low"];
            const boxes = [f.region!, ...(f.regions ?? []).filter((r) => r.page === page)];
            return boxes.map((r, k) => (
              <rect key={`${i}-${k}`} x={r.x} y={r.y} width={r.width} height={r.height}
                fill={col.fill} stroke={col.stroke} strokeWidth={Math.max(2, info.width / 300)}>
                <title>{`${LABELS[f.type]} (${f.severity ?? "low"}): ${f.description}`}</title>
              </rect>
            ));
          })}
        </svg>
      </Box>
      <Typography variant="caption" color="text.secondary">
        Rojo: severidad alta · Naranja: media · Amarillo: baja ({regions.length} hallazgo(s) en esta página). Pase el cursor sobre cada región para ver la explicación.
      </Typography>
    </Box>
  );
}

export default function ResultSidebarComponent() {
  const { result, open, tab, setTab, close } = useStore();
  return (
    <Drawer anchor="right" open={open && !!result} onClose={close}
      PaperProps={{ sx: { width: { xs: "100%", md: 560 }, p: 2 } }}>
      {result && (
        <>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="h6" noWrap>{result.filename}</Typography>
            <IconButton onClick={close} aria-label="Cerrar"><CloseIcon /></IconButton>
          </Stack>
          <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
            <Tab label="Texto OCR" /><Tab label="Forense" /><Tab label="Vista previa" />
          </Tabs>
          <Box sx={{ overflowY: "auto" }}>
            {tab === 0 && <OcrTab r={result} />}
            {tab === 1 && <ForensicTab r={result} />}
            {tab === 2 && <PreviewTab r={result} />}
          </Box>
        </>
      )}
    </Drawer>
  );
}


