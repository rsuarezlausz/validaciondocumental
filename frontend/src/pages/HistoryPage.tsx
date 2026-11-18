import { useEffect, useState } from "react";
import { Alert, Chip, Paper, Table, TableBody, TableCell, TableHead, TableRow, Typography, CircularProgress, Link } from "@mui/material";
import { DocumentService, errorMessage } from "../api/documentService";
import { useStore } from "../store";
import type { HistoryItem } from "../types";

export default function HistoryPage() {
  const show = useStore((s) => s.show);
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    DocumentService.history().then(setItems).catch((e) => setError(errorMessage(e)));
  }, []);

  const reopen = async (id: string) => {
    try { show(await DocumentService.results(id)); } catch (e) { setError(errorMessage(e)); }
  };

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!items) return <CircularProgress />;
  return (
    <Paper variant="outlined">
      <Typography variant="h6" sx={{ p: 2 }}>Historial</Typography>
      <Table size="small">
        <TableHead>
          <TableRow><TableCell>Archivo</TableCell><TableCell>Fecha</TableCell><TableCell>Estado</TableCell><TableCell>Score</TableCell></TableRow>
        </TableHead>
        <TableBody>
          {items.length === 0 && <TableRow><TableCell colSpan={4}>Aún no hay documentos procesados.</TableCell></TableRow>}
          {items.map((i) => (
            <TableRow key={i.document_id} hover>
              <TableCell>
                {i.status === "completed"
                  ? <Link component="button" onClick={() => reopen(i.document_id)}>{i.filename}</Link>
                  : i.filename}
              </TableCell>
              <TableCell>{new Date(i.uploaded_at).toLocaleString()}</TableCell>
              <TableCell>{i.status}</TableCell>
              <TableCell>
                {i.tampering_probability != null && (
                  <Chip size="small" label={`${Math.round(i.tampering_probability * 100)}%`}
                    color={i.tampering_probability >= 0.7 ? "error" : i.tampering_probability >= 0.4 ? "warning" : "success"} />
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}
