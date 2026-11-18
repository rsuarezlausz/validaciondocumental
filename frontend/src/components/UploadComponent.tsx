import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { Alert, Box, LinearProgress, Paper, Typography, Button } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { ACCEPTED, DocumentService, MAX_BYTES, errorMessage, validateFile } from "../api/documentService";
import { useStore } from "../store";

type Phase = "idle" | "uploading" | "analyzing";

export default function UploadComponent({ onDone }: { onDone?: () => void }) {
  const show = useStore((s) => s.show);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handle = useCallback(async (file: File) => {
    const invalid = validateFile(file);
    if (invalid) return setError(invalid);
    setError(null);
    try {
      setPhase("uploading"); setProgress(0);
      const id = await DocumentService.upload(file, setProgress);
      setPhase("analyzing");
      await DocumentService.analyze(id);
      show(await DocumentService.waitForResult(id));
      onDone?.();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setPhase("idle");
    }
  }, [show, onDone]);

  const onDrop = useCallback((accepted: File[], rejected: FileRejection[]) => {
    if (rejected.length) return setError(rejected[0].errors[0].message);
    if (accepted[0]) void handle(accepted[0]);
  }, [handle]);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop, accept: ACCEPTED, maxSize: MAX_BYTES, multiple: false, noClick: true, disabled: phase !== "idle",
  });

  const busy = phase !== "idle";
  return (
    <Paper
      {...getRootProps()}
      variant="outlined"
      sx={{ p: 5, textAlign: "center", borderStyle: "dashed", borderWidth: 2,
            borderColor: isDragActive ? "primary.main" : "divider", bgcolor: isDragActive ? "action.hover" : "background.paper" }}
    >
      <input {...getInputProps()} data-testid="file-input" />
      <CloudUploadIcon sx={{ fontSize: 56 }} color="primary" />
      <Typography variant="h6">Arrastre un documento aquí</Typography>
      <Typography color="text.secondary" gutterBottom>PDF, JPG o PNG · máx. 20 MB</Typography>
      <Button variant="contained" onClick={open} disabled={busy}>Seleccionar archivo</Button>
      {busy && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="body2" gutterBottom>
            {phase === "uploading" ? `Subiendo… ${progress}%` : "Analizando (OCR + forense)…"}
          </Typography>
          <LinearProgress
            variant={phase === "uploading" ? "determinate" : "indeterminate"}
            value={progress}
          />
        </Box>
      )}
      {error && <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>{error}</Alert>}
    </Paper>
  );
}
