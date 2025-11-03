// components/AreaStatsDialog.jsx
import { useEffect, useState, useRef } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Divider,
  Paper,
} from "@mui/material";
import Draggable from "react-draggable";

function PaperComponent(props) {
  const nodeRef = useRef(null);
  return (
    <Draggable
      nodeRef={nodeRef}
      handle="#draggable-dialog-title"
      cancel={'[class*="MuiDialogContent-root"]'}
    >
      <Paper {...props} ref={nodeRef} />
    </Draggable>
  );
}

/**
 * Props:
 * - open
 * - onClose
 * - requestFn: (payload) => Promise<{ noCoverage?: boolean, stats?: {min,max,mean,median,std,count,valid_pct}, hist?: {bins:number[],counts:number[]} }>
 * - payload: { filepath, field, product, elevation?, height?, filters?, polygon }
 */
export default function AreaStatsDialog({ open, onClose, requestFn, payload }) {
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError("");
    setResp(null);
    (async () => {
      try {
        const r = await requestFn(payload);
        setResp(r);
      } catch (e) {
        setError(e?.response?.data?.detail || String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [open, requestFn, payload]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="sm"
      hideBackdrop
      disableEnforceFocus
      disableAutoFocus
      disableRestoreFocus
      disableScrollLock
      slotProps={{
        root: { sx: { pointerEvents: "none" } },
      }}
      PaperProps={{
        sx: { pointerEvents: "auto" },
      }}
      PaperComponent={PaperComponent}
      aria-labelledby="draggable-dialog-title"
    >
      <DialogTitle id="draggable-dialog-title">
        Estadísticas del área
      </DialogTitle>
      <DialogContent dividers>
        {loading && <Typography>Calculando…</Typography>}
        {error && <Typography color="error">{error}</Typography>}
        {resp?.noCoverage && (
          <Typography>No hay cobertura de datos en el polígono.</Typography>
        )}

        {resp?.stats && (
          <Box>
            <Typography variant="subtitle1">Resumen</Typography>
            <Box
              component="table"
              sx={{ mt: 1, width: "100%", "& td": { py: 0.5 } }}
            >
              <tbody>
                <tr>
                  <td>
                    <b>Mín</b>
                  </td>
                  <td>{resp.stats.min}</td>
                </tr>
                <tr>
                  <td>
                    <b>Máx</b>
                  </td>
                  <td>{resp.stats.max}</td>
                </tr>
                <tr>
                  <td>
                    <b>Media</b>
                  </td>
                  <td>{resp.stats.mean}</td>
                </tr>
                <tr>
                  <td>
                    <b>Mediana</b>
                  </td>
                  <td>{resp.stats.median}</td>
                </tr>
                <tr>
                  <td>
                    <b>Desv. std</b>
                  </td>
                  <td>{resp.stats.std}</td>
                </tr>
                <tr>
                  <td>
                    <b>Pixeles válidos</b>
                  </td>
                  <td>{resp.stats.count}</td>
                </tr>
                <tr>
                  <td>
                    <b>% válido</b>
                  </td>
                  <td>{resp.stats.valid_pct}%</td>
                </tr>
              </tbody>
            </Box>

            {resp.hist && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1">Histograma</Typography>
                <Typography variant="body2" sx={{ opacity: 0.7 }}>
                  {resp.hist.bins.length} bins
                </Typography>
              </>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cerrar</Button>
      </DialogActions>
    </Dialog>
  );
}
