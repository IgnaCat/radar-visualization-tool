import { useEffect, useState, useRef } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  TextField,
  MenuItem,
  Typography,
  Divider,
  Paper,
} from "@mui/material";
import Draggable from "react-draggable";

const FIELD_OPTIONS = ["DBZH", "KDP", "RHOHV", "ZDR"];

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

export default function PseudoRHIDialog({
  open,
  onClose,
  filepath, // string del archivo subido
  radarSite, // { lat, lon }
  fields_present = FIELD_OPTIONS,
  onRequestPickPoint, // fn -> activa pick mode en el mapa
  pickedPoint, // { lat, lon } desde el mapa
  onClearPickedPoint, // limpiar selección si hace falta
  onGenerate, // fn async que llama API (generatePseudoRHI)
}) {
  const [field, setField] = useState(fields_present[0] || "DBZH");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [loading, setLoading] = useState(false);
  const [resultImg, setResultImg] = useState(null);
  const [error, setError] = useState("");

  // sync coords desde el picker
  useEffect(() => {
    if (pickedPoint) {
      setLat(pickedPoint.lat.toFixed(6));
      setLon(pickedPoint.lon.toFixed(6));
    }
  }, [pickedPoint]);

  const handlePick = () => {
    setResultImg(null);
    setError("");
    onRequestPickPoint?.();
  };

  const handleGenerate = async () => {
    setResultImg(null);
    setError("");
    if (!filepath) {
      setError("Seleccione un archivo primero");
      return;
    }
    const end_lat = Number(lat);
    const end_lon = Number(lon);
    if (!Number.isFinite(end_lat) || !Number.isFinite(end_lon)) {
      setError("Lat/Lon inválidos");
      return;
    }
    try {
      setLoading(true);
      const resp = await onGenerate({
        filepath,
        field,
        end_lat,
        end_lon,
      });
      setResultImg(resp?.[0].image_url || null);
    } catch (e) {
      setError(e?.response?.data?.detail || String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    // setResultImg(null);
    setError("");
    onClearPickedPoint?.();
    onClose?.();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      fullWidth
      maxWidth="md"
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
        Pseudo-RHI (corte vertical)
      </DialogTitle>
      <DialogContent dividers>
        <Typography variant="body2" gutterBottom>
          Seleccioná un punto destino en el mapa
        </Typography>

        <Box display="grid" gridTemplateColumns="1fr 1fr" gap={2} mt={1}>
          <TextField
            select
            label="Campo"
            value={field}
            onChange={(e) => setField(e.target.value)}
          >
            {fields_present.map((f) => (
              <MenuItem key={f} value={f}>
                {f}
              </MenuItem>
            ))}
          </TextField>
        </Box>

        <Box display="grid" gridTemplateColumns="1fr 1fr auto" gap={2} mt={2}>
          <TextField
            label="Latitud"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
          />
          <TextField
            label="Longitud"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
          />
          <Button variant="outlined" onClick={handlePick}>
            Seleccionar en mapa
          </Button>
        </Box>

        {radarSite && (
          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            Centro del radar: lat {radarSite.lat.toFixed?.(4) ?? radarSite.lat},
            lon {radarSite.lon.toFixed?.(4) ?? radarSite.lon}
          </Typography>
        )}

        {error && (
          <Typography color="error" mt={2}>
            {error}
          </Typography>
        )}

        {resultImg && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box display="flex" justifyContent="center">
              <img
                src={resultImg}
                alt="pseudo-rhi"
                style={{ maxWidth: "100%", borderRadius: 8 }}
              />
            </Box>
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose} color="secondary">
          Cerrar
        </Button>
        <Button onClick={handleGenerate} variant="contained" disabled={loading}>
          {loading ? "Generando..." : "Generar corte"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
