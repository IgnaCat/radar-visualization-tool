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
  IconButton,
  Collapse,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import Draggable from "react-draggable";
import RadarFilterControls from "./RadarFilterControls";

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
  filepath,
  radarSite,
  fields_present = FIELD_OPTIONS,
  onRequestPickPoint,
  pickedPoint,
  onClearPickedPoint,
  onGenerate,
  onLinePreviewChange,
  onAutoClose,
  onAutoReopen,
}) {
  const [field, setField] = useState(fields_present[0] || "DBZH");
  const [startLat, setStartLat] = useState("");
  const [startLon, setStartLon] = useState("");
  const [endLat, setEndLat] = useState("");
  const [endLon, setEndLon] = useState("");
  const [loading, setLoading] = useState(false);
  const [resultImg, setResultImg] = useState(null);
  const [error, setError] = useState("");
  const [filters, setFilters] = useState([]);
  const [pickTarget, setPickTarget] = useState(null); // 'start' | 'end' | null
  const [autoFlowActive, setAutoFlowActive] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [maxLengthKm, setMaxLengthKm] = useState(240);
  const [maxHeightKm, setMaxHeightKm] = useState(20);

  const handlePickStart = () => {
    setResultImg(null);
    setError("");
    setPickTarget("start");
    setAutoFlowActive(true);
    onRequestPickPoint?.();
    onAutoClose?.();
  };

  const handlePickEnd = () => {
    setResultImg(null);
    setError("");
    setPickTarget("end");
    setAutoFlowActive(true);
    onRequestPickPoint?.();
    onAutoClose?.();
  };

  // Map click handling: automatic chaining start -> end
  useEffect(() => {
    if (pickedPoint && pickTarget) {
      const lat = pickedPoint.lat.toFixed(6);
      const lon = pickedPoint.lon.toFixed(6);
      if (pickTarget === "start") {
        setStartLat(lat);
        setStartLon(lon);
        setPickTarget("end");
        onRequestPickPoint?.();
        return;
      }
      if (pickTarget === "end") {
        setEndLat(lat);
        setEndLon(lon);
        setPickTarget(null);
      }
    }
  }, [pickedPoint, pickTarget, onRequestPickPoint]);

  // Auto reopen when points are chosen
  useEffect(() => {
    // Reabrir cuando:
    // 1. El diálogo está cerrado
    // 2. El flujo automático está activo
    // 3. Ya no estamos en modo de selección (pickTarget === null)
    // 4. Hay al menos un punto final seleccionado
    if (
      !open &&
      autoFlowActive &&
      pickTarget === null &&
      endLat !== "" &&
      endLon !== ""
    ) {
      onAutoReopen?.();
      // Evitar re-aperturas repetidas
      setAutoFlowActive(false);
    }
  }, [
    open,
    autoFlowActive,
    pickTarget,
    endLat,
    endLon,
    onAutoReopen,
  ]);

  // Update preview line
  useEffect(() => {
    // Prioridad: usar punto de inicio explícito si existe
    // Fallback al origen del radar SOLO si no hay inicio explícito y no estamos en flujo de selección activo
    const hasExplicitStart = startLat !== "" && startLon !== "";
    const hasEnd = endLat !== "" && endLon !== "";

    let startPoint = null;
    if (hasExplicitStart) {
      startPoint = { lat: Number(startLat), lon: Number(startLon) };
    } else if (hasEnd && radarSite && !pickTarget) {
      // Si ya se eligió el fin pero no hay inicio explícito y no estamos en medio de elegir puntos,
      // usar el origen del radar como inicio implícito
      startPoint = { lat: radarSite.lat, lon: radarSite.lon };
    }

    onLinePreviewChange?.({
      start: startPoint,
      end: hasEnd ? { lat: Number(endLat), lon: Number(endLon) } : null,
    });
  }, [startLat, startLon, endLat, endLon, radarSite, pickTarget, onLinePreviewChange]);

  const handleGenerate = async () => {
    setResultImg(null);
    setError("");
    if (!filepath) {
      setError("Seleccione un archivo primero");
      return;
    }
    const sLat = Number(startLat);
    const sLon = Number(startLon);
    const eLat = Number(endLat);
    const eLon = Number(endLon);
    if (!Number.isFinite(eLat) || !Number.isFinite(eLon)) {
      setError("Lat/Lon de destino inválidos");
      return;
    }
    if ((startLat === "" || startLon === "") && radarSite) {
      // usar centro del radar como inicio implícito
    } else if (!Number.isFinite(sLat) || !Number.isFinite(sLon)) {
      setError("Lat/Lon de inicio inválidos");
      return;
    }
    try {
      setLoading(true);
      const resp = await onGenerate({
        filepath,
        field,
        start_lat: startLat === "" || startLon === "" ? undefined : sLat,
        start_lon: startLat === "" || startLon === "" ? undefined : sLon,
        end_lat: eLat,
        end_lon: eLon,
        filters,
        max_length_km: Number(maxLengthKm),
        max_height_km: Number(maxHeightKm),
      });
      setResultImg(resp?.[0].image_url || null);
    } catch (e) {
      setError(e?.response?.data?.detail || String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setError("");
    onClearPickedPoint?.();
    // Cancelar cualquier flujo automático pendiente
    setAutoFlowActive(false);
    setPickTarget(null);
    onClose?.();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
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
        Pseudo-RHI (corte vertical)
      </DialogTitle>
      <DialogContent id="draggable-dialog-title" dividers>
        <Typography variant="body2" gutterBottom>
          Seleccioná los puntos en el mapa
        </Typography>

        <Box display="grid" gridTemplateColumns="1fr" gap={2} mt={2}>
          <Box display="grid" gridTemplateColumns="1fr" gap={1}>
            <TextField
              select
              size="small"
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

          <Typography variant="subtitle2">Punto de inicio</Typography>
          <Box display="grid" gridTemplateColumns="1fr 1fr auto" gap={2}>
            <TextField
              size="small"
              label="Latitud inicio"
              value={startLat}
              onChange={(e) => setStartLat(e.target.value)}
              disabled={pickTarget === "end"}
            />
            <TextField
              size="small"
              label="Longitud inicio"
              value={startLon}
              onChange={(e) => setStartLon(e.target.value)}
              disabled={pickTarget === "end"}
            />
            <Button
              variant="outlined"
              onClick={handlePickStart}
              disabled={pickTarget === "end"}
            >
              Elegir en mapa
            </Button>
          </Box>
          {pickTarget === "end" && startLat !== "" && startLon !== "" && (
            <Typography variant="caption" sx={{ opacity: 0.7 }}>
              Seleccioná ahora el punto de fin en el mapa…
            </Typography>
          )}

          <Typography variant="subtitle2">Punto de fin</Typography>
          <Box display="grid" gridTemplateColumns="1fr 1fr auto" gap={2}>
            <TextField
              size="small"
              label="Latitud fin"
              value={endLat}
              onChange={(e) => setEndLat(e.target.value)}
              disabled={pickTarget === "start"}
            />
            <TextField
              size="small"
              label="Longitud fin"
              value={endLon}
              onChange={(e) => setEndLon(e.target.value)}
              disabled={pickTarget === "start"}
            />
            <Button
              variant="outlined"
              onClick={handlePickEnd}
              disabled={pickTarget === "start"}
            >
              Elegir en mapa
            </Button>
          </Box>
        </Box>

        {radarSite && (
          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            Centro del radar: lat {radarSite.lat.toFixed?.(4) ?? radarSite.lat},
            lon {radarSite.lon.toFixed?.(4) ?? radarSite.lon}
          </Typography>
        )}

        <Box mt={3}>
          <Box display="flex" alignItems="center" gap={1}>
            <IconButton
              size="small"
              onClick={() => setShowFilters((v) => !v)}
              aria-label={showFilters ? "Ocultar filtros" : "Mostrar filtros"}
            >
              {showFilters ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
            <Typography variant="subtitle2" sx={{ userSelect: "none" }}>
              Filtros
            </Typography>
          </Box>
          <Collapse in={showFilters} timeout="auto" unmountOnExit>
            <Box mt={1}>
              <RadarFilterControls
                selectedField={field}
                onFiltersChange={setFilters}
                showVariableFilterDefault={true}
              />
              <Box mt={2} display="grid" gridTemplateColumns="1fr 1fr" gap={2}>
                <TextField
                  size="small"
                  label="Distancia máx (km)"
                  type="number"
                  value={maxLengthKm}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (!Number.isFinite(v)) return;
                    setMaxLengthKm(Math.min(500, Math.max(1, v)));
                  }}
                  helperText="Rango horizontal del corte"
                />
                <TextField
                  size="small"
                  label="Altura máx (km)"
                  type="number"
                  value={maxHeightKm}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (!Number.isFinite(v)) return;
                    setMaxHeightKm(Math.min(30, Math.max(0.5, v)));
                  }}
                  helperText="Altura vertical del corte"
                />
              </Box>
            </Box>
          </Collapse>
        </Box>

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
