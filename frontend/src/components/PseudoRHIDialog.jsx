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
  filepath, // string del archivo subido
  radarSite, // { lat, lon }
  fields_present = FIELD_OPTIONS,
  onRequestPickPoint, // fn -> activa pick mode en el mapa
  pickedPoint, // { lat, lon } desde el mapa
  onClearPickedPoint, // limpiar selección si hace falta
  onGenerate, // fn async que llama API (generatePseudoRHI)
  onLinePreviewChange, // opcional: ( { start: {lat,lon} | null, end: {lat,lon} | null } ) => void
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

  // sync coords desde el picker
  useEffect(() => {
    if (pickedPoint && pickTarget) {
      const lat = pickedPoint.lat.toFixed(6);
      const lon = pickedPoint.lon.toFixed(6);
      if (pickTarget === "start") {
        setStartLat(lat);
        setStartLon(lon);
      } else if (pickTarget === "end") {
        setEndLat(lat);
        setEndLon(lon);
      }
      setPickTarget(null);
    }
  }, [pickedPoint, pickTarget]);

  useEffect(() => {
    onLinePreviewChange?.({
      start:
        startLat !== "" && startLon !== ""
          ? { lat: Number(startLat), lon: Number(startLon) }
          : null,
      end:
        endLat !== "" && endLon !== ""
          ? { lat: Number(endLat), lon: Number(endLon) }
          : null,
    });
  }, [startLat, startLon, endLat, endLon]);

  const handlePickStart = () => {
    setResultImg(null);
    setError("");
    setPickTarget("start");
    onRequestPickPoint?.();
  };
  const handlePickEnd = () => {
    setResultImg(null);
    setError("");
    setPickTarget("end");
    onRequestPickPoint?.();
  };

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
    // start es opcional, si no se informa usamos el centro del radar (comportamiento anterior)
    if ((startLat === "" || startLon === "") && radarSite) {
      // no forzamos al usuario a completar start si hay radarSite
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
      <DialogContent id="draggable-dialog-title" dividers>
        <Typography variant="body2" gutterBottom>
          Seleccioná un punto destino en el mapa
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
            />
            <TextField
              size="small"
              label="Longitud inicio"
              value={startLon}
              onChange={(e) => setStartLon(e.target.value)}
            />
            <Button variant="outlined" onClick={handlePickStart}>
              Elegir en mapa
            </Button>
          </Box>

          <Typography variant="subtitle2">Punto de fin</Typography>
          <Box display="grid" gridTemplateColumns="1fr 1fr auto" gap={2}>
            <TextField
              size="small"
              label="Latitud fin"
              value={endLat}
              onChange={(e) => setEndLat(e.target.value)}
            />
            <TextField
              size="small"
              label="Longitud fin"
              value={endLon}
              onChange={(e) => setEndLon(e.target.value)}
            />
            <Button variant="outlined" onClick={handlePickEnd}>
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

        <RadarFilterControls
          selectedField={field}
          onFiltersChange={setFilters}
          showVariableFilterDefault={true}
        />

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
