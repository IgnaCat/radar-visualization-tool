import { useState, useEffect } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Slider,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Tooltip,
  Checkbox,
  FormControlLabel,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

const WEIGHT_FUNC_OPTIONS = [
  { value: "nearest", label: "Nearest Neighbor (default)" },
  { value: "Barnes2", label: "Barnes2" },
  { value: "Cressman", label: "Cressman" },
];

/**
 * SettingsDialog — Configuración global de interpolación y sincronización temporal
 *
 * Props:
 *  open: boolean
 *  onClose: () => void
 *  onApply: ({ deltaT, weightFunc, maxNeighbors, smoothing }) => void
 *  initialSettings: { deltaT, weightFunc, maxNeighbors, smoothing }
 */
export default function SettingsDialog({
  open,
  onClose,
  onApply,
  initialSettings = {},
}) {
  const [deltaT, setDeltaT] = useState(initialSettings.deltaT ?? 0);
  const [weightFunc, setWeightFunc] = useState(
    initialSettings.weightFunc ?? "nearest",
  );
  const [maxNeighbors, setMaxNeighbors] = useState(
    initialSettings.maxNeighbors ?? 1,
  );
  const [smoothingEnabled, setSmoothingEnabled] = useState(
    initialSettings.smoothing?.enabled ?? false,
  );
  const [smoothingMethod, setSmoothingMethod] = useState(
    initialSettings.smoothing?.method ?? "median",
  );
  const [smoothingSigma, setSmoothingSigma] = useState(
    initialSettings.smoothing?.sigma ?? 0.8,
  );
  const [smoothingMedianSize, setSmoothingMedianSize] = useState(
    initialSettings.smoothing?.median_size ?? 3,
  );

  // Sincronizar con initialSettings cada vez que el diálogo se abre
  useEffect(() => {
    if (open) {
      setDeltaT(initialSettings.deltaT ?? 0);
      setWeightFunc(initialSettings.weightFunc ?? "nearest");
      setMaxNeighbors(initialSettings.maxNeighbors ?? 1);
      setSmoothingEnabled(initialSettings.smoothing?.enabled ?? false);
      setSmoothingMethod(initialSettings.smoothing?.method ?? "median");
      setSmoothingSigma(initialSettings.smoothing?.sigma ?? 0.8);
      setSmoothingMedianSize(initialSettings.smoothing?.median_size ?? 3);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const isNearest = weightFunc === "nearest";
  // nearest fuerza max_neighbors=1
  const effectiveMaxNeighbors = isNearest ? 1 : maxNeighbors;

  const handleWeightFuncChange = (e) => {
    const val = e.target.value;
    setWeightFunc(val);
    if (val === "nearest") setMaxNeighbors(1);
  };

  const handleMaxNeighborsInput = (e) => {
    if (isNearest) return;
    const v = parseInt(e.target.value, 10);
    if (!isNaN(v) && v >= 1 && v <= 500) setMaxNeighbors(v);
  };

  const handleDeltaTInput = (e) => {
    const v = parseInt(e.target.value, 10);
    if (!isNaN(v) && v >= 0 && v <= 600) setDeltaT(v);
  };

  const handleApply = () => {
    onApply({
      deltaT: Number(deltaT),
      weightFunc,
      maxNeighbors: effectiveMaxNeighbors,
      smoothing: {
        enabled: Boolean(smoothingEnabled),
        method: smoothingMethod,
        sigma: Number(smoothingSigma),
        median_size: Number(smoothingMedianSize),
        only_when_nearest: false,
      },
    });
    onClose();
  };

  const shouldSmoothBeDisabled = !smoothingEnabled;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle sx={{ px: 2, py: 1.5, fontSize: "1.1rem", lineHeight: 1.2 }}>
        Configuración
      </DialogTitle>

      <DialogContent
        dividers
        sx={{
          px: 2,
          py: 1.25,
          "& .MuiTypography-subtitle2": { fontSize: "0.9rem" },
          "& .MuiFormControlLabel-label": { fontSize: "0.86rem" },
          "& .MuiTypography-caption": { fontSize: "0.72rem" },
        }}
      >
        {/* ── Delta T ─────────────────────────────────────────── */}
        <Box sx={{ mb: 2 }}>
          <Box
            sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.75 }}
          >
            <Typography variant="subtitle2">
              Tolerancia temporal entre radares (ΔT)
            </Typography>
            <Tooltip
              title="Tiempo máximo de diferencia (en segundos) para considerar que dos radares distintos corresponden al mismo instante y mostrarlos juntos en el mismo frame."
              placement="right"
            >
              <InfoOutlinedIcon
                sx={{ fontSize: 15, color: "text.secondary" }}
              />
            </Tooltip>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.25 }}>
            <Slider
              value={Number(deltaT)}
              onChange={(_, val) => setDeltaT(val)}
              min={0}
              max={600}
              step={30}
              sx={{ flex: 1 }}
            />
            <TextField
              value={deltaT}
              onChange={handleDeltaTInput}
              inputProps={{ min: 0, max: 600, type: "number" }}
              size="small"
              sx={{ width: 84 }}
              InputProps={{
                endAdornment: (
                  <Typography variant="caption" sx={{ whiteSpace: "nowrap" }}>
                    s
                  </Typography>
                ),
              }}
            />
          </Box>
        </Box>

        <Divider sx={{ mb: 2 }} />

        {/* ── Interpolación ────────────────────────────────────── */}
        <Box>
          <Box
            sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 1.25 }}
          >
            <Typography variant="subtitle2">
              Interpolación espacial (operador W)
            </Typography>
            <Tooltip
              title="Cambiar función de peso o cantidad de vecinos invalida el operador W cacheado. El primer procesamiento con un nuevo valor puede tardar 2–5 min. El resultado se guarda en disco para usos posteriores."
              placement="right"
            >
              <InfoOutlinedIcon
                sx={{ fontSize: 15, color: "text.secondary" }}
              />
            </Tooltip>
          </Box>

          {/* Función de peso */}
          <FormControl fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel>Función de peso</InputLabel>
            <Select
              value={weightFunc}
              label="Función de peso"
              onChange={handleWeightFuncChange}
            >
              {WEIGHT_FUNC_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Max vecinos */}
          <Box>
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.75 }}
            >
              <Typography
                variant="body2"
                color={isNearest ? "text.disabled" : "text.primary"}
              >
                Máx. vecinos por punto de grilla
              </Typography>
              <Tooltip
                title="Limita cuántos gates de radar pueden contribuir a cada voxel. Menos vecinos = más rápido pero más áspero. 'Vecino más cercano' fija este valor en 1."
                placement="right"
              >
                <InfoOutlinedIcon
                  sx={{ fontSize: 15, color: "text.secondary" }}
                />
              </Tooltip>
            </Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.25 }}>
              <Slider
                value={effectiveMaxNeighbors}
                onChange={(_, val) => !isNearest && setMaxNeighbors(val)}
                min={1}
                max={100}
                step={1}
                disabled={isNearest}
                sx={{ flex: 1 }}
              />
              <TextField
                value={effectiveMaxNeighbors}
                onChange={handleMaxNeighborsInput}
                inputProps={{ min: 1, max: 500, type: "number" }}
                size="small"
                disabled={isNearest}
                sx={{ width: 84 }}
              />
            </Box>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Suavizado */}
          <Box>
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.75 }}
            >
              <Typography variant="subtitle2">Suavizado visual</Typography>
              <Tooltip
                title="Aplica un suavizado sobre la imagen final para reducir aspecto pixelado. No cambia la interpolación científica base."
                placement="right"
              >
                <InfoOutlinedIcon
                  sx={{ fontSize: 15, color: "text.secondary" }}
                />
              </Tooltip>
            </Box>

            <FormControlLabel
              control={
                <Checkbox
                  size="small"
                  checked={smoothingEnabled}
                  onChange={(e) => setSmoothingEnabled(e.target.checked)}
                />
              }
              label="Habilitar suavizado"
            />

            <FormControl fullWidth size="small" sx={{ mt: 0.75 }}>
              <InputLabel>Método</InputLabel>
              <Select
                value={smoothingMethod}
                label="Método"
                onChange={(e) => setSmoothingMethod(e.target.value)}
                disabled={shouldSmoothBeDisabled}
              >
                <MenuItem value="gaussian">Gaussiano</MenuItem>
                <MenuItem value="median">Mediana</MenuItem>
              </Select>
            </FormControl>

            {smoothingMethod === "gaussian" ? (
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1.25,
                  mt: 0.75,
                }}
              >
                <Slider
                  value={Number(smoothingSigma)}
                  onChange={(_, val) => setSmoothingSigma(Number(val))}
                  min={0}
                  max={3}
                  step={0.1}
                  disabled={shouldSmoothBeDisabled}
                  sx={{ flex: 1 }}
                />
                <TextField
                  value={smoothingSigma}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v) && v >= 0 && v <= 5) setSmoothingSigma(v);
                  }}
                  inputProps={{ min: 0, max: 5, step: 0.1, type: "number" }}
                  size="small"
                  disabled={shouldSmoothBeDisabled}
                  sx={{ width: 84 }}
                  InputProps={{
                    endAdornment: (
                      <Typography
                        variant="caption"
                        sx={{ whiteSpace: "nowrap" }}
                      >
                        σ
                      </Typography>
                    ),
                  }}
                />
              </Box>
            ) : (
              <Box sx={{ mt: 0.75 }}>
                <TextField
                  value={smoothingMedianSize}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!isNaN(v) && v >= 1 && v <= 15) {
                      setSmoothingMedianSize(v);
                    }
                  }}
                  inputProps={{ min: 1, max: 15, step: 1, type: "number" }}
                  size="small"
                  label="Ventana (px)"
                  disabled={shouldSmoothBeDisabled}
                  sx={{ width: 130 }}
                />
              </Box>
            )}
          </Box>
        </Box>
      </DialogContent>

      <DialogActions sx={{ px: 2, py: 1, gap: 0.5 }}>
        <Button onClick={onClose} size="small">
          Cancelar
        </Button>
        <Button onClick={handleApply} variant="contained" size="small">
          Aplicar
        </Button>
      </DialogActions>
    </Dialog>
  );
}
