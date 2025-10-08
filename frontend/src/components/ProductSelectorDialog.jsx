import { useEffect, useMemo, useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  Typography,
  Box,
  Slider,
  TextField,
  InputAdornment,
  Checkbox,
  Divider,
} from "@mui/material";
import LayerControlList from "./LayerControlList";

const MARKS_01 = [
  { value: 0, label: "0" },
  { value: 0.25, label: "0.25" },
  { value: 0.5, label: "0.5" },
  { value: 0.75, label: "0.75" },
  { value: 1, label: "1" },
];

// Si llega un alias raro del archivo, lo “canonizamos”
const CANON = { dbzh: "DBZH", zdr: "ZDR", rhohv: "RHOHV", kdp: "KDP" };

function canonize(name = "") {
  const k = String(name).toLowerCase();
  return CANON[k] || name.toUpperCase();
}

// Crear capas a partir de fields_present
function deriveLayersFromFields(fields_present) {
  const uniq = Array.from(new Set((fields_present || []).map(canonize)));
  // Orden sugerido: DBZH primero si existe
  const order = ["DBZH", "KDP", "RHOHV", "ZDR"];
  const sorted = uniq.sort((a, b) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });
  return sorted.map((f, i) => ({
    id: f.toLowerCase(),
    label: f,
    field: f,
    enabled: i === 0,
    opacity: 1,
  }));
}

export default function ProductSelectorDialog({
  open,
  fields_present = ["DBZH"],
  elevations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  onClose,
  onConfirm,
  initialProduct = "ppi",
  initialCappiHeight = 2000,
  initialElevation = 0,
  initialFilters = {
    rhohv: { enabled: true, min: 0, max: 0.92 },
    other: { enabled: false, min: 0, max: 1.0 },
  },
}) {
  const derivedLayers = useMemo(
    () => deriveLayersFromFields(fields_present),
    [fields_present]
  );

  const [layers, setLayers] = useState(derivedLayers);
  const [product, setProduct] = useState(initialProduct);
  const [height, setHeight] = useState(initialCappiHeight);

  // Elevación: trabajemos con índices de elevación
  // (si pasan grados en initialElevation, lo mapeamos al índice más cercano)
  const initialElevationIndex = useMemo(() => {
    if (!Array.isArray(elevations) || elevations.length === 0) return 0;
    const idx =
      typeof initialElevation === "number" &&
      elevations.includes(initialElevation)
        ? elevations.indexOf(initialElevation)
        : elevations
            .map((deg, i) => [i, Math.abs(deg - (initialElevation || 0))])
            .sort((a, b) => a[1] - b[1])[0]?.[0] ?? 0;
    return Math.max(0, Math.min(elevations.length - 1, idx));
  }, [elevations, initialElevation]);

  const [elevationIdx, setElevationIdx] = useState(initialElevationIndex);
  const [filters, setFilters] = useState(structuredClone(initialFilters));

  // Si cambian props (ej. suben otro archivo), reseteamos estado dependiente
  useEffect(() => {
    setLayers(derivedLayers);
  }, [derivedLayers]);
  useEffect(() => {
    setElevationIdx(initialElevationIndex);
  }, [initialElevationIndex]);

  const isCAPPI = product === "cappi";
  const isPPI = product === "ppi";

  const setRhohv = (patch) =>
    setFilters((f) => ({ ...f, rhohv: { ...f.rhohv, ...patch } }));
  const setOther = (patch) =>
    setFilters((f) => ({ ...f, other: { ...f.other, ...patch } }));

  const clamp01 = (v) => Math.max(0, Math.min(1, Number(v)));

  const handleAccept = () => {
    const out = [];
    if (filters.rhohv?.enabled) {
      let min = clamp01(filters.rhohv.min ?? 0);
      let max = clamp01(filters.rhohv.max ?? 1);
      if (min > max) [min, max] = [max, min];
      out.push({ field: "RHOHV", type: "range", min, max, enabled: true });
    }

    onConfirm({
      layers,
      product,
      height: isCAPPI ? height : undefined,
      elevation: isPPI ? elevationIdx : undefined,
      filters: out,
    });
    onClose();
  };

  const handleClose = () => {
    setLayers(derivedLayers);
    setProduct(initialProduct);
    setHeight(initialCappiHeight);
    setElevationIdx(initialElevationIndex);
    setFilters(structuredClone(initialFilters));
    onClose();
  };

  // Marks del slider de elevación usando los grados reales, espaciando cada n para no saturar
  const elevMarks = useMemo(() => {
    if (!Array.isArray(elevations)) return [];
    const N = elevations.length;
    const step = N > 9 ? Math.ceil(N / 9) : 1; // máx 9 marcas visibles
    return elevations
      .map((deg, i) =>
        i % step === 0 ? { value: i, label: String(deg) } : null
      )
      .filter(Boolean);
  }, [elevations]);

  const maxIdx = Math.max(0, (elevations?.length || 1) - 1);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Opciones de Visualización</DialogTitle>

      <DialogContent dividers>
        <Typography variant="subtitle1" gutterBottom>
          Seleccionar Producto
        </Typography>
        <FormControl component="fieldset" fullWidth>
          <RadioGroup
            value={product}
            onChange={(e) => setProduct(e.target.value)}
          >
            <FormControlLabel value="ppi" control={<Radio />} label="PPI" />
            <FormControlLabel
              value="colmax"
              control={<Radio />}
              label="COLMAX"
            />
            <FormControlLabel value="cappi" control={<Radio />} label="CAPPI" />
          </RadioGroup>
        </FormControl>

        <Divider sx={{ my: 2 }} />

        {/* Variables reales del archivo */}
        <Box mt={2}>
          <LayerControlList items={layers} onChange={setLayers} />
        </Box>

        {isPPI && (
          <Box mt={2}>
            <Typography variant="subtitle1" gutterBottom>
              Seleccionar elevación (°)
            </Typography>
            <Box px={1}>
              <Slider
                value={elevationIdx}
                onChange={(_, v) => setElevationIdx(v)}
                step={1}
                min={0}
                max={maxIdx}
                marks={elevMarks}
                valueLabelDisplay="auto"
                valueLabelFormat={(i) => elevations?.[i] ?? i}
              />
              <Typography variant="caption" sx={{ opacity: 0.7 }}>
                Elevación seleccionada:{" "}
                {elevations?.[elevationIdx] ?? elevationIdx}°
              </Typography>
            </Box>
          </Box>
        )}

        {isCAPPI && (
          <Box mt={2}>
            <Typography variant="subtitle1" gutterBottom>
              Seleccionar altura (m)
            </Typography>
            <Box px={1}>
              <TextField
                fullWidth
                type="number"
                variant="outlined"
                value={height}
                onChange={(e) => setHeight(Number(e.target.value))}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">m</InputAdornment>
                  ),
                }}
              />
            </Box>
          </Box>
        )}

        {(isPPI || isCAPPI) && <Divider sx={{ my: 2 }} />}

        {/* ---- Filtros por rango ---- */}
        <Typography variant="subtitle1" gutterBottom mt={2}>
          Filtros
        </Typography>

        {/* RHOHV */}
        <Box mt={1} px={1}>
          <FormControlLabel
            control={
              <Checkbox
                checked={!!filters.rhohv?.enabled}
                onChange={(e) => setRhohv({ enabled: e.target.checked })}
              />
            }
            label="RHOHV"
          />
          <Box
            display="flex"
            alignItems="center"
            gap={2}
            pl={5}
            sx={{ flexWrap: "wrap" }}
          >
            <Slider
              value={[
                Number(filters.rhohv?.min ?? 0),
                Number(filters.rhohv?.max ?? 1),
              ]}
              onChange={(_, v) => {
                const [min, max] = v;
                setRhohv({ min, max });
              }}
              step={0.01}
              min={0}
              max={1}
              marks={MARKS_01}
              valueLabelDisplay="auto"
              disabled={!filters.rhohv?.enabled}
              sx={{ flex: 1, minWidth: 220 }}
            />
            <TextField
              type="number"
              size="small"
              label="Min"
              value={Number(filters.rhohv?.min ?? 0)}
              onChange={(e) => setRhohv({ min: clamp01(e.target.value) })}
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              disabled={!filters.rhohv?.enabled}
            />
            <TextField
              type="number"
              size="small"
              label="Max"
              value={Number(filters.rhohv?.max ?? 1)}
              onChange={(e) => setRhohv({ max: clamp01(e.target.value) })}
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              disabled={!filters.rhohv?.enabled}
            />
          </Box>
        </Box>

        {/* (dejé tu “otro filtro” igual) */}
        <Box mt={2} mb={4} px={1}>
          {/* ... */}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose} color="secondary">
          Cancelar
        </Button>
        <Button onClick={handleAccept} variant="contained">
          Aceptar
        </Button>
      </DialogActions>
    </Dialog>
  );
}
