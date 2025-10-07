import { useState } from "react";
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
  FormGroup,
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

const DEFAULT_LAYERS = [
  { id: "dbzh", label: "DBZH", enabled: true, opacity: 1 },
  { id: "kdp0.5", label: "KDP", enabled: false, opacity: 1 },
  { id: "rhohv0.5", label: "RHOHV", enabled: false, opacity: 1 },
  { id: "zdr0.5", label: "ZDR", enabled: false, opacity: 1 },
  { id: "z0.5", label: "Z", enabled: false, opacity: 1 },
  { id: "vrad", label: "VRAD", enabled: false, opacity: 1 },
];

export default function ProductSelectorDialog({
  open,
  onClose,
  onConfirm,
  initialLayers = DEFAULT_LAYERS,
  initialProduct = "ppi",
  initialCappiHeight = 2000,
  initialElevation = 0,
  initialFilters = {
    rhohv: { enabled: true, min: 0, max: 0.92 },
    other: { enabled: false, min: 0, max: 1.0 },
  },
}) {
  const [layers, setLayers] = useState(initialLayers);
  const [product, setProduct] = useState(initialProduct);
  const [height, setHeight] = useState(initialCappiHeight);
  const [elevation, setElevation] = useState(initialElevation);
  const [filters, setFilters] = useState(structuredClone(initialFilters));

  const isCAPPI = product === "cappi";
  const isPPI = product === "ppi";
  const FILTER_KEYS = { RHOHV: "RHOHV", OTHER: "Other" };

  const resetState = () => {
    setLayers(structuredClone(initialLayers));
    setProduct(initialProduct);
    setHeight(initialCappiHeight);
    setElevation(initialElevation);
    setFilters(structuredClone(initialFilters));
  };

  const setRhohv = (patch) =>
    setFilters((f) => ({ ...f, rhohv: { ...f.rhohv, ...patch } }));
  const setOther = (patch) =>
    setFilters((f) => ({ ...f, other: { ...f.other, ...patch } }));

  const clamp01 = (v) => Math.max(0, Math.min(1, Number(v)));

  const handleAccept = () => {
    // normalizar y construir payload de filtros
    const out = [];

    if (filters.rhohv?.enabled) {
      let min = clamp01(filters.rhohv.min ?? 0);
      let max = clamp01(filters.rhohv.max ?? 1);
      if (min > max) [min, max] = [max, min]; // swap si vienen invertidos
      out.push({
        field: "RHOHV",
        type: "range",
        min: min,
        max: max,
      });
    }

    onConfirm({
      layers,
      product,
      height: isCAPPI ? height : undefined,
      elevation: isPPI ? elevation : undefined,
      filters: out,
    });
    onClose();
  };

  const handleClose = () => {
    resetState();
    onClose();
  };

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

        <Box mt={2}>
          <LayerControlList items={layers} onChange={setLayers} />
        </Box>

        {isPPI && (
          <Box mt={2}>
            <Typography variant="subtitle1" gutterBottom>
              Seleccionar elevación
            </Typography>
            <Box px={1}>
              <Slider
                value={elevation}
                onChange={(_, v) => setElevation(v)}
                step={1}
                min={0}
                max={12}
                marks={[
                  { value: 0, label: "0" },
                  { value: 2, label: "2" },
                  { value: 4, label: "4" },
                  { value: 6, label: "6" },
                  { value: 8, label: "8" },
                  { value: 10, label: "10" },
                  { value: 12, label: "12" },
                ]}
                valueLabelDisplay="auto"
              />
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

        {/* ---- Filtros ---- */}
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

        {/* Otro filtro 0–1 (renombrá si lo vas a usar) */}
        <Box mt={2} mb={4} px={1}>
          <FormControlLabel
            control={
              <Checkbox
                checked={!!filters.other?.enabled}
                onChange={(e) => setOther({ enabled: e.target.checked })}
              />
            }
            label="Otro filtro"
          />
          <Box display="flex" alignItems="center" gap={2} pl={5}>
            <Slider
              value={Number(filters.other?.min ?? 0.8)}
              onChange={(_, v) => setOther({ min: v })}
              step={0.01}
              min={0}
              max={1}
              marks={MARKS_01}
              valueLabelDisplay="auto"
              disabled={!filters.other?.enabled}
              sx={{ flex: 1 }}
            />
            <TextField
              type="number"
              size="small"
              value={Number(filters.other?.min ?? 0.8)}
              onChange={(e) =>
                setOther({
                  min: Math.max(0, Math.min(1, Number(e.target.value))),
                })
              }
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              disabled={!filters.other?.enabled}
            />
          </Box>
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
