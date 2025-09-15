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

const MARKS_01 = [
  { value: 0, label: "0" },
  { value: 0.25, label: "0.25" },
  { value: 0.5, label: "0.5" },
  { value: 0.75, label: "0.75" },
  { value: 1, label: "1" },
];

export default function ProductSelectorDialog({
  open,
  onClose,
  onConfirm,
  initialProduct = "ppi",
  initialCappiHeight = 2000,
  initialElevation = 0,
  initialFilters = {
    excludeTransition: true,
    rhohv: { enabled: true, min: 0.92 },
    other: { enabled: false, min: 0.8 },
  },
}) {
  const [product, setProduct] = useState(initialProduct);
  const [height, setHeight] = useState(initialCappiHeight);
  const [elevation, setElevation] = useState(initialElevation);
  const [filters, setFilters] = useState(structuredClone(initialFilters));

  const isCAPPI = product === "cappi";
  const isPPI = product === "ppi";
  const FILTER_KEYS = { RHOHV: "RHOHV", OTHER: "Other" };

  const resetState = () => {
    setProduct(initialProduct);
    setHeight(initialCappiHeight);
    setElevation(initialElevation);
    setFilters(structuredClone(initialFilters));
  };

  const handleClose = () => {
    resetState();
    onClose();
  };

  const handleAccept = () => {
    const pairs = [];
    if (filters.rhohv?.enabled) {
      pairs.push([FILTER_KEYS.RHOHV, Number(filters.rhohv.min)]);
    }

    onConfirm({
      product,
      height: isCAPPI ? height : undefined,
      elevation: isPPI ? elevation : undefined,
      filters: pairs,
    });
    onClose();
  };

  const setRhohv = (patch) =>
    setFilters((f) => ({ ...f, rhohv: { ...f.rhohv, ...patch } }));
  const setOther = (patch) =>
    setFilters((f) => ({ ...f, other: { ...f.other, ...patch } }));

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Configuración</DialogTitle>

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
        <Typography variant="subtitle1" gutterBottom>
          Filtros
        </Typography>

        <FormGroup>
          {/* Excluir transición */}
          <FormControlLabel
            control={
              <Checkbox
                checked={!!filters.excludeTransition}
                onChange={(e) =>
                  setFilters((f) => ({
                    ...f,
                    excludeTransition: e.target.checked,
                  }))
                }
              />
            }
            label="Excluir rayos de transición"
          />
        </FormGroup>

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
          <Box display="flex" alignItems="center" gap={2} pl={5}>
            <Slider
              value={Number(filters.rhohv?.min ?? 0.92)}
              onChange={(_, v) => setRhohv({ min: v })}
              step={0.01}
              min={0}
              max={1}
              marks={MARKS_01}
              valueLabelDisplay="auto"
              disabled={!filters.rhohv?.enabled}
              sx={{ flex: 1 }}
            />
            <TextField
              type="number"
              size="small"
              value={Number(filters.rhohv?.min ?? 0.92)}
              onChange={(e) =>
                setRhohv({
                  min: Math.max(0, Math.min(1, Number(e.target.value))),
                })
              }
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              disabled={!filters.rhohv?.enabled}
            />
          </Box>
        </Box>

        {/* Otro filtro 0–1 (renombrá si lo vas a usar) */}
        <Box mt={2} px={1}>
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
