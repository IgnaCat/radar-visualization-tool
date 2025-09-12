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
} from "@mui/material";

export default function ProductSelectorDialog({
  open,
  onClose,
  onConfirm,
  initialProduct = "ppi",
  initialCappiHeight = 2000,
  initialElevation = 0,
}) {
  const [product, setProduct] = useState(initialProduct);
  const [height, setHeight] = useState(initialCappiHeight);
  const [elevation, setElevation] = useState(initialElevation);

  const isCAPPI = product === "cappi";
  const isPPI = product === "ppi";

  const resetState = () => {
    setProduct(initialProduct);
    setHeight(initialCappiHeight);
    setElevation(initialElevation);
  };

  const handleClose = () => {
    resetState();
    onClose();
  };

  const handleAccept = () => {
    onConfirm({
      product,
      height: isCAPPI ? height : undefined,
      elevation: isPPI ? elevation : undefined,
    });
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Seleccionar producto</DialogTitle>

      <DialogContent dividers>
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
