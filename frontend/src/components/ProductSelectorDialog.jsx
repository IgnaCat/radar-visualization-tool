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
} from "@mui/material";

export default function ProductSelectorDialog({ open, onClose, onConfirm }) {
  const [selected, setSelected] = useState("ppi");

  const handleConfirm = () => {
    onConfirm(selected);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Seleccionar producto</DialogTitle>
      <DialogContent>
        <FormControl component="fieldset">
          <RadioGroup
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            <FormControlLabel value="PPI" control={<Radio />} label="PPI" />
            <FormControlLabel
              value="COLMAX"
              control={<Radio />}
              label="COLMAX"
            />
            <FormControlLabel value="CAPPI" control={<Radio />} label="CAPPI" />
          </RadioGroup>
        </FormControl>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="secondary">
          Cancelar
        </Button>
        <Button onClick={handleConfirm} color="primary" variant="contained">
          Aceptar
        </Button>
      </DialogActions>
    </Dialog>
  );
}
