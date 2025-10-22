import React, { useCallback } from "react";
import PropTypes from "prop-types";
import {
  Box,
  Checkbox,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Slider,
  Typography,
  Divider,
  Tooltip,
} from "@mui/material";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";

/**
 * items: [
 *   { id: 'colmax', label: 'COLMAX(Z)', enabled: true, opacity: 1 },
 *   { id: 'kdp0.5', label: 'KDP@0.5', enabled: false, opacity: 0.8 },
 *   ...
 * ]
 *
 * onChange(nextItems) -> devuelve lista actualizada (orden/estado/opacity)
 */
function LayerControlList({ title = "Productos de Radar", items, onChange }) {
  // --- Drag & Drop (HTML5 nativo) ---
  const onDragStart = useCallback((e, fromIdx) => {
    e.dataTransfer.setData("text/plain", String(fromIdx));
    e.dataTransfer.effectAllowed = "move";
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e, toIdx) => {
      e.preventDefault();
      const fromIdx = Number(e.dataTransfer.getData("text/plain"));
      if (Number.isNaN(fromIdx) || fromIdx === toIdx) return;
      const next = items.slice();
      const [moved] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, moved);
      onChange(next);
    },
    [items, onChange]
  );

  const toggleEnabled = (idx) => {
    const next = items.slice();
    const currentEnabledCount = next.filter((l) => l.enabled).length;

    // Si intenta habilitar y ya hay 3 habilitadas no permitir
    if (!next[idx].enabled && currentEnabledCount >= 3) {
      alert("Solo se pueden habilitar hasta 3 capas al mismo tiempo.");
      return;
    }

    next[idx] = { ...next[idx], enabled: !next[idx].enabled };
    onChange(next);
  };

  const changeOpacity = (idx, value) => {
    const next = items.slice();
    next[idx] = { ...next[idx], opacity: value };
    onChange(next);
  };

  return (
    <Box>
      <Typography
        variant="subtitle1"
        sx={{
          mb: 1,
          p: 1,
        }}
      >
        {title}
      </Typography>

      {items.length === 0 && (
        <Typography variant="body2" sx={{ p: 1 }}>
          No hay productos disponibles
        </Typography>
      )}

      <List disablePadding>
        {items.length > 0 &&
          items.map((it, idx) => (
            <Box
              key={it.id}
              draggable
              onDragStart={(e) => onDragStart(e, idx)}
              onDragOver={onDragOver}
              onDrop={(e) => onDrop(e, idx)}
              sx={{
                p: 1,
                mb: 1,
                width: "90%",
                bgcolor: "background.paper",
              }}
            >
              {/* Nombre arriba */}
              <ListItem disableGutters sx={{ py: 0 }}>
                <ListItemText
                  primary={<Typography variant="body1">{it.label}</Typography>}
                />
              </ListItem>

              {/* Abajo: checkbox + slider + manija de arrastre */}
              <Box display="flex" alignItems="center" gap={1}>
                <Checkbox
                  checked={!!it.enabled}
                  onChange={() => toggleEnabled(idx)}
                  disabled={
                    !it.enabled && items.filter((l) => l.enabled).length >= 3
                  }
                  inputProps={{ "aria-label": `activar ${it.label}` }}
                />

                <Slider
                  value={Number(it.opacity ?? 1)}
                  onChange={(_, v) => changeOpacity(idx, v)}
                  min={0}
                  max={1}
                  step={0.01}
                  valueLabelDisplay="auto"
                  sx={{ flex: 1 }}
                />

                <Tooltip title="Arrastrar para cambiar el orden">
                  <IconButton size="small" sx={{ cursor: "grab", px: 1 }}>
                    <DragIndicatorIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          ))}
      </List>

      <Divider sx={{ mt: 1 }} />
    </Box>
  );
}

LayerControlList.propTypes = {
  title: PropTypes.string,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      enabled: PropTypes.bool.isRequired,
      opacity: PropTypes.number.isRequired, // 0..1
    })
  ).isRequired,
  onChange: PropTypes.func.isRequired,
};

export default LayerControlList;
