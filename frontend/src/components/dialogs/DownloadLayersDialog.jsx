import { useState, useEffect, useMemo } from "react";
import {
  Box,
  Button,
  Checkbox,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import DownloadIcon from "@mui/icons-material/Download";
import LayersIcon from "@mui/icons-material/Layers";

/**
 * DownloadLayersDialog - Diálogo para seleccionar qué capas GeoTIFF descargar.
 *
 * Props:
 * - open: boolean
 * - onClose: () => void
 * - layers: LayerResult[] (con anotación `.radar` añadida en App.jsx)
 * - onDownload: (selectedLayers: LayerResult[]) => void
 */
export default function DownloadLayersDialog({
  open,
  onClose,
  layers = [],
  onDownload,
}) {
  const [selected, setSelected] = useState(new Set());

  const downloadableLayers = useMemo(
    () => layers.filter((l) => l.image_url),
    [layers],
  );

  // Seleccionar todas las capas por defecto al abrir
  useEffect(() => {
    if (open) {
      setSelected(new Set(downloadableLayers.map((_, i) => i)));
    }
  }, [open, downloadableLayers]);

  const allSelected =
    selected.size === downloadableLayers.length &&
    downloadableLayers.length > 0;
  const someSelected = selected.size > 0;

  const toggleAll = () => {
    if (allSelected) {
      setSelected(new Set());
    } else {
      setSelected(new Set(downloadableLayers.map((_, i) => i)));
    }
  };

  const toggleOne = (idx) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const handleDownload = () => {
    const toDownload = downloadableLayers.filter((_, i) => selected.has(i));
    onDownload?.(toDownload);
    onClose();
  };

  const uniqueRadars = [
    ...new Set(downloadableLayers.map((l) => l.radar).filter(Boolean)),
  ];
  const isMultiRadar = uniqueRadars.length > 1;

  // Colores fijos para chips de radar (hasta 6 radares distintos)
  const RADAR_COLORS = [
    { bg: "rgba(25, 118, 210, 0.12)", color: "#1565C0" },
    { bg: "rgba(46, 125, 50, 0.12)", color: "#1B5E20" },
    { bg: "rgba(183, 28, 28, 0.12)", color: "#B71C1C" },
    { bg: "rgba(106, 27, 154, 0.12)", color: "#4A148C" },
    { bg: "rgba(230, 81, 0, 0.12)", color: "#E65100" },
    { bg: "rgba(0, 96, 100, 0.12)", color: "#006064" },
  ];
  const radarColorMap = Object.fromEntries(
    uniqueRadars.map((r, i) => [r, RADAR_COLORS[i % RADAR_COLORS.length]]),
  );

  const getFilename = (layer) => {
    if (!layer.image_url) return "";
    return layer.image_url.split("/").pop() || "";
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xs"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: "12px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
        },
      }}
    >
      {/* Header */}
      <DialogTitle sx={{ p: 0 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 2.5,
            pt: 2,
            pb: 1.5,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <LayersIcon sx={{ color: "primary.main", fontSize: "1rem" }} />
            <Typography
              variant="subtitle1"
              sx={{ fontWeight: 600, fontSize: "13px", lineHeight: 1.3 }}
            >
              Descargar capas GeoTIFF
            </Typography>
          </Box>
          <IconButton
            onClick={onClose}
            size="small"
            sx={{ color: "text.secondary", ml: 1 }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>

      <Divider />

      {/* Content */}
      <DialogContent sx={{ p: 0 }}>
        {downloadableLayers.length === 0 ? (
          <Box sx={{ px: 2.5, py: 3.5, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              No hay capas disponibles para descargar.
            </Typography>
          </Box>
        ) : (
          <>
            {/* Toggle seleccionar todo */}
            <Box
              sx={{
                px: 2,
                pt: 1.5,
                pb: 0.5,
                display: "flex",
                alignItems: "center",
              }}
            >
              <Checkbox
                size="small"
                checked={allSelected}
                indeterminate={someSelected && !allSelected}
                onChange={toggleAll}
                sx={{ p: 0.5, mr: 0.5 }}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ cursor: "pointer", userSelect: "none" }}
                onClick={toggleAll}
              >
                {allSelected ? "Deseleccionar todo" : "Seleccionar todo"}
              </Typography>
            </Box>

            {/* Lista de capas */}
            <List dense disablePadding sx={{ px: 1, pb: 1.5 }}>
              {downloadableLayers.map((layer, idx) => {
                const isSelected = selected.has(idx);
                const radarColor = layer.radar
                  ? radarColorMap[layer.radar]
                  : null;
                return (
                  <ListItem
                    key={`${layer.field}-${layer.source_file}-${idx}`}
                    dense
                    onClick={() => toggleOne(idx)}
                    sx={{
                      borderRadius: "8px",
                      cursor: "pointer",
                      mb: 0.5,
                      pr: 1,
                      backgroundColor: isSelected
                        ? "rgba(25, 118, 210, 0.06)"
                        : "transparent",
                      transition: "background-color 0.15s",
                      "&:hover": {
                        backgroundColor: isSelected
                          ? "rgba(25, 118, 210, 0.10)"
                          : "rgba(0, 0, 0, 0.04)",
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <Checkbox
                        size="small"
                        checked={isSelected}
                        onChange={() => toggleOne(idx)}
                        onClick={(e) => e.stopPropagation()}
                        sx={{ p: 0.5 }}
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 0.75,
                            flexWrap: "wrap",
                          }}
                        >
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: 500, fontSize: "12px" }}
                          >
                            {layer.field || "Campo desconocido"}
                          </Typography>
                          {isMultiRadar && layer.radar && radarColor && (
                            <Chip
                              label={layer.radar}
                              size="small"
                              sx={{
                                fontSize: "9px",
                                height: 18,
                                px: 0.25,
                                backgroundColor: radarColor.bg,
                                color: radarColor.color,
                                fontWeight: 600,
                                "& .MuiChip-label": { px: "6px" },
                              }}
                            />
                          )}
                        </Box>
                      }
                      secondary={
                        <Typography
                          component="span"
                          variant="caption"
                          sx={{
                            fontFamily: "monospace",
                            fontSize: "9px",
                            color: "text.disabled",
                            display: "block",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            maxWidth: 260,
                            mt: 0.25,
                          }}
                        >
                          {getFilename(layer)}
                        </Typography>
                      }
                    />
                  </ListItem>
                );
              })}
            </List>
          </>
        )}
      </DialogContent>

      <Divider />

      {/* Footer */}
      <DialogActions sx={{ px: 2.5, py: 1.5, gap: 1 }}>
        <Button
          onClick={onClose}
          size="small"
          color="inherit"
          sx={{ fontSize: "12px", textTransform: "none" }}
        >
          Cancelar
        </Button>
        <Button
          onClick={handleDownload}
          disabled={!someSelected}
          variant="contained"
          size="small"
          startIcon={<DownloadIcon />}
          sx={{
            fontSize: "12px",
            borderRadius: "8px",
            textTransform: "none",
            boxShadow: "none",
            "&:hover": { boxShadow: "none" },
          }}
        >
          {someSelected ? `Descargar (${selected.size})` : "Descargar"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
