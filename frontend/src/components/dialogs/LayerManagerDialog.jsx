import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Collapse,
  Divider,
  FormControlLabel,
  IconButton,
  List,
  ListItem,
  Paper,
  Slider,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import FilterListIcon from "@mui/icons-material/FilterList";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import OpacityIcon from "@mui/icons-material/Opacity";
import CloseIcon from "@mui/icons-material/Close";
import {
  FIELD_LIMITS,
  initFilterForField,
  convertToBackend,
} from "../../utils/radarFields";

/**
 * Diálogo para gestionar el orden de las capas visibles en el mapa.
 * - Permite reordenar capas con drag & drop
 * - La primera capa es la capa activa para herramientas
 * - Sincroniza con ProductSelectorDialog y ActiveLayerPicker
 */
export default function LayerManagerDialog({
  open,
  onClose,
  layers = [], // Array de LayerResult del frame actual
  onReorder, // (newOrder) => void - callback para notificar nuevo orden
  onToggleLayerVisibility, // (field, source_file) => void - toggle visibilidad
  hiddenLayers = new Set(), // Set de "field::source_file" keys ocultas
  opacityByLayer = {}, // { "FIELD::source_file": number } opacidades por capa
  onLayerOpacityChange, // (field, source_file, opacity) => void
  filtersPerField = {}, // { FIELD: [{field, min, max}] } filtros iniciales por campo
  onApplyFilters, // (filtersPerField) => void - callback al aplicar filtros
}) {
  const [orderedLayers, setOrderedLayers] = useState([]);
  const lastUpdateRef = React.useRef(0);

  // Estado para filtros por campo (internos al diálogo)
  const [localFilters, setLocalFilters] = useState({});   // { FIELD: { rhohv: {enabled, min}, range: {enabled, min, max} } }
  const [openFilterFor, setOpenFilterFor] = useState(new Set()); // campos con panel de filtros abierto

  // Inicializar localFilters desde prop filtersPerField cuando cambia externamente
  // (al aplicar filtros o al resetear desde ProductSelector).
  // No depende de 'open' para que los valores escritos por el usuario persistan
  // al cerrar y reabrir el diálogo sin haber aplicado.
  useEffect(() => {
    const init = {};
    for (const [field, ruleArr] of Object.entries(filtersPerField || {})) {
      const key = String(field).toUpperCase();
      const lim = FIELD_LIMITS[key] || { min: 0, max: 1 };
      init[key] = initFilterForField(key);
      for (const r of ruleArr || []) {
        const rField = String(r.field || "").toUpperCase();
        if (rField === "RHOHV" && key !== "RHOHV") {
          init[key].rhohv = { enabled: true, min: r.min ?? 0.8 };
        } else if (rField === key) {
          init[key].range = { enabled: true, min: r.min ?? lim.min, max: r.max ?? lim.max };
        }
      }
    }
    setLocalFilters(init);
  }, [filtersPerField]);

  const toggleFilterPanel = useCallback((field) => {
    const key = String(field).toUpperCase();
    setOpenFilterFor((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
    // Ensure a default entry exists for this field
    setLocalFilters((prev) => {
      if (prev[key]) return prev;
      return { ...prev, [key]: initFilterForField(key) };
    });
  }, []);

  const updateFilter = useCallback((field, section, patch) => {
    const key = String(field).toUpperCase();
    setLocalFilters((prev) => ({
      ...prev,
      [key]: {
        ...(prev[key] || initFilterForField(key)),
        [section]: { ...(prev[key]?.[section] || {}), ...patch },
      },
    }));
  }, []);

  const handleApplyFiltersClick = useCallback(() => {
    onApplyFilters?.(convertToBackend(localFilters));
  }, [localFilters, onApplyFilters]);

  const hasAnyFilter = useMemo(() =>
    Object.values(localFilters).some(
      (f) => f?.rhohv?.enabled || f?.range?.enabled
    ), [localFilters]);

  // Estado para drag del panel completo
  const [position, setPosition] = useState({ x: 68, y: 70 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Handlers para drag del panel completo
  const handlePanelMouseDown = useCallback((e) => {
    // Solo permitir drag desde el header, no desde los elementos interactivos
    if (e.target.closest('button') || e.target.closest('.MuiSlider-root')) {
      return;
    }
    setIsDragging(true);
    setDragOffset({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    });
  }, [position]);

  const handlePanelMouseMove = useCallback((e) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragOffset.x,
      y: e.clientY - dragOffset.y,
    });
  }, [isDragging, dragOffset]);

  const handlePanelMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Agregar/remover listeners globales para drag
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handlePanelMouseMove);
      document.addEventListener('mouseup', handlePanelMouseUp);
      return () => {
        document.removeEventListener('mousemove', handlePanelMouseMove);
        document.removeEventListener('mouseup', handlePanelMouseUp);
      };
    }
  }, [isDragging, handlePanelMouseMove, handlePanelMouseUp]);

  // Sincronizar con las capas externas cuando cambien
  useEffect(() => {
    if (!open) return;

    // Si no hay capas, limpiar la lista local
    if (layers.length === 0) {
      setOrderedLayers([]);
      return;
    }

    // Solo actualizar si ha pasado suficiente tiempo desde la última actualización local
    const now = Date.now();
    if (now - lastUpdateRef.current < 100) {
      return; // Ignorar actualizaciones inmediatas después de drag
    }

    // Ordenar por el campo 'order' existente
    const sorted = [...layers].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    setOrderedLayers(sorted);
  }, [open, layers]);

  // --- Drag & Drop ---
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
      if (Number.isNaN(fromIdx) || fromIdx === toIdx) {
        return;
      }

      const next = [...orderedLayers];
      const [moved] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, moved);

      setOrderedLayers(next);

      // Marcar timestamp para prevenir re-sincronización inmediata
      lastUpdateRef.current = Date.now();

      // Notificar cambio al padre
      onReorder?.(next);
    },
    [orderedLayers, onReorder],
  );

  // Manejar cuando se cancela el drag (soltar fuera del área)
  const onDragEnd = useCallback(() => {
    // No hacer nada especial
  }, []);

  const buildLayerLabel = (layer) => {
    const field = layer?.field || "Unknown";
    const radar = layer?.radar || "";

    // Si hay múltiples radares, mostrar radar y campo
    const uniqueRadars = [
      ...new Set(orderedLayers.map((l) => l.radar).filter(Boolean)),
    ];
    if (uniqueRadars.length > 1) {
      return `${radar} - ${field}`;
    }

    // Si es un solo radar, solo mostrar el campo
    return field;
  };

  const handleToggle = useCallback(
    (layer) => {
      onToggleLayerVisibility?.(layer.field, layer.source_file);
    },
    [onToggleLayerVisibility],
  );

  /** Clave compuesta para identificar una capa única (field + archivo fuente) */
  const getLayerKey = (layer) =>
    `${String(layer.field || "").toUpperCase()}::${layer.source_file || ""}`;

  /** Opacidad actual de una capa (default 1) */
  const getLayerOpacity = (layer) => {
    const key = getLayerKey(layer);
    return typeof opacityByLayer[key] === "number" ? opacityByLayer[key] : 1;
  };

  const handleOpacityChange = useCallback(
    (layer, value) => {
      onLayerOpacityChange?.(layer.field, layer.source_file, value);
    },
    [onLayerOpacityChange],
  );

  return (
    <Collapse
      in={open}
      orientation="horizontal"
      timeout={200}
      easing={{
        enter: "cubic-bezier(0.4, 0, 0.2, 1)",
        exit: "cubic-bezier(0.4, 0, 0.6, 1)",
      }}
    >
      <Paper
        elevation={3}
        sx={{
          position: "absolute",
          top: position.y,
          left: position.x,
          zIndex: 999,
          width: 370,
          maxHeight: "calc(100vh - 100px)",
          overflowY: "auto",
          backgroundColor: "rgba(255, 255, 255, 0.98)",
          backdropFilter: "blur(8px)",
          borderRadius: "8px",
          boxShadow: isDragging
            ? "0 8px 24px rgba(0,0,0,0.3)"
            : "0 4px 12px rgba(0,0,0,0.2)",
          transition: isDragging
            ? "box-shadow 0.2s"
            : "opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s",
          userSelect: isDragging ? "none" : "auto",
        }}
      >
        <Box
          onMouseDown={handlePanelMouseDown}
          sx={{
            padding: "12px 16px",
            borderBottom: "1px solid rgba(0, 0, 0, 0.08)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            cursor: isDragging ? "grabbing" : "grab",
            userSelect: "none",
          }}
        >
          <Typography
            variant="subtitle1"
            sx={{
              fontWeight: 600,
              fontSize: "14px",
              color: "#212121",
            }}
          >
            Orden de Capas
          </Typography>
          <IconButton size="small" onClick={onClose} sx={{ padding: "4px" }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        <Box sx={{ padding: "12px 16px" }}>
          {orderedLayers.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No hay capas visibles
            </Typography>
          ) : (
            <>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: "block", mb: 2 }}
              >
                La primera capa es la capa activa para herramientas. Arrastra
                para reordenar.
              </Typography>

              <List disablePadding>
                {orderedLayers.map((layer, idx) => (
                  <ListItem
                    key={`${layer.field}-${layer.source_file}-${idx}`}
                    draggable
                    onDragStart={(e) => onDragStart(e, idx)}
                    onDragOver={onDragOver}
                    onDrop={(e) => onDrop(e, idx)}
                    onDragEnd={onDragEnd}
                    sx={{
                      bgcolor:
                        idx === 0 ? "action.selected" : "background.paper",
                      border: "1px solid",
                      borderColor: idx === 0 ? "primary.main" : "divider",
                      borderRadius: 1,
                      mb: 1,
                      cursor: "grab",
                      padding: "8px",
                      opacity: hiddenLayers.has(
                        `${layer.field}::${layer.source_file}`,
                      )
                        ? 0.5
                        : 1,
                      "&:active": {
                        cursor: "grabbing",
                      },
                      transition: "all 0.2s",
                    }}
                  >
                    <Box
                      display="flex"
                      alignItems="center"
                      width="100%"
                      gap={1}
                    >
                      <Tooltip title="Arrastrar para reordenar">
                        <IconButton size="small" sx={{ cursor: "grab" }}>
                          <DragIndicatorIcon />
                        </IconButton>
                      </Tooltip>

                      <Box flex={1}>
                        <Typography
                          variant="body2"
                          fontWeight={idx === 0 ? "bold" : "normal"}
                        >
                          {buildLayerLabel(layer)}
                        </Typography>
                        {idx === 0 && (
                          <Typography variant="caption" color="primary">
                            Capa activa
                          </Typography>
                        )}
                      </Box>

                      <Typography variant="caption" color="text.secondary">
                        #{idx + 1}
                      </Typography>

                      <Tooltip title="Filtros de capa">
                        <IconButton
                          size="small"
                          onClick={() => toggleFilterPanel(layer.field)}
                          sx={{
                            padding: "4px",
                            color:
                              localFilters[String(layer.field).toUpperCase()]?.rhohv?.enabled ||
                                localFilters[String(layer.field).toUpperCase()]?.range?.enabled
                                ? "primary.main"
                                : "text.secondary",
                          }}
                        >
                          <FilterListIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>

                    {/* Slider de opacidad por capa */}
                    <Box
                      display="flex"
                      alignItems="center"
                      width="100%"
                      gap={1}
                      sx={{ pl: 1.5, pr: 1, mt: 0.5 }}
                    >
                      <Tooltip title="Opacidad">
                        <OpacityIcon
                          fontSize="small"
                          sx={{ color: "text.secondary", fontSize: 16 }}
                        />
                      </Tooltip>
                      <Slider
                        size="small"
                        min={0}
                        max={1}
                        step={0.05}
                        value={getLayerOpacity(layer)}
                        onChange={(_e, val) => handleOpacityChange(layer, val)}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(v) => `${Math.round(v * 100)}%`}
                        sx={{
                          flex: 1,
                          "& .MuiSlider-thumb": { width: 14, height: 14 },
                          "& .MuiSlider-track": { height: 3 },
                          "& .MuiSlider-rail": { height: 3 },
                        }}
                      />
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ minWidth: 32, textAlign: "right" }}
                      >
                        {Math.round(getLayerOpacity(layer) * 100)}%
                      </Typography>

                      <Tooltip
                        title={
                          hiddenLayers.has(
                            `${layer.field}::${layer.source_file}`,
                          )
                            ? "Mostrar capa"
                            : "Ocultar capa"
                        }
                      >
                        <IconButton
                          size="small"
                          onClick={() => handleToggle(layer)}
                          sx={{
                            padding: "4px",
                            color: hiddenLayers.has(
                              `${layer.field}::${layer.source_file}`,
                            )
                              ? "text.disabled"
                              : "text.secondary",
                            "&:hover": {
                              color: hiddenLayers.has(
                                `${layer.field}::${layer.source_file}`,
                              )
                                ? "success.main"
                                : "warning.main",
                              backgroundColor: hiddenLayers.has(
                                `${layer.field}::${layer.source_file}`,
                              )
                                ? "rgba(76, 175, 80, 0.08)"
                                : "rgba(255, 152, 0, 0.08)",
                            },
                          }}
                        >
                          {hiddenLayers.has(
                            `${layer.field}::${layer.source_file}`,
                          ) ? (
                            <VisibilityOffIcon fontSize="small" />
                          ) : (
                            <VisibilityIcon fontSize="small" />
                          )}
                        </IconButton>
                      </Tooltip>
                    </Box>

                    {/* Panel de filtros por capa */}
                    <Collapse
                      in={openFilterFor.has(String(layer.field).toUpperCase())}
                      timeout="auto"
                      unmountOnExit
                    >
                      <Box
                        sx={{
                          pl: 1.5,
                          pr: 1,
                          pt: 0.5,
                          pb: 0.5,
                          borderTop: "1px solid rgba(0,0,0,0.06)",
                          mt: 0.5,
                        }}
                      >
                        {/* RHOHV filter — only for non-RHOHV fields */}
                        {String(layer.field).toUpperCase() !== "RHOHV" && (
                          <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                            <Checkbox
                              size="small"
                              checked={
                                !!(localFilters[String(layer.field).toUpperCase()]?.rhohv?.enabled)
                              }
                              onChange={(e) =>
                                updateFilter(layer.field, "rhohv", { enabled: e.target.checked })
                              }
                              sx={{ p: 0 }}
                            />
                            <Typography variant="caption" sx={{ minWidth: 60 }}>
                              RHOHV ≥
                            </Typography>
                            <TextField
                              size="small"
                              type="number"
                              value={localFilters[String(layer.field).toUpperCase()]?.rhohv?.min ?? 0.8}
                              onChange={(e) =>
                                updateFilter(layer.field, "rhohv", { min: e.target.value })
                              }
                              onKeyDown={(e) => e.key === "Enter" && handleApplyFiltersClick()}
                              disabled={
                                !localFilters[String(layer.field).toUpperCase()]?.rhohv?.enabled
                              }
                              inputProps={{ step: 0.01, min: 0, max: 1 }}
                              sx={{ width: 72 }}
                            />
                          </Box>
                        )}

                        {/* Self-range filter */}
                        <Box display="flex" alignItems="center" gap={1}>
                          <Checkbox
                            size="small"
                            checked={
                              !!(localFilters[String(layer.field).toUpperCase()]?.range?.enabled)
                            }
                            onChange={(e) =>
                              updateFilter(layer.field, "range", { enabled: e.target.checked })
                            }
                            sx={{ p: 0 }}
                          />
                          <Typography variant="caption" sx={{ minWidth: 60 }}>
                            {String(layer.field).toUpperCase()}
                          </Typography>
                          <TextField
                            size="small"
                            type="number"
                            label="min"
                            value={localFilters[String(layer.field).toUpperCase()]?.range?.min ?? (FIELD_LIMITS[String(layer.field).toUpperCase()]?.min ?? 0)}
                            onChange={(e) =>
                              updateFilter(layer.field, "range", { min: e.target.value })
                            }
                            onKeyDown={(e) => e.key === "Enter" && handleApplyFiltersClick()}
                            disabled={
                              !localFilters[String(layer.field).toUpperCase()]?.range?.enabled
                            }
                            inputProps={{ step: 0.1 }}
                            sx={{ width: 76 }}
                          />
                          <TextField
                            size="small"
                            type="number"
                            label="max"
                            value={localFilters[String(layer.field).toUpperCase()]?.range?.max ?? (FIELD_LIMITS[String(layer.field).toUpperCase()]?.max ?? 1)}
                            onChange={(e) =>
                              updateFilter(layer.field, "range", { max: e.target.value })
                            }
                            onKeyDown={(e) => e.key === "Enter" && handleApplyFiltersClick()}
                            disabled={
                              !localFilters[String(layer.field).toUpperCase()]?.range?.enabled
                            }
                            inputProps={{ step: 0.1 }}
                            sx={{ width: 76 }}
                          />
                        </Box>
                      </Box>
                    </Collapse>
                  </ListItem>
                ))}
              </List>

              {onApplyFilters && (
                <Box mt={2}>
                  <Divider sx={{ mb: 1.5 }} />
                  <Button
                    fullWidth
                    variant={hasAnyFilter ? "contained" : "outlined"}
                    size="small"
                    onClick={handleApplyFiltersClick}
                    color="primary"
                  >
                    Aplicar filtros
                  </Button>
                </Box>
              )}
            </>
          )}
        </Box>
      </Paper>
    </Collapse>
  );
}
