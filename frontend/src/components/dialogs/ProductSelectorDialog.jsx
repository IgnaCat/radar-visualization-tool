import { useEffect, useMemo, useState, useRef } from "react";
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
  IconButton,
  Collapse,
  Chip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import LayerControlList from "../controls/LayerControlList";
import { formatSourceDisplay } from "../../utils/fieldAnalysis";

const MARKS_01 = [
  { value: 0, label: "0" },
  { value: 0.25, label: "0.25" },
  { value: 0.5, label: "0.5" },
  { value: 0.75, label: "0.75" },
  { value: 1, label: "1" },
];

const FIELD_LIMITS = {
  DBZH: { min: -30, max: 70 },
  DBZV: { min: -30, max: 70 },
  DBZHF: { min: -30, max: 70 },
  ZDR: { min: -5, max: 10.5 },
  RHOHV: { min: 0.3, max: 1.0 },
  KDP: { min: 0, max: 8 },
  VRAD: { min: -35, max: 35 },
  WRAD: { min: 0, max: 10 },
  PHIDP: { min: -180, max: 180 },
};

// Si llega un alias raro del archivo, lo “canonizamos”
const CANON = {
  dbzh: "DBZH",
  zdr: "ZDR",
  rhohv: "RHOHV",
  kdp: "KDP",
  dbzv: "DBZV",
  vrad: "VRAD",
  wrad: "WRAD",
  phidp: "PHIDP",
};

function canonize(name = "") {
  const k = String(name).toLowerCase();
  return CANON[k] || name.toUpperCase();
}

// Crear capas a partir del análisis de campos
// Es decir, crea una lista de capas con metadata con un orden logico
function deriveLayersFromFieldAnalysis(fieldAnalysis) {
  if (!fieldAnalysis || !fieldAnalysis.allFields) {
    return [];
  }

  const { commonFields, specificFields, allFields, sameRadarStrategy } =
    fieldAnalysis;

  // Orden sugerido: DBZH primero si existe
  const order = [
    "DBZH",
    "DBZV",
    "KDP",
    "RHOHV",
    "ZDR",
    "VRAD",
    "WRAD",
    "PHIDP",
  ];

  const allLayers = [];

  // Si no hay campos comunes ni específicos (una sola fuente),
  // usar allFields directamente sin chips
  if (commonFields.length === 0 && specificFields.length === 0) {
    const sortedFields = [...allFields].sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    });

    sortedFields.forEach((f, i) => {
      allLayers.push({
        id: f.toLowerCase(),
        label: f,
        field: f,
        enabled: i === 0, // Solo el primer campo habilitado por defecto
        opacity: 1,
        isCommon: false, // No mostrar chip de común
        sources: [],
      });
    });

    return allLayers;
  }

  // Primero agregar campos comunes
  const sortedCommon = [...commonFields].sort((a, b) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });

  sortedCommon.forEach((f, i) => {
    allLayers.push({
      id: f.toLowerCase(),
      label: f,
      field: f,
      enabled: i === 0, // Solo el primer campo habilitado por defecto
      opacity: 1,
      isCommon: true,
      sources: [],
      simplified: sameRadarStrategy, // Pasar info para simplificar display
    });
  });

  // Luego agregar campos específicos
  specificFields.forEach((sourceInfo) => {
    const { source, fields } = sourceInfo;
    const sortedFields = [...fields]
      .filter((f) => !commonFields.includes(f)) // Excluir campos ya agregados como comunes
      .sort((a, b) => {
        const ia = order.indexOf(a);
        const ib = order.indexOf(b);
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      });

    sortedFields.forEach((f) => {
      // Verificar si ya existe este campo en la lista
      const existing = allLayers.find((l) => l.field === f);
      if (existing) {
        // Agregar esta fuente al campo existente
        existing.sources.push(source);
        existing.isCommon = false;
        existing.simplified = sameRadarStrategy;
      } else {
        // Crear nueva entrada
        allLayers.push({
          id: f.toLowerCase(),
          label: f,
          field: f,
          enabled: false,
          opacity: 1,
          isCommon: false,
          sources: [source],
          simplified: sameRadarStrategy,
        });
      }
    });
  });

  // Asegurar que al menos el primer campo esté habilitado por defecto
  if (allLayers.length > 0 && !allLayers.some((l) => l.enabled)) {
    allLayers[0].enabled = true;
  }

  return allLayers;
}

// Función legacy para compatibilidad con fields_present
function deriveLayersFromFields(fields_present) {
  const uniq = Array.from(new Set((fields_present || []).map(canonize)));
  // Orden sugerido: DBZH primero si existe
  const order = [
    "DBZH",
    "DBZV",
    "KDP",
    "RHOHV",
    "ZDR",
    "VRAD",
    "WRAD",
    "PHIDP",
  ];
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
    isCommon: true,
    sources: [],
  }));
}

export default function ProductSelectorDialog({
  open,
  fieldAnalysis = null, // Información sobre campos comunes y específicos
  fields_present = [], // Fallback para compatibilidad legacy (vacío por defecto)
  elevations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  volumes = [],
  radars = [],
  onClose,
  onConfirm,
  initialProduct = "ppi",
  initialCappiHeight = 2000,
  initialElevation = 0,
  initialLayers = [],
  initialFilters = {
    rhohv: { enabled: false, min: 0.92, max: 1.0 },
    other: { enabled: false, min: 0, max: 1.0 },
  },
}) {
  const MAX_RADARS = 3;

  // Siempre recalcular desde fieldAnalysis si está disponible
  // Solo usar initialLayers para preservar el estado enabled/opacity del usuario
  const derivedLayers = useMemo(() => {
    // Primero calcular las capas desde fieldAnalysis o fields_present
    let baseLayers = [];

    if (
      fieldAnalysis &&
      fieldAnalysis.allFields &&
      fieldAnalysis.allFields.length > 0
    ) {
      baseLayers = deriveLayersFromFieldAnalysis(fieldAnalysis);
    } else if (fields_present && fields_present.length > 0) {
      baseLayers = deriveLayersFromFields(fields_present);
    }

    // Si no hay datos base, retornar vacío
    if (baseLayers.length === 0) {
      return [];
    }

    // Si hay initialLayers, preservar solo enabled/opacity para campos que existen
    if (initialLayers && initialLayers.length > 0) {
      return baseLayers.map((bl, idx) => {
        const existing = initialLayers.find((il) => il.field === bl.field);
        if (existing) {
          return {
            ...bl, // Usar toda la metadata actualizada (isCommon, sources, etc.)
            enabled: existing.enabled,
            opacity: existing.opacity,
          };
        }
        // Campo nuevo: habilitarlo solo si es el primero y no hay ninguno habilitado
        const anyEnabled = initialLayers.some((il) =>
          baseLayers.some((bl2) => bl2.field === il.field && il.enabled),
        );
        return {
          ...bl,
          enabled: !anyEnabled && idx === 0,
        };
      });
    }

    return baseLayers;
  }, [fieldAnalysis, fields_present, initialLayers]);

  // Asegurar que derivedLayers tenga al menos un campo habilitado
  const derivedLayersWithDefault = useMemo(() => {
    if (derivedLayers.length === 0) return [];
    if (derivedLayers.some((l) => l.enabled)) return derivedLayers;

    // Si ninguno está habilitado, habilitar el primero
    return derivedLayers.map((l, i) => ({
      ...l,
      enabled: i === 0,
    }));
  }, [derivedLayers]);

  const [layers, setLayers] = useState(derivedLayersWithDefault);
  const [product, setProduct] = useState(initialProduct);
  const [height, setHeight] = useState(initialCappiHeight);
  const [selectedVolumes, setSelectedVolumes] = useState(volumes);
  const [selectedRadars, setSelectedRadars] = useState(radars);

  // Elevación: trabajemos con índices de elevación
  const initialElevationIndex = useMemo(() => {
    const N = Array.isArray(elevations) ? elevations.length : 0;
    const idx = Number.isInteger(initialElevation) ? initialElevation : 0;
    return Math.max(0, Math.min(Math.max(N - 1, 0), idx));
  }, [elevations, initialElevation]);

  const [elevationIdx, setElevationIdx] = useState(initialElevationIndex);
  const [filters, setFilters] = useState(structuredClone(initialFilters));
  const [showFilters, setShowFilters] = useState(false);

  // Ref para evitar loops infinitos
  const lastDerivedFieldsRef = useRef("");

  // Actualizar capas preservando el orden del usuario cuando cambian los campos disponibles
  // (acordarse q el usuario puede haber reordenado o habilitado/deshabilitado capas)
  useEffect(() => {
    // Crear una firma única de derivedLayersWithDefault basada en los campos
    const derivedSignature = derivedLayersWithDefault
      .map((dl) => `${dl.field}:${dl.isCommon}:${dl.sources?.length || 0}`)
      .join("|");

    // Solo procesar si la firma cambió
    if (derivedSignature === lastDerivedFieldsRef.current) {
      return;
    }

    lastDerivedFieldsRef.current = derivedSignature;

    if (layers.length === 0 || !layers.some((l) => l.enabled)) {
      setLayers(derivedLayersWithDefault);
      return;
    }

    // Verificar si el conjunto de campos cambió significativamente
    const currentFields = new Set(layers.map((l) => l.field));
    const derivedFields = new Set(derivedLayersWithDefault.map((l) => l.field));

    // Detectar campos nuevos o eliminados
    const hasNewFields = derivedLayersWithDefault.some(
      (dl) => !currentFields.has(dl.field),
    );
    const hasRemovedFields = layers.some((l) => !derivedFields.has(l.field));

    // Si cambió significativamente el conjunto de campos, reemplazar preservando estados enabled
    if (hasNewFields || hasRemovedFields) {
      const updatedLayers = derivedLayersWithDefault.map((dl) => {
        const existing = layers.find((l) => l.field === dl.field);
        if (existing) {
          // Preservar enabled/opacity del usuario, actualizar metadata
          return {
            ...dl,
            enabled: existing.enabled,
            opacity: existing.opacity,
          };
        }
        return dl;
      });

      // Asegurar que al menos un campo esté habilitado
      if (updatedLayers.length > 0 && !updatedLayers.some((l) => l.enabled)) {
        updatedLayers[0].enabled = true;
      }

      setLayers(updatedLayers);
      return;
    }

    // Si solo cambió metadata (isCommon, sources), actualizar
    const metadataChanged = layers.some((l) => {
      const derived = derivedLayersWithDefault.find(
        (dl) => dl.field === l.field,
      );
      if (!derived) return false;

      // Comparar metadata de forma más robusta
      const isCommonChanged = l.isCommon !== derived.isCommon;
      const sourcesChanged =
        (l.sources?.length || 0) !== (derived.sources?.length || 0);

      return isCommonChanged || sourcesChanged;
    });

    if (metadataChanged) {
      setLayers((prev) =>
        prev.map((l) => {
          const derived = derivedLayersWithDefault.find(
            (dl) => dl.field === l.field,
          );
          if (derived) {
            return {
              ...l,
              isCommon: derived.isCommon,
              sources: derived.sources || [],
            };
          }
          return l;
        }),
      );
    }
  }, [derivedLayersWithDefault, layers]);

  useEffect(() => {
    setElevationIdx(initialElevationIndex);
  }, [initialElevationIndex]);

  useEffect(() => {
    setSelectedVolumes(volumes);
  }, [volumes]);

  // Resetear la altura a la inicial cuando se cambia desde/hacia CAPPI, para evitar confusiones
  useEffect(() => {
    if (product !== "cappi") {
      setHeight(initialCappiHeight);
    }
  }, [product, initialCappiHeight]);

  useEffect(() => {
    // Si el radar seleccionado ya no está en la lista, resetear a vacío
    setSelectedRadars(Array.isArray(radars) ? radars.slice(0, MAX_RADARS) : []);
  }, [radars]);

  // Variable activa (usamos para los filtros)
  const activeField = (
    layers.find((l) => l.enabled)?.field ||
    layers.find((l) => l.enabled)?.label ||
    "DBZH"
  ).toUpperCase();
  const limits = FIELD_LIMITS[activeField] || { min: 0, max: 1 };

  const [activeRange, setActiveRange] = useState([limits.min, limits.max]);

  useEffect(() => {
    const lim = FIELD_LIMITS[activeField] || { min: 0, max: 1 };
    // Si el filtro other está habilitado, mantener sus valores, sino resetear a los límites del campo
    if (!filters.other?.enabled) {
      setActiveRange([lim.min, lim.max]);
    }
  }, [activeField, filters.other?.enabled]);

  const isCAPPI = product === "cappi";
  const isPPI = product === "ppi";

  const setRhohv = (patch) =>
    setFilters((f) => ({ ...f, rhohv: { ...f.rhohv, ...patch } }));
  const setOther = (patch) =>
    setFilters((f) => ({ ...f, other: { ...f.other, ...patch } }));

  const clamp01 = (v) => Math.max(0, Math.min(1, Number(v)));

  const handleAccept = () => {
    // Validar que haya al menos un campo habilitado
    const enabledCount = layers.filter((l) => l.enabled).length;
    if (enabledCount === 0 && product !== "colmax") {
      alert("Debe seleccionar al menos un campo para visualizar");
      return;
    }

    const filtersOut = [];

    // Filtro de rango de variable activa (solo si está habilitado)
    if (filters.other?.enabled) {
      const [amin, amax] = activeRange;
      filtersOut.push({
        field: activeField,
        type: "range",
        min: amin,
        max: amax,
        enabled: true,
      });
    }

    // Filtro RHOHV
    if (filters.rhohv?.enabled) {
      let min = clamp01(filters.rhohv.min ?? 0);
      let max = clamp01(filters.rhohv.max ?? 1);
      if (min > max) [min, max] = [max, min];
      filtersOut.push({
        field: "RHOHV",
        type: "range",
        min,
        max,
        enabled: true,
      });
    }

    // Para COLMAX, forzar DBZH como única capa
    const finalLayers =
      product === "colmax"
        ? [
            {
              id: "dbzh",
              label: "DBZH",
              field: "DBZH",
              enabled: true,
              opacity: 1,
            },
          ]
        : layers;

    onConfirm({
      layers: finalLayers,
      product,
      height: isCAPPI ? height : undefined,
      elevation: isPPI ? elevationIdx : undefined,
      filters: filtersOut,
      selectedVolumes,
      selectedRadars,
    });
    onClose();
  };

  const handleClose = () => {
    setLayers(derivedLayersWithDefault);
    setProduct(initialProduct);
    setHeight(initialCappiHeight);
    setElevationIdx(initialElevationIndex);
    setFilters(structuredClone(initialFilters));
    setSelectedVolumes(volumes);
    setSelectedRadars(radars);
    onClose();
  };

  // Marks del slider de elevación
  const elevMarks = useMemo(() => {
    const N = Array.isArray(elevations) ? elevations.length : 0;
    const step = N > 9 ? Math.ceil(N / 9) : 1; // máx 9 marcas visibles
    return Array.from({ length: N }, (_, i) =>
      i % step === 0 ? { value: i, label: String(i) } : null,
    ).filter(Boolean);
  }, [elevations]);

  const maxIdx = Math.max(0, (elevations?.length || 1) - 1);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Opciones de Visualización</DialogTitle>

      <DialogContent dividers>
        {/* Grid layout: Vista a la izquierda, Volúmenes y Radares a la derecha */}
        <Box display="grid" gridTemplateColumns="1fr 1fr" gap={3}>
          {/* Columna izquierda: Seleccionar Vista */}
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Seleccionar Vista
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
                <FormControlLabel
                  value="cappi"
                  control={<Radio />}
                  label="CAPPI"
                />
              </RadioGroup>
            </FormControl>
          </Box>

          {/* Columna derecha: Volúmenes y Radares apilados */}
          <Box display="flex" flexDirection="column" gap={2}>
            {/* Selección de volúmenes */}
            {Array.isArray(volumes) && volumes.length > 0 && (
              <Box>
                <Typography variant="subtitle1" mb={2} gutterBottom>
                  Seleccionar volúmenes
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {volumes.map((vol, idx) => {
                    const isSelected = selectedVolumes.includes(vol);
                    return (
                      <Button
                        key={vol}
                        variant={isSelected ? "contained" : "outlined"}
                        onClick={() => {
                          setSelectedVolumes((prev) =>
                            prev.includes(vol)
                              ? prev.filter((v) => v !== vol)
                              : [...prev, vol],
                          );
                        }}
                        sx={{
                          borderRadius: 999,
                          backgroundColor: isSelected ? "#888" : "#eee",
                          color: isSelected ? "#fff" : "#333",
                          fontWeight: 500,
                          textTransform: "none",
                          boxShadow: isSelected ? 2 : 0,
                          transition: "all 0.2s",
                          "&:hover": {
                            backgroundColor: isSelected ? "#555" : "#ccc",
                            color: isSelected ? "#fff" : "#111",
                          },
                          minWidth: 90,
                          px: 2,
                          py: 1,
                        }}
                      >
                        {`Volumen ${vol}`}
                      </Button>
                    );
                  })}
                </Box>
              </Box>
            )}

            {/* Selección de radares */}
            {Array.isArray(radars) && radars.length > 0 && (
              <Box>
                <Typography variant="subtitle1" mb={2} gutterBottom>
                  Seleccionar radares
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {radars.map((site) => {
                    const isSelected = selectedRadars.includes(site);
                    const atMax =
                      !isSelected && selectedRadars.length >= MAX_RADARS;
                    return (
                      <Button
                        key={site}
                        variant={isSelected ? "contained" : "outlined"}
                        disabled={atMax}
                        onClick={() => {
                          setSelectedRadars((prev) => {
                            const already = prev.includes(site);
                            if (already) return prev.filter((s) => s !== site);
                            if (prev.length >= MAX_RADARS) return prev; // ignore if at limit
                            return [...prev, site];
                          });
                        }}
                        sx={{
                          borderRadius: 999,
                          backgroundColor: isSelected ? "#888" : "#eee",
                          color: isSelected ? "#fff" : "#333",
                          fontWeight: 500,
                          textTransform: "none",
                          boxShadow: isSelected ? 2 : 0,
                          transition: "all 0.2s",
                          "&:hover": {
                            backgroundColor: isSelected ? "#555" : "#ccc",
                            color: isSelected ? "#fff" : "#111",
                          },
                          minWidth: 90,
                          px: 2,
                          py: 1,
                        }}
                      >
                        {String(site)}
                      </Button>
                    );
                  })}
                </Box>
                <Typography
                  variant="caption"
                  sx={{ opacity: 0.7, display: "block", mt: 0.5 }}
                >
                  Máximo {MAX_RADARS} radares a la vez.
                </Typography>
              </Box>
            )}
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Variables reales del archivo - Ocultar para COLMAX */}
        {product !== "colmax" && (
          <Box mt={2}>
            {/* Información sobre campos comunes y específicos */}
            <LayerControlList items={layers} onChange={setLayers} />
          </Box>
        )}

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
                valueLabelFormat={(i) => i}
              />
              <Typography variant="caption" sx={{ opacity: 0.7 }}>
                Índice de elevación seleccionado: {elevationIdx}
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
        <Box mt={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <IconButton
              size="small"
              onClick={() => setShowFilters((v) => !v)}
              aria-label={showFilters ? "Ocultar filtros" : "Mostrar filtros"}
            >
              {showFilters ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
            <Typography variant="subtitle1" sx={{ userSelect: "none" }}>
              Filtros
            </Typography>
          </Box>
          <Collapse in={showFilters} timeout="auto" unmountOnExit>
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

            {/* Filtros de variable seleccionada */}
            <Box mt={2} mb={2}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={!!filters.other?.enabled}
                    onChange={(e) => setOther({ enabled: e.target.checked })}
                  />
                }
                label={`Rango de ${activeField}`}
              />
              <Box display="flex" alignItems="center" gap={1} pl={5}>
                <Slider
                  value={activeRange}
                  onChange={(_, v) => setActiveRange(v)}
                  step={0.1}
                  min={limits.min}
                  max={limits.max}
                  valueLabelDisplay="auto"
                  disabled={!filters.other?.enabled}
                  sx={{ flex: 1, minWidth: 180 }}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Min"
                  value={activeRange[0]}
                  onChange={(e) =>
                    setActiveRange(([_, b]) => [Number(e.target.value), b])
                  }
                  disabled={!filters.other?.enabled}
                  sx={{ width: 80 }}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Max"
                  value={activeRange[1]}
                  onChange={(e) =>
                    setActiveRange(([a, _]) => [a, Number(e.target.value)])
                  }
                  disabled={!filters.other?.enabled}
                  sx={{ width: 80 }}
                />
              </Box>
            </Box>
          </Collapse>
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
