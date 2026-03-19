import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Typography,
  Slider,
  TextField,
  Checkbox,
  FormControlLabel,
  Divider,
} from "@mui/material";
import { FIELD_LIMITS, MARKS_01 } from "../../utils/radarFields";

function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

export default function RadarFilterControls({
  selectedField = "DBZH",
  onFiltersChange,
  initialFilters = {
    rhohv: { enabled: false, min: 0.92, max: 1.0 },
    other: { enabled: true, min: 0, max: 1.0 },
  },
}) {
  const activeField = String(selectedField || "DBZH").toUpperCase();
  const limits = FIELD_LIMITS[activeField] || { min: 0, max: 1 };

  const [activeRange, setActiveRange] = useState([limits.min, limits.max]);
  const [rhohv, setRhohv] = useState({
    enabled: !!initialFilters?.rhohv?.enabled,
    min: Number(initialFilters?.rhohv?.min ?? 0.92),
    max: Number(initialFilters?.rhohv?.max ?? 1.0),
  });

  // Resetear rangos cuando cambia la variable activa
  useEffect(() => {
    const l = FIELD_LIMITS[activeField] || { min: 0, max: 1 };
    setActiveRange([l.min, l.max]);
  }, [activeField]);

  // Emitir filtros hacia arriba cada vez que cambie algo
  useEffect(() => {
    const [amin, amax] = activeRange;

    const out = [
      {
        field: activeField,
        type: "range",
        min: Number(amin),
        max: Number(amax),
        enabled: true,
      },
    ];

    // RHOHV: disponible siempre. Si el activo NO es RHOHV, suele ser tu QC obligado;
    // si ES RHOHV, solo se aplica si lo activás explícitamente.
    if (rhohv.enabled) {
      let rmin = clamp01(rhohv.min);
      let rmax = clamp01(rhohv.max);
      if (rmin > rmax) [rmin, rmax] = [rmax, rmin];
      out.push({
        field: "RHOHV",
        type: "range",
        min: rmin,
        max: rmax,
        enabled: true,
      });
    }

    onFiltersChange?.(out);
  }, [
    activeField,
    activeRange,
    rhohv.enabled,
    rhohv.min,
    rhohv.max,
    onFiltersChange,
  ]);

  return (
    <Box
      mt={1}
      sx={{
        "& .MuiFormControlLabel-root": { m: 0 },
        "& .MuiSlider-root": { height: 4 }, // slider más fino
        "& .MuiTextField-root": { width: 74, m: 0 },
        "& .MuiTypography-subtitle1": { fontSize: "0.86rem" },
      }}
    >
      <Divider sx={{ my: 1.25 }} />

      <Typography variant="subtitle1" gutterBottom>
        Filtros
      </Typography>

      {/* ---- RHOHV ---- */}
      {selectedField !== "RHOHV" && (
        <Box mt={0.5} px={1}>
          <Box
            display="flex"
            alignItems="center"
            gap={1}
            sx={{ flexWrap: "wrap" }}
          >
            <FormControlLabel
              control={
                <Checkbox
                  size="small"
                  checked={!!rhohv.enabled}
                  onChange={(e) =>
                    setRhohv((prev) => ({ ...prev, enabled: e.target.checked }))
                  }
                />
              }
              label="RHOHV"
              sx={{ mr: 1 }}
            />
            <Slider
              value={[Number(rhohv.min ?? 0.92), Number(rhohv.max ?? 1.0)]}
              onChange={(_, v) => {
                const [min, max] = v;
                setRhohv((prev) => ({ ...prev, min, max }));
              }}
              step={0.01}
              min={0}
              max={1}
              marks={MARKS_01}
              valueLabelDisplay="auto"
              disabled={!rhohv.enabled}
              sx={{ flex: 1, minWidth: 170, mx: 1 }}
            />
            <TextField
              type="number"
              size="small"
              label="Min"
              value={Number(rhohv.min ?? 0.92)}
              onChange={(e) =>
                setRhohv((prev) => ({ ...prev, min: clamp01(e.target.value) }))
              }
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              disabled={!rhohv.enabled}
            />
            <TextField
              type="number"
              size="small"
              label="Max"
              value={Number(rhohv.max ?? 1)}
              onChange={(e) =>
                setRhohv((prev) => ({ ...prev, max: clamp01(e.target.value) }))
              }
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              disabled={!rhohv.enabled}
            />
          </Box>
        </Box>
      )}

      {/* ---- Rango de variable activa ---- */}
      <Box
        mt={3}
        mb={3}
        display="flex"
        alignItems="center"
        gap={1.25}
        pl={1.5}
        sx={{ flexWrap: "wrap" }}
      >
        <Typography variant="subtitle1">Rango de {activeField}</Typography>
        <Box px={0.5} display="flex" alignItems="center" gap={1.25}>
          <Slider
            value={activeRange}
            onChange={(_, v) => setActiveRange(v)}
            step={0.1}
            min={limits.min}
            max={limits.max}
            valueLabelDisplay="auto"
            sx={{ flex: 1, minWidth: 170, mr: 1, ml: 1 }}
          />
          <TextField
            size="small"
            type="number"
            label="Min"
            value={activeRange[0]}
            onChange={(e) =>
              setActiveRange(([_, b]) => [Number(e.target.value), b])
            }
          />
          <TextField
            size="small"
            type="number"
            label="Max"
            value={activeRange[1]}
            onChange={(e) =>
              setActiveRange(([a, _]) => [a, Number(e.target.value)])
            }
          />
        </Box>
      </Box>
    </Box>
  );
}
