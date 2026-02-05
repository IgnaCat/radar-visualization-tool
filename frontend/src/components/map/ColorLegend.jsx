import React, { useState, useEffect } from "react";
import { Typography } from "@mui/material";
import { getColormapColors } from "../../api/backend";

// --- Función para decidir color de texto ---
function getTextColor(hex) {
  if (!hex || typeof hex !== "string") return "#000";
  const clean = hex.replace("#", "");
  const bigint = parseInt(
    clean.length === 3
      ? clean
          .split("")
          .map((c) => c + c)
          .join("")
      : clean,
    16,
  );
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  // luminancia perceptual
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.6 ? "#000" : "#FFF";
}

// --- Configuración de paletas HARDCODED (para colormaps default) ---
const LEGENDS = {
  DBZH: {
    steps: [
      { value: 70, color: "#FF29E3", label: "Lluvia muy intensa / granizo" },
      { value: 60, color: "#FF2A98", label: "Lluvia muy intensa / granizo" },
      { value: 50, color: "#FF2A0C", label: "Lluvia intensa" },
      { value: 40, color: "#f7a600", label: "Lluvia intensa" },
      { value: 30, color: "#EAF328", label: "Lluvia moderada" },
      { value: 20, color: "#00AD5A", label: "Lluvia leve" },
      { value: 10, color: "#00E68A", label: "Llovizna" },
      { value: 0, color: "#95b4dc", label: "Neblina" },
      { value: -10, color: "#f0f6f2", label: "Nubes no precipitantes" },
    ],
  },
  DBZHF: {
    steps: [
      { value: 70, color: "#FF29E3", label: "Lluvia muy intensa / granizo" },
      { value: 60, color: "#FF2A98", label: "Lluvia muy intensa / granizo" },
      { value: 50, color: "#FF2A0C", label: "Lluvia intensa" },
      { value: 40, color: "#f7a600", label: "Lluvia intensa" },
      { value: 30, color: "#EAF328", label: "Lluvia moderada" },
      { value: 20, color: "#00AD5A", label: "Lluvia leve" },
      { value: 10, color: "#00E68A", label: "Llovizna" },
      { value: 0, color: "#95b4dc", label: "Neblina" },
      { value: -10, color: "#f0f6f2", label: "Nubes no precipitantes" },
    ],
  },
  DBZV: {
    steps: [
      { value: 70, color: "#FF29E3", label: "Lluvia muy intensa / granizo" },
      { value: 60, color: "#FF2A98", label: "Lluvia muy intensa / granizo" },
      { value: 50, color: "#FF2A0C", label: "Lluvia intensa" },
      { value: 40, color: "#f7a600", label: "Lluvia intensa" },
      { value: 30, color: "#EAF328", label: "Lluvia moderada" },
      { value: 20, color: "#00AD5A", label: "Lluvia leve" },
      { value: 10, color: "#00E68A", label: "Llovizna" },
      { value: 0, color: "#95b4dc", label: "Neblina" },
      { value: -10, color: "#f0f6f2", label: "Nubes no precipitantes" },
    ],
  },
  ZDR: {
    steps: [
      { value: 7, color: "#ff0064", label: "Eco biológico / Ruido" },
      {
        value: 5.5,
        color: "#F9EA3C",
        label: "LLuvia muy intensa / Eco biológico",
      },
      { value: 4, color: "#489D39", label: "Lluvia muy intensa" },
      { value: 3, color: "#00FFFF", label: "Lluvia intensa" },
      { value: 2.5, color: "#66b3df", label: "Lluvia moderada" },
      { value: 1.5, color: "#0055FF", label: "Lluvia leve" },
      { value: 0, color: "#b7b7b7", label: "Granizo / Llovisna / Nieve" },
    ],
  },
  RHOHV: {
    steps: [
      { value: "1", color: "#ff00ff", label: "Agua" },
      {
        value: ".98",
        color: "#ab0000",
        label: "Gotas de agua de varios tamaños",
      },
      { value: ".95", color: "#fc886f", label: "Granizo" },
      { value: ".9", color: "#F9EA3C", label: "Banda brillante" },
      { value: ".85", color: "#04cf00", label: "Banda brillante" },
      { value: ".8", color: "#7be3fb", label: "Eco no meteorológico" },
      { value: ".7", color: "#4fc7ff", label: "Eco no meteorológico" },
      { value: ".5", color: "#458cd4", label: "Eco no meteorológico" },
      { value: ".3", color: "#2948f7", label: "Eco no meteorológico" },
    ],
  },
  KDP: {
    steps: [
      { value: 7, color: "#ff00ff", label: "Hielo recubierto en agua" },
      { value: 6, color: "#ff0053", label: "Lluvia muy intensa" },
      { value: 5, color: "#fb4a00", label: "Lluvia muy intensa" },
      { value: 4, color: "#f5cc00", label: "Lluvia intensa" },
      { value: 3, color: "#2cac3b", label: "Lluvia moderada/intensa" },
      { value: 2, color: "#38DC70", label: "Lluvia moderada" },
      { value: 1, color: "#769ED5", label: "Llovisna" },
      { value: 0, color: "#dce7ed", label: "Granizo" },
    ],
  },
  VRAD: {
    steps: [
      { value: 70, color: "#FF29E3", label: "Lluvia muy intensa y granizo" },
      { value: 30, color: "#ff0000", label: "m/s desde el radar" },
      { value: 20, color: "#d40000", label: "m/s desde el radar" },
      { value: 10, color: "#a40000", label: "m/s desde el radar" },
      { value: 0, color: "#200b0b", label: "tangencial al radar" },
      { value: -10, color: "#009f00", label: "m/s hacia el radar" },
      { value: -20, color: "#00cf00", label: "m/s hacia el radar" },
      { value: -30, color: "#00ff00", label: "m/s hacia el radar" },
    ],
  },
  WRAD: {
    steps: [
      { value: 8, color: "#7f2704", label: "Muy alta dispersión" },
      { value: 6, color: "#d94801", label: "Alta dispersión" },
      { value: 4, color: "#fd8d3c", label: "Moderada dispersión" },
      { value: 2, color: "#fdae6b", label: "Baja dispersión" },
      { value: 1, color: "#fdd0a2", label: "Muy baja dispersión" },
      { value: 0, color: "#fff5eb", label: "Sin dispersión" },
    ],
  },
  PHIDP: {
    steps: [
      { value: 180, color: "#800080", label: "Fase alta / ciclo completo" },
      { value: 120, color: "#0000ff", label: "Fase positiva" },
      { value: 60, color: "#00ffff", label: "Cambio de fase moderado" },
      { value: 0, color: "#00ff00", label: "Sin cambio de fase" },
      { value: -60, color: "#ffff00", label: "Cambio de fase leve (negativo)" },
      {
        value: -120,
        color: "#ff8000",
        label: "Cambio de fase moderado (negativo)",
      },
      {
        value: -180,
        color: "#ff0000",
        label: "Fase negativa / ciclo completo",
      },
    ],
  },
};

// Rango de valores por campo (vmin y vmax) para colormaps dinámicos
const FIELD_RANGES = {
  DBZH: { vmin: -30, vmax: 70 },
  DBZHF: { vmin: -30, vmax: 70 },
  DBZV: { vmin: -30, vmax: 70 },
  ZDR: { vmin: -5, vmax: 10.5 },
  RHOHV: { vmin: 0, vmax: 1 },
  KDP: { vmin: 0, vmax: 8 },
  VRAD: { vmin: -35, vmax: 35 },
  WRAD: { vmin: 0, vmax: 10 },
  PHIDP: { vmin: 0, vmax: 360 },
};

// Mapeo de colormaps default por campo
const DEFAULT_COLORMAPS = {
  DBZH: "grc_th",
  DBZHF: "grc_th",
  DBZV: "grc_th",
  ZDR: "grc_zdr2",
  RHOHV: "grc_rho",
  KDP: "grc_rain",
  VRAD: "NWSVel",
  WRAD: "Oranges",
  PHIDP: "Theodore16",
};

// Extraer solo los valores de los LEGENDS para usarlos consistentemente
const FIELD_VALUES = {
  DBZH: [70, 60, 50, 40, 30, 20, 10, 0, -10],
  DBZHF: [70, 60, 50, 40, 30, 20, 10, 0, -10],
  DBZV: [70, 60, 50, 40, 30, 20, 10, 0, -10],
  ZDR: [7, 5.5, 4, 3, 2.5, 1.5, 0],
  RHOHV: ["1", ".98", ".95", ".9", ".85", ".8", ".7", ".5", ".3"],
  KDP: [7, 6, 5, 4, 3, 2, 1, 0],
  VRAD: [70, 30, 20, 10, 0, -10, -20, -30],
  WRAD: [8, 6, 4, 2, 1, 0],
  PHIDP: [180, 120, 60, 0, -60, -120, -180],
};

export default function ColorLegend({
  overlayData, // Array de capas actuales con field y colormap
  style,
}) {
  const [colormapCache, setColormapCache] = useState({});
  const [loading, setLoading] = useState(false);

  // Extraer campos únicos con sus colormaps
  const fieldColormapPairs = React.useMemo(() => {
    if (!Array.isArray(overlayData) || overlayData.length === 0) {
      return [];
    }

    const pairs = new Map();
    overlayData.forEach((layer) => {
      if (layer.field) {
        const colormap =
          layer.colormap || DEFAULT_COLORMAPS[layer.field] || "grc_th";
        const key = `${layer.field}_${colormap}`;
        if (!pairs.has(key)) {
          pairs.set(key, {
            field: layer.field,
            colormap: colormap,
          });
        }
      }
    });

    return Array.from(pairs.values());
  }, [overlayData]);

  // Determinar qué colormaps necesitan cargarse dinámicamente
  const dynamicColormaps = React.useMemo(() => {
    return fieldColormapPairs.filter(({ field, colormap }) => {
      const fieldKey = String(field || "").toUpperCase();
      const defaultCmap = DEFAULT_COLORMAPS[fieldKey];
      // Solo cargar dinámicamente si NO es el colormap default
      return colormap !== defaultCmap;
    });
  }, [fieldColormapPairs]);

  // Cargar colormaps necesarios (solo los no-default)
  useEffect(() => {
    const loadColormaps = async () => {
      const missingColormaps = dynamicColormaps.filter(
        ({ colormap }) => !colormapCache[colormap],
      );

      if (missingColormaps.length === 0) return;

      setLoading(true);
      try {
        const promises = missingColormaps.map(async ({ colormap }) => {
          try {
            const data = await getColormapColors(colormap, 10); // 10 pasos para la leyenda
            return { colormap, colors: data.colors };
          } catch (error) {
            console.error(`Error loading colormap ${colormap}:`, error);
            return { colormap, colors: null };
          }
        });

        const results = await Promise.all(promises);
        const newCache = { ...colormapCache };
        results.forEach(({ colormap, colors }) => {
          if (colors) {
            newCache[colormap] = colors;
          }
        });
        setColormapCache(newCache);
      } catch (error) {
        console.error("Error loading colormaps:", error);
      } finally {
        setLoading(false);
      }
    };

    loadColormaps();
  }, [dynamicColormaps]);

  const renderLegendColumn = ({ field, colormap }) => {
    const fieldKey = String(field || "").toUpperCase();
    const defaultCmap = DEFAULT_COLORMAPS[fieldKey];
    const isDefaultColormap = colormap === defaultCmap;

    // Si es el colormap default, usar el hardcoded
    if (isDefaultColormap && LEGENDS[fieldKey]) {
      const legendData = LEGENDS[fieldKey];

      return (
        <div
          key={`${fieldKey}_${colormap}`}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: 60,
          }}
        >
          <Typography variant="subtitle" color="white" gutterBottom>
            {fieldKey}
          </Typography>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 5,
            }}
          >
            {legendData.steps.map((item) => {
              const textColor = getTextColor(item.color);
              return (
                <div
                  key={`${fieldKey}-${item.value}`}
                  title={item.label}
                  style={{
                    width: 23,
                    height: 23,
                    borderRadius: "50%",
                    backgroundColor: item.color,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: "bold",
                    color: textColor,
                    fontSize: 15,
                  }}
                >
                  {item.value}
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    // Para colormaps no-default, usar el dinámico
    const key = `${field}_${colormap}`;
    const colors = colormapCache[colormap];

    if (!colors) {
      return null; // Todavía cargando
    }

    // Usar los mismos valores que el hardcoded para consistencia
    const fieldValues = FIELD_VALUES[fieldKey] || [];

    if (fieldValues.length === 0) {
      return null; // Sin valores definidos para este campo
    }

    // Obtener rango del campo
    const range = FIELD_RANGES[fieldKey] || { vmin: 0, vmax: 100 };
    const numColors = colors.length;

    // Mapear cada valor a su color correspondiente según normalización
    // colors[0] = vmin, colors[numColors-1] = vmax
    const colorValuePairs = fieldValues.map((value) => {
      // Normalizar el valor entre 0 y 1
      const numValue = typeof value === "string" ? parseFloat(value) : value;
      const normalized = (numValue - range.vmin) / (range.vmax - range.vmin);

      // Obtener índice del color (clampeado entre 0 y numColors-1)
      const colorIndex = Math.max(
        0,
        Math.min(numColors - 1, Math.round(normalized * (numColors - 1))),
      );

      return {
        color: colors[colorIndex],
        value: value,
      };
    });

    return (
      <div
        key={key}
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          minWidth: 60,
        }}
      >
        <Typography variant="subtitle" color="white" gutterBottom>
          {fieldKey}
        </Typography>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 5,
          }}
        >
          {colorValuePairs.map(({ color, value }, idx) => {
            const textColor = getTextColor(color);

            return (
              <div
                key={`${key}-${idx}`}
                title={`${fieldKey}: ${value}`}
                style={{
                  width: 23,
                  height: 23,
                  borderRadius: "50%",
                  backgroundColor: color,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: "bold",
                  color: textColor,
                  fontSize: 15,
                }}
              >
                {value}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  if (fieldColormapPairs.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        bottom: 10,
        zIndex: 1000,
        display: "flex",
        flexDirection: "row",
        alignItems: "flex-start",
        gap: 15,
        padding: 4,
        ...style,
      }}
    >
      {fieldColormapPairs.map(renderLegendColumn)}
    </div>
  );
}
