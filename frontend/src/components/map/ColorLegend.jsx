import React, { useState, useEffect } from "react";
import { Typography } from "@mui/material";
import { getColormapLegend } from "../../api/backend";

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

export default function ColorLegend({
  overlayData, // Array de capas actuales con field y colormap
  style,
}) {
  const [legendCache, setLegendCache] = useState({});

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

  // Cargar todas las leyendas desde backend (valores + colores)
  useEffect(() => {
    const loadLegends = async () => {
      const missingLegends = fieldColormapPairs.filter(
        ({ field, colormap }) =>
          !legendCache[`${String(field).toUpperCase()}_${colormap}`],
      );

      if (missingLegends.length === 0) return;

      try {
        const promises = missingLegends.map(async ({ field, colormap }) => {
          const fieldKey = String(field).toUpperCase();
          const cacheKey = `${fieldKey}_${colormap}`;
          try {
            const data = await getColormapLegend(fieldKey, colormap);
            return {
              cacheKey,
              legend: {
                values: data.values,
                colors: data.colors,
              },
            };
          } catch (error) {
            console.error(
              `Error loading legend ${fieldKey}/${colormap}:`,
              error,
            );
            return { cacheKey, legend: null };
          }
        });

        const results = await Promise.all(promises);
        setLegendCache((prevCache) => {
          const newCache = { ...prevCache };
          results.forEach(({ cacheKey, legend }) => {
            if (legend) {
              newCache[cacheKey] = legend;
            }
          });
          return newCache;
        });
      } catch (error) {
        console.error("Error loading legends:", error);
      }
    };

    loadLegends();
  }, [fieldColormapPairs, legendCache]);

  const renderLegendColumn = ({ field, colormap }) => {
    const fieldKey = String(field || "").toUpperCase();

    const cacheKey = `${fieldKey}_${colormap}`;
    const legend = legendCache[cacheKey];
    if (!legend) {
      return null; // Todavía cargando
    }

    const colors = legend.colors || [];
    const fieldValues = legend.values || [];
    if (fieldValues.length === 0) {
      return null; // Sin valores definidos para este campo
    }

    // Crear gradiente continuo
    // Los colores van de arriba (mayor valor) hacia abajo (menor valor)
    const gradient = `linear-gradient(to bottom, ${colors.join(", ")})`;

    // Altura de la barra
    const barHeight = 200;
    const barWidth = 14;

    return (
      <div
        key={`${fieldKey}_${colormap}`}
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          gap: 0,
          marginLeft: 4,
        }}
      >
        {/* Barra de colores continua */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            position: "relative",
          }}
        >
          <Typography
            variant="subtitle2"
            color="white"
            sx={{
              marginBottom: 1,
              fontWeight: "bold",
              fontSize: "0.8rem",
              textShadow: "1px 1px 1px rgba(0,0,0,0.9)",
            }}
          >
            {fieldKey}
          </Typography>

          {/* Barra con gradiente */}
          <div
            style={{
              width: barWidth,
              height: barHeight,
              background: gradient,
              borderRadius: 6,
              position: "relative",
            }}
          />
        </div>

        {/* Valores a la derecha de la barra */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            height: barHeight,
            marginTop: 22, // Compensar el título
          }}
        >
          {fieldValues.map((value, idx) => (
            <div
              key={`${fieldKey}-value-${idx}`}
              style={{
                fontSize: "0.7rem",
                color: "white",
                fontWeight: "350",
                textShadow: "1px 1px 1px rgba(0,0,0,0.9)",
                lineHeight: 1,
              }}
              title={`${fieldKey}: ${value}`}
            >
              {value}
            </div>
          ))}
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
        left: 10,
        bottom: 10,
        zIndex: 1000,
        display: "flex",
        flexDirection: "row",
        alignItems: "flex-start",
        gap: 4,
        padding: 2,
        ...style,
      }}
    >
      {fieldColormapPairs.map(renderLegendColumn)}
    </div>
  );
}
