import React from "react";
import ColorLegendField from "./ColorLegendField";

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

  // Mantener slots estables por leyenda para que nuevas capas no "salten"
  // a posiciones previas cuando cambia el orden del overlay.
  const [legendSlots, setLegendSlots] = React.useState({});

  React.useEffect(() => {
    const keys = fieldColormapPairs.map(
      ({ field, colormap }) => `${field}_${colormap}`,
    );

    setLegendSlots((prev) => {
      const next = {};
      const used = new Set();

      // Preservar slots ya asignados para claves todavía visibles
      keys.forEach((k) => {
        if (Object.prototype.hasOwnProperty.call(prev, k)) {
          next[k] = prev[k];
          used.add(prev[k]);
        }
      });

      // Asignar slots consecutivos a nuevas claves
      keys.forEach((k) => {
        if (!Object.prototype.hasOwnProperty.call(next, k)) {
          let slot = 0;
          while (used.has(slot)) slot += 1;
          next[k] = slot;
          used.add(slot);
        }
      });

      return next;
    });
  }, [fieldColormapPairs]);

  if (fieldColormapPairs.length === 0) {
    return null;
  }

  return (
    <div style={{ display: "contents" }}>
      {fieldColormapPairs.map(({ field, colormap }) => {
        const key = `${field}_${colormap}`;
        const slot = legendSlots[key] ?? 0;
        return (
          <ColorLegendField
            key={key}
            field={field}
            colormap={colormap}
            defaultX={10 + slot * 100} // Escalonar horizontalmente desde abajo-izquierda
            defaultY={10}
            fromBottom={true}
          />
        );
      })}
    </div>
  );
}
