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

  if (fieldColormapPairs.length === 0) {
    return null;
  }

  return (
    <div style={{ display: "contents" }}>
      {fieldColormapPairs.map(({ field, colormap }, idx) => (
        <ColorLegendField
          key={`${field}_${colormap}`}
          field={field}
          colormap={colormap}
          defaultX={10 + idx * 100} // Escalonar horizontalmente desde abajo-izquierda
          defaultY={10}
          fromBottom={true}
        />
      ))}
    </div>
  );
}
