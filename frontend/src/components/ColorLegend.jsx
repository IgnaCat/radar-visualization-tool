import React from "react";

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
    16
  );
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  // luminancia perceptual
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.6 ? "#000" : "#FFF";
}

// --- Configuración de paletas ---
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
};

export default function ColorLegend({ field = "DBZH" }) {
  const legendData = LEGENDS[field.toUpperCase()] || LEGENDS.DBZH;

  return (
    <div
      style={{
        position: "absolute",
        left: 40,
        bottom: 10,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "5px",
      }}
    >
      {legendData.steps.map((item) => {
        const textColor = getTextColor(item.color);
        return (
          <div
            key={item.value}
            title={item.label}
            style={{
              width: "26px",
              height: "26px",
              borderRadius: "50%",
              backgroundColor: item.color,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: "bold",
              color: textColor,
              fontSize: "16px",
            }}
          >
            {item.value}
          </div>
        );
      })}
    </div>
  );
}
