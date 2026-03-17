import React, { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { getColormapLegend } from "../../api/backend";
import { useDraggableElement } from "../../hooks/useDraggableElement";

export default function ColorLegendField({
  field,
  colormap,
  defaultX = 10,
  defaultY = 70,
  fromBottom = false,
}) {
  const [legend, setLegend] = useState(null);
  const [loading, setLoading] = useState(true);
  const {
    elementRef,
    style: dragStyle,
    onMouseDown,
  } = useDraggableElement(defaultX, defaultY, fromBottom);

  useEffect(() => {
    const loadLegend = async () => {
      try {
        const fieldKey = String(field).toUpperCase();
        const data = await getColormapLegend(fieldKey, colormap);
        setLegend({
          values: data.values,
          colors: data.colors,
        });
      } catch (error) {
        console.error(`Error loading legend ${field}/${colormap}:`, error);
        setLegend(null);
      } finally {
        setLoading(false);
      }
    };

    loadLegend();
  }, [field, colormap]);

  if (loading || !legend) {
    return null;
  }

  const colors = legend.colors || [];
  const fieldValues = legend.values || [];

  if (fieldValues.length === 0) {
    return null;
  }

  const fieldKey = String(field).toUpperCase();
  const gradient = `linear-gradient(to bottom, ${colors.join(", ")})`;
  const barHeight = 200;
  const barWidth = 14;

  return (
    <Paper
      ref={elementRef}
      onMouseDown={onMouseDown}
      sx={{
        ...dragStyle,
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: 2,
        padding: 1.5,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        borderRadius: 1,
        userSelect: "none",
      }}
    >
      {/* Contenedor principal */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Typography
          variant="subtitle2"
          sx={{
            color: "white",
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

      {/* Valores a la derecha */}
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
    </Paper>
  );
}
