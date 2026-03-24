import React, { useEffect, useState } from "react";
import { Typography, Paper } from "@mui/material";
import { getColormapLegend } from "../../api/backend";
import { useDraggableElement } from "../../hooks/useDraggableElement";

function getLabelOffsetPercent(value, vmin, vmax) {
  const numericValue = Number(value);

  if (
    !Number.isFinite(numericValue) ||
    !Number.isFinite(vmin) ||
    !Number.isFinite(vmax) ||
    vmax === vmin
  ) {
    return 50;
  }

  const normalized = (numericValue - vmin) / (vmax - vmin);
  const clamped = Math.min(Math.max(normalized, 0), 1);
  return (1 - clamped) * 100;
}

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
      setLoading(true);

      try {
        const fieldKey = String(field).toUpperCase();
        const data = await getColormapLegend(fieldKey, colormap);
        setLegend({
          values: data.values,
          colors: data.colors,
          gradientColors: data.gradient_colors,
          unit: data.unit,
          vmin: data.vmin,
          vmax: data.vmax,
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
  const gradientColors = legend.gradientColors || colors;
  const fieldValues = legend.values || [];
  const unit = (legend.unit || "").trim();
  const vmin = Number(legend.vmin);
  const vmax = Number(legend.vmax);

  if (fieldValues.length === 0 || gradientColors.length === 0) {
    return null;
  }

  const fieldKey = String(field).toUpperCase();
  const gradient = `linear-gradient(to bottom, ${[...gradientColors].reverse().join(", ")})`;
  const barHeight = 200;
  const barWidth = 14;
  const labelsGap = 8;
  const labelsWidth = 36;

  return (
    <Paper
      ref={elementRef}
      onMouseDown={onMouseDown}
      sx={{
        ...dragStyle,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        paddingBlock: 1.5,
        paddingLeft: 2,
        paddingRight: 3.5,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        borderRadius: 3,
        userSelect: "none",
        width: "fit-content",
        overflow: "visible",
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
            position: "relative",
            width: barWidth,
            height: barHeight,
          }}
        >
          <div
            style={{
              width: barWidth,
              height: barHeight,
              background: gradient,
              borderRadius: 6,
              position: "relative",
            }}
          />

          {/* Valores a la derecha */}
          <div
            style={{
              position: "absolute",
              left: `calc(100% + ${labelsGap}px)`,
              top: 0,
              height: barHeight,
              width: labelsWidth,
            }}
          >
            {fieldValues.map((value, idx) => {
              const top = getLabelOffsetPercent(value, vmin, vmax);

              return (
                <div
                  key={`${fieldKey}-value-${idx}`}
                  style={{
                    position: "absolute",
                    top: `${top}%`,
                    transform: "translateY(-50%)",
                    fontSize: "0.7rem",
                    color: "white",
                    fontWeight: "350",
                    textShadow: "1px 1px 1px rgba(0,0,0,0.9)",
                    lineHeight: 1,
                    whiteSpace: "nowrap",
                  }}
                  title={`${fieldKey}: ${value}`}
                >
                  {value}
                </div>
              );
            })}
          </div>
        </div>

        {unit ? (
          <Typography
            variant="caption"
            sx={{
              color: "white",
              marginTop: 1,
              fontWeight: 500,
              fontSize: "0.68rem",
              textShadow: "1px 1px 1px rgba(0,0,0,0.9)",
            }}
          >
            {unit}
          </Typography>
        ) : null}
      </div>
    </Paper>
  );
}
