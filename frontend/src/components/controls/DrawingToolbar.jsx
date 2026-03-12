import { useState } from "react";
import { Box, IconButton, Paper, Tooltip, Zoom } from "@mui/material";
import { useSnackbar } from "notistack";
import CreateIcon from "@mui/icons-material/Create";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import TextFieldsIcon from "@mui/icons-material/TextFields";
import TimelineIcon from "@mui/icons-material/Timeline";
import TrendingFlatIcon from "@mui/icons-material/TrendingFlat";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import PolylineIcon from "@mui/icons-material/Polyline";

const ANNOTATION_LABELS = {
  text: "texto",
  line: "línea",
  arrow: "flecha",
  rect: "rectángulo",
  circle: "círculo",
  polygon: "polígono",
};

/**
 * DrawingToolbar - Barra de herramientas de anotaciones visuales sobre el mapa.
 * Expande hacia abajo al hacer clic en el botón principal (lápiz).
 *
 * Props:
 * - markerMode: booleano que indica si está activo el modo de marcadores
 * - onToggleMarkerMode: función para alternar modo de marcadores
 * - annotationMode: 'text'|'line'|'arrow'|'rect'|'circle'|'polygon'|null
 * - onSetAnnotationMode: función(mode) para activar/desactivar un modo de anotación
 */
export default function DrawingToolbar({
  markerMode = false,
  onToggleMarkerMode,
  annotationMode = null,
  onSetAnnotationMode,
}) {
  const [expanded, setExpanded] = useState(false);
  const { enqueueSnackbar } = useSnackbar();

  const handleMarkerClick = () => {
    try {
      onToggleMarkerMode?.();
      enqueueSnackbar(
        markerMode ? "Modo marcador desactivado" : "Modo marcador activado",
        { variant: "info" },
      );
    } catch {
      enqueueSnackbar("Error al cambiar modo marcador", { variant: "error" });
    }
  };

  const handleAnnotationTool = (mode) => {
    try {
      onSetAnnotationMode?.(mode);
      const isDeactivating = annotationMode === mode;
      const label = ANNOTATION_LABELS[mode] ?? mode;
      enqueueSnackbar(
        isDeactivating ? `Modo ${label} desactivado` : `Modo ${label} activado`,
        { variant: "info" },
      );
    } catch {
      enqueueSnackbar("Error al cambiar modo de dibujo", { variant: "error" });
    }
  };

  const tools = [
    {
      icon: <LocationOnIcon />,
      tooltip: markerMode ? "Desactivar marcadores" : "Agregar marcadores",
      action: handleMarkerClick,
      active: markerMode,
    },
    {
      icon: <TextFieldsIcon />,
      tooltip:
        annotationMode === "text"
          ? "Desactivar texto"
          : "Agregar texto (click para colocar)",
      action: () => handleAnnotationTool("text"),
      active: annotationMode === "text",
    },
    {
      icon: <TimelineIcon />,
      tooltip:
        annotationMode === "line"
          ? "Cancelar línea"
          : "Dibujar línea (click para puntos, cuadrado para terminar)",
      action: () => handleAnnotationTool("line"),
      active: annotationMode === "line",
    },
    {
      icon: <TrendingFlatIcon />,
      tooltip:
        annotationMode === "arrow"
          ? "Cancelar flecha"
          : "Dibujar flecha (click para puntos, cuadrado para terminar)",
      action: () => handleAnnotationTool("arrow"),
      active: annotationMode === "arrow",
    },
    {
      icon: <CropSquareIcon />,
      tooltip:
        annotationMode === "rect"
          ? "Cancelar rectángulo"
          : "Dibujar rectángulo (2 clics: esquinas opuestas)",
      action: () => handleAnnotationTool("rect"),
      active: annotationMode === "rect",
    },
    {
      icon: <RadioButtonUncheckedIcon />,
      tooltip:
        annotationMode === "circle"
          ? "Cancelar círculo"
          : "Dibujar círculo (1er clic: centro, 2do: radio)",
      action: () => handleAnnotationTool("circle"),
      active: annotationMode === "circle",
    },
    {
      icon: <PolylineIcon />,
      tooltip:
        annotationMode === "polygon"
          ? "Cancelar polígono"
          : "Dibujar polígono (click para vértices, cuadrado para cerrar)",
      action: () => handleAnnotationTool("polygon"),
      active: annotationMode === "polygon",
    },
  ];

  return (
    <Paper
      elevation={0}
      className="no-print"
      sx={{
        position: "absolute",
        top: 58,
        right: 12,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        backgroundColor: "#fff",
        borderRadius: "8px",
        padding: "4px",
        gap: "4px",
      }}
    >
      {/* Botón de lápiz (toggle) */}
      <Tooltip
        title={
          expanded ? "Ocultar herramientas" : "Mostrar herramientas de dibujo"
        }
        placement="left"
      >
        <IconButton
          onClick={() => setExpanded((v) => !v)}
          sx={{
            width: 30,
            height: 30,
            borderRadius: "6px",
            color: "#000",
            backgroundColor: expanded ? "rgba(0, 0, 0, 0.08)" : "transparent",
            transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            "& .MuiSvgIcon-root": {
              fontSize: "1.25rem",
            },
            "&:hover": {
              backgroundColor: "rgba(0, 0, 0, 0.08)",
            },
          }}
        >
          <CreateIcon />
        </IconButton>
      </Tooltip>

      {/* Herramientas expandibles hacia abajo */}
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          overflow: "hidden",
          maxHeight: expanded ? `${tools.length * 38}px` : "0px",
          transition: "max-height 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        {tools.map((tool, index) => (
          <Zoom
            key={index}
            in={expanded}
            style={{
              transitionDelay: expanded ? `${index * 50}ms` : "0ms",
            }}
          >
            <Tooltip title={tool.tooltip} placement="left">
              <IconButton
                onClick={tool.action}
                sx={{
                  width: 30,
                  height: 30,
                  borderRadius: "6px",
                  color: "#000",
                  backgroundColor: tool.active
                    ? "rgba(74, 144, 226, 0.2)"
                    : "transparent",
                  transition: "all 0.2s ease",
                  "& .MuiSvgIcon-root": {
                    fontSize: "1.25rem",
                  },
                  "&:hover": {
                    backgroundColor: tool.active
                      ? "rgba(74, 144, 226, 0.3)"
                      : "rgba(0, 0, 0, 0.08)",
                    transform: "scale(1.05)",
                  },
                }}
              >
                {tool.icon}
              </IconButton>
            </Tooltip>
          </Zoom>
        ))}
      </Box>
    </Paper>
  );
}
