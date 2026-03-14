import { keyframes } from "@mui/system";
import { Box, IconButton, Paper, Tooltip } from "@mui/material";
import VisibilityIcon from "@mui/icons-material/Visibility";
import ContentCutIcon from "@mui/icons-material/ContentCut";
import PercentIcon from "@mui/icons-material/Percent";
import ImageSearchIcon from "@mui/icons-material/ImageSearch";
import LayersIcon from "@mui/icons-material/Layers";
import TimelineIcon from "@mui/icons-material/Timeline";
import MapIcon from "@mui/icons-material/Map";
import PaletteIcon from "@mui/icons-material/Palette";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";

const activeModePulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(126, 211, 255, 0.28), 0 2px 6px rgba(0, 0, 0, 0.2);
    transform: scale(1);
  }
  50% {
    box-shadow: 0 0 0 6px rgba(126, 211, 255, 0.08), 0 4px 10px rgba(31, 71, 117, 0.28);
    transform: scale(1.04);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(126, 211, 255, 0), 0 2px 6px rgba(0, 0, 0, 0.2);
    transform: scale(1);
  }
`;

export default function VerticalToolbar({
  onChangeProductClick,
  onPseudoRhiClick,
  onAreaStatsClick,
  onPixelStatToggle,
  onMapSelectorToggle,
  onPaletteSelectorToggle,
  onElevationProfileClick,
  onLayerManagerToggle,
  onFileManagerToggle,
  areaStatsActive = false,
  pixelStatActive = false,
  mapSelectorActive = false,
  paletteSelectorActive = false,
  layerManagerActive = false,
  fileManagerActive = false,
}) {
  const tools = [
    {
      icon: <VisibilityIcon />,
      tooltip: "Opciones de visualización",
      action: onChangeProductClick,
      active: false,
    },
    {
      icon: <LayersIcon />,
      tooltip: "Capas",
      action: onLayerManagerToggle,
      active: layerManagerActive,
    },
    {
      icon: <FolderOpenIcon />,
      tooltip: "Archivos cargados",
      action: onFileManagerToggle,
      active: fileManagerActive,
    },
    {
      icon: <MapIcon />,
      tooltip: "Mapas base",
      action: onMapSelectorToggle,
      active: mapSelectorActive,
    },
    {
      icon: <PaletteIcon />,
      tooltip: "Paletas de colores",
      action: onPaletteSelectorToggle,
      active: paletteSelectorActive,
    },
    {
      icon: <ContentCutIcon />,
      tooltip: "Generar Pseudo-RHI",
      action: onPseudoRhiClick,
      active: false,
    },
    {
      icon: <PercentIcon />,
      tooltip: "Estadísticas de área",
      action: onAreaStatsClick,
      active: areaStatsActive,
      mode: true,
    },
    {
      icon: <ImageSearchIcon />,
      tooltip: "Ver valor pixel",
      action: onPixelStatToggle,
      active: pixelStatActive,
      mode: true,
    },
    {
      icon: <TimelineIcon />,
      tooltip: "Perfil de elevación",
      action: onElevationProfileClick,
      active: false,
    },
  ];

  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        top: 70, // Debajo del HeaderCard
        left: 12,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        backgroundColor: "transparent",
        backdropFilter: "none",
        borderRadius: "8px",
        boxShadow: "none",
        padding: "8px 0",
      }}
    >
      {tools.map((tool, index) => (
        <Box key={index}>
          <Tooltip
            title={
              tool.mode && tool.active ? `${tool.tooltip} activo` : tool.tooltip
            }
            placement="right"
          >
            <IconButton
              onClick={tool.action}
              aria-pressed={tool.active}
              sx={{
                width: 30,
                height: 30,
                borderRadius: "8px",
                margin: "2px 7px",
                color: "#fff",
                position: "relative",
                overflow: "visible",
                backgroundColor:
                  tool.mode && tool.active
                    ? "rgba(111, 191, 235, 1)"
                    : tool.active
                      ? "rgba(74, 144, 226, 1)"
                      : "rgba(74, 144, 226, 0.85)",
                boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
                transition: "all 0.25s ease",
                animation:
                  tool.mode && tool.active
                    ? `${activeModePulse} 2.2s ease-in-out infinite`
                    : "none",
                "&::after":
                  tool.mode && tool.active
                    ? {
                        content: '""',
                        position: "absolute",
                        top: -3,
                        right: -3,
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        backgroundColor: "#dff7ff",
                        boxShadow: "0 0 0 2px rgba(50, 106, 164, 0.7)",
                      }
                    : undefined,
                "& .MuiSvgIcon-root": {
                  fontSize: "1.25rem", // Mantiene el tamaño del icono
                },
                "&:hover": {
                  backgroundColor:
                    tool.mode && tool.active
                      ? "rgba(126, 211, 255, 1)"
                      : "rgba(74, 144, 226, 1)",
                  boxShadow: "0 3px 8px rgba(0,0,0,0.3)",
                  transform:
                    tool.mode && tool.active ? "scale(1.05)" : "scale(1.02)",
                },
              }}
            >
              {tool.icon}
            </IconButton>
          </Tooltip>
          {/* Divider opcional después del 4to elemento (después de paletas) */}
          {index === 4 && (
            <Box
              sx={{
                height: "1px",
                backgroundColor: "rgba(255, 255, 255, 0.2)",
                margin: "4px 12px",
              }}
            />
          )}
        </Box>
      ))}
    </Paper>
  );
}
