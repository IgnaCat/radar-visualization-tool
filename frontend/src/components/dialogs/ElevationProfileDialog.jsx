import { useEffect, useState, useRef } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Button,
  Box,
  Typography,
  Paper,
  CircularProgress,
  IconButton,
} from "@mui/material";
import { Close as CloseIcon, Add as AddIcon } from "@mui/icons-material";
import Draggable from "react-draggable";
import {
  AreaChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";

function PaperComponent(props) {
  const nodeRef = useRef(null);
  return (
    <Draggable
      nodeRef={nodeRef}
      handle="#draggable-dialog-title"
      cancel={'[class*="MuiDialogContent-root"]'}
    >
      <Paper {...props} ref={nodeRef} />
    </Draggable>
  );
}

/**
 * Custom tooltip para mostrar información detallada al hacer hover
 */
function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <Paper
        sx={{
          padding: 1.5,
          backgroundColor: "rgba(255, 255, 255, 0.95)",
          border: "1px solid #ccc",
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: "bold" }}>
          Distancia: {data.distance.toFixed(2)} km
        </Typography>
        <Typography variant="body2" color="primary">
          Altura: {data.elevation.toFixed(0)} m
        </Typography>
        <Typography variant="caption" color="textSecondary">
          Lat: {data.lat.toFixed(4)}°, Lon: {data.lon.toFixed(4)}°
        </Typography>
      </Paper>
    );
  }
  return null;
}

/**
 * Props:
 * - open: boolean
 * - onClose: () => void
 * - onRequestDraw: () => void - solicitar que el usuario dibuje una línea
 * - drawnCoordinates: {lat, lon}[] - coordenadas dibujadas por el usuario
 * - onGenerate: (coordinates) => Promise<response>
 * - onClearDrawing: () => void - limpiar el dibujo del mapa
 * - onHighlightPoint: (lat, lon) => void - resaltar un punto en el mapa
 */
export default function ElevationProfileDialog({
  open,
  onClose,
  onRequestDraw,
  drawnCoordinates = [],
  drawingFinished = false,
  onGenerate,
  onClearDrawing,
  onHighlightPoint,
  onProfileGenerated,
}) {
  const [loading, setLoading] = useState(false);
  const [profileData, setProfileData] = useState(null);
  const [error, setError] = useState("");
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  // Limpiar al cerrar
  const handleClose = () => {
    setProfileData(null);
    setError("");
    setHoveredPoint(null);
    setIsDrawing(false);
    setIsMinimized(false);
    onClearDrawing?.();
    onClose();
  };

  // Iniciar modo dibujo
  const handleStartDrawing = () => {
    setProfileData(null);
    setError("");
    setIsDrawing(true);
    setIsMinimized(true); // Minimizar el diálogo
    onRequestDraw?.();
  };

  // Generar perfil cuando el usuario completa el dibujo
  useEffect(() => {
    if (!drawingFinished || !drawnCoordinates || drawnCoordinates.length < 2) {
      return;
    }

    const generateProfile = async () => {
      setIsDrawing(false);
      setIsMinimized(false); // Restaurar el diálogo
      setLoading(true);
      setError("");

      try {
        const response = await onGenerate(drawnCoordinates);
        setProfileData(response.data);
        onProfileGenerated?.(); // Notificar que se generó el perfil
      } catch (err) {
        setError(err?.response?.data?.detail || String(err));
        onProfileGenerated?.(); // Notificar incluso si hay error
      } finally {
        setLoading(false);
      }
    };

    generateProfile();
  }, [drawingFinished, drawnCoordinates, onGenerate, onProfileGenerated]);

  // Manejar hover en el gráfico
  const handleMouseMove = (e) => {
    if (e && e.activePayload && e.activePayload.length > 0) {
      const point = e.activePayload[0].payload;
      setHoveredPoint(point);
      onHighlightPoint?.(point.lat, point.lon);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
    onHighlightPoint?.(null, null);
  };

  // Calcular ancho dinámico basado en la distancia total
  const calculateDialogWidth = () => {
    if (!profileData?.profile || profileData.profile.length === 0) {
      return "sm";
    }
    const lastPoint = profileData.profile[profileData.profile.length - 1];
    const totalDistance = lastPoint?.distance || 0;

    // Escalar el ancho según la distancia
    if (totalDistance < 80) return "sm"; // ~600px
    if (totalDistance < 600) return "md"; // ~900px
    if (totalDistance < 900) return "lg"; // ~1200px
    return "xl"; // ~1536px
  };

  return (
    <Dialog
      open={open && !isMinimized}
      onClose={handleClose}
      fullWidth
      maxWidth={calculateDialogWidth()}
      hideBackdrop
      disableEnforceFocus
      disableAutoFocus
      disableRestoreFocus
      disableScrollLock
      slotProps={{
        root: { sx: { pointerEvents: "none" } },
      }}
      PaperProps={{
        sx: { pointerEvents: "auto", minHeight: "320px" },
      }}
      PaperComponent={PaperComponent}
      aria-labelledby="draggable-dialog-title"
    >
      <DialogTitle
        id="draggable-dialog-title"
        sx={{
          cursor: "move",
          backgroundColor: "#f5f5f5",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          paddingRight: 0.5,
          paddingTop: 0.5,
          paddingBottom: 0.5,
        }}
      >
        <Typography
          variant="subtitle2"
          component="span"
          sx={{ fontWeight: 600 }}
        >
          Perfil de Elevación
        </Typography>
        <Box sx={{ display: "flex", gap: 0.5 }}>
          {profileData && (
            <IconButton
              onClick={handleStartDrawing}
              color="primary"
              size="small"
              title="Dibujar nueva línea"
              sx={{
                "& .MuiSvgIcon-root": {
                  fontSize: "1.25rem",
                },
              }}
            >
              <AddIcon />
            </IconButton>
          )}
          <IconButton
            onClick={handleClose}
            color="secondary"
            size="small"
            title="Cerrar"
            sx={{
              "& .MuiSvgIcon-root": {
                fontSize: "1.25rem",
              },
            }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers sx={{ minHeight: "250px" }}>
        {!profileData && !loading && !error && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              minHeight: "220px",
            }}
          >
            <Typography variant="body1" sx={{ mb: 2 }}>
              {isDrawing
                ? "Haz click en el mapa para agregar puntos. Click en el cuadrado blanco para terminar."
                : "Dibuja una línea en el mapa para generar el perfil de elevación."}
            </Typography>
            {!isDrawing && (
              <Button variant="contained" onClick={handleStartDrawing}>
                Comenzar a dibujar
              </Button>
            )}
            <Typography
              variant="caption"
              sx={{ mt: 2, color: "text.secondary" }}
            >
              Atajos: ESC para cancelar, Enter para terminar, Delete para borrar
              último punto
            </Typography>
          </Box>
        )}

        {loading && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              minHeight: "220px",
            }}
          >
            <CircularProgress />
            <Typography sx={{ mt: 2 }}>
              Generando perfil de elevación...
            </Typography>
          </Box>
        )}

        {error && (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
            }}
          >
            <Typography color="error">{error}</Typography>
          </Box>
        )}

        {profileData &&
          profileData.profile &&
          profileData.profile.length > 0 && (
            <Box>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart
                  data={profileData.profile}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={handleMouseLeave}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <defs>
                    <linearGradient
                      id="elevationGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop offset="0%" stopColor="#ff6b35" stopOpacity={0.9} />
                      <stop
                        offset="95%"
                        stopColor="#ff6b35"
                        stopOpacity={0.1}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="distance"
                    label={{
                      value: "Distancia (km)",
                      position: "insideBottom",
                      offset: -5,
                    }}
                    interval="preserveStartEnd"
                    minTickGap={40}
                    tickFormatter={(value) => Math.round(value)}
                  />
                  <YAxis
                    label={{
                      value: "Altura (m)",
                      angle: -90,
                      position: "insideLeft",
                    }}
                    tickFormatter={(value) => value.toFixed(0)}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="elevation"
                    stroke="none"
                    fill="url(#elevationGradient)"
                  />
                  <Line
                    type="monotone"
                    dataKey="elevation"
                    stroke="#ff6b35"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6 }}
                  />
                  {hoveredPoint && (
                    <ReferenceDot
                      x={hoveredPoint.distance}
                      y={hoveredPoint.elevation}
                      r={8}
                      fill="#ff0000"
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </Box>
          )}

        {profileData &&
          profileData.profile &&
          profileData.profile.length === 0 && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
              }}
            >
              <Typography>
                No se pudo obtener datos de elevación para la línea dibujada.
              </Typography>
            </Box>
          )}
      </DialogContent>
    </Dialog>
  );
}
