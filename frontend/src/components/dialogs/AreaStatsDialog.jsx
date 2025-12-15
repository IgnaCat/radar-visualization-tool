// components/AreaStatsDialog.jsx
import { useEffect, useState, useRef } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Divider,
  Paper,
  Card,
  CardContent,
  Grid,
  Chip,
  IconButton,
  Tooltip,
} from "@mui/material";
import {
  TrendingDown,
  TrendingUp,
  ShowChart,
  Functions,
  Straighten,
  BarChart as BarChartIcon,
  CheckCircle,
  Add as AddIcon,
} from "@mui/icons-material";
import Draggable from "react-draggable";

// Mapeo de unidades para cada campo
const FIELD_UNITS = {
  WRAD: "m/s",
  KDP: "deg/km",
  DBZV: "dBZ",
  DBZH: "dBZ",
  DBZHF: "dBZ",
  ZDR: "dBZ",
  VRAD: "m/s",
  RHOHV: "",
  PHIDP: "deg",
};

// Nombres bonitos para los campos
const FIELD_LABELS = {
  DBZH: "Reflectividad Horizontal",
  DBZV: "Reflectividad Vertical",
  DBZHF: "Reflectividad Horizontal Filtrada",
  ZDR: "Diferencia de Reflectividad",
  RHOHV: "Correlación Cruzada",
  KDP: "Diferencial de Fase",
  VRAD: "Velocidad Radial",
  WRAD: "Ancho Espectral",
  PHIDP: "Fase Diferencial",
};

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
 * Props:
 * - open
 * - onClose
 * - requestFn: (payload) => Promise<{ noCoverage?: boolean, stats?: {min,max,mean,median,std,count,valid_pct}, hist?: {bins:number[],counts:number[]} }>
 * - payload: { filepath, field, product, elevation?, height?, filters?, polygon }
 */
export default function AreaStatsDialog({ open, onClose, requestFn, payload }) {
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError("");
    setResp(null);
    (async () => {
      try {
        const r = await requestFn(payload);
        setResp(r);
      } catch (e) {
        setError(e?.response?.data?.detail || String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [open, requestFn, payload]);

  const currentField = payload?.field || "DBZH";
  const currentUnit = FIELD_UNITS[currentField] || "";

  // Componente para una tarjeta de estadística
  const StatCard = ({ icon, label, value, color = "primary.main" }) => {
    const IconComponent = icon;
    return (
      <Card
        elevation={0}
        sx={{
          bgcolor: "background.paper",
          border: "1px solid",
          borderColor: "divider",
          height: "100%",
        }}
      >
        <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
          <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
            <IconComponent sx={{ fontSize: 20, color, mr: 1 }} />
            <Typography variant="caption" color="text.secondary">
              {label}
            </Typography>
          </Box>
          <Typography variant="h7" component="div" fontWeight="bold">
            {value}
            {currentUnit && (
              <Typography
                component="span"
                variant="body2"
                color="text.secondary"
                sx={{ ml: 0.5 }}
              >
                {currentUnit}
              </Typography>
            )}
          </Typography>
        </CardContent>
      </Card>
    );
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="md"
      hideBackdrop
      disableEnforceFocus
      disableAutoFocus
      disableRestoreFocus
      disableScrollLock
      slotProps={{
        root: { sx: { pointerEvents: "none" } },
      }}
      PaperProps={{
        sx: { pointerEvents: "auto" },
      }}
      PaperComponent={PaperComponent}
      aria-labelledby="draggable-dialog-title"
    >
      <DialogTitle
        id="draggable-dialog-title"
        sx={{
          cursor: "move",
          borderBottom: "1px solid",
          borderColor: "divider",
          pb: 2,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <BarChartIcon />
          <Typography variant="h6" component="span">
            Estadísticas del Área
          </Typography>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        {loading && (
          <Box sx={{ textAlign: "center", py: 4 }}>
            <Typography color="text.secondary">Calculando estadísticas...</Typography>
          </Box>
        )}

        {error && (
          <Box
            sx={{
              p: 2,
              bgcolor: "error.light",
              color: "error.contrastText",
              borderRadius: 1,
            }}
          >
            <Typography>{error}</Typography>
          </Box>
        )}

        {resp?.noCoverage && (
          <Box
            sx={{
              p: 2,
              bgcolor: "warning.light",
              color: "warning.contrastText",
              borderRadius: 1,
              textAlign: "center",
            }}
          >
            <Typography>No hay cobertura de datos en el polígono seleccionado.</Typography>
          </Box>
        )}

        {resp?.stats && (
          <Box>
            {/* Header con variable actual */}
            <Box
              sx={{
                mt: 2,
                mb: 2,
                p: 1,
                bgcolor: "primary.main",
                color: "primary.contrastText",
                borderRadius: 1,
              }}
            >
              <Typography variant="overline" sx={{ opacity: 0.9 }}>
                Variable analizada
              </Typography>
              <Typography variant="subtitle1" fontWeight="bold">
                {FIELD_LABELS[currentField] || currentField}
              </Typography>
              {/* {currentUnit && (
                <Chip
                  label={currentUnit}
                  size="small"
                  sx={{
                    mt: 1,
                    bgcolor: "rgba(255,255,255,0.2)",
                    color: "inherit",
                  }}
                />
              )} */}
            </Box>

            {/* Grid de estadísticas principales */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={6} sm={4}>
                <StatCard
                  icon={TrendingDown}
                  label="Mínimo"
                  value={resp.stats.min.toFixed(2)}
                  color="info.main"
                />
              </Grid>
              <Grid item xs={6} sm={4}>
                <StatCard
                  icon={TrendingUp}
                  label="Máximo"
                  value={resp.stats.max.toFixed(2)}
                  color="error.main"
                />
              </Grid>
              <Grid item xs={6} sm={4}>
                <StatCard
                  icon={Functions}
                  label="Media"
                  value={resp.stats.mean.toFixed(2)}
                  color="success.main"
                />
              </Grid>
              <Grid item xs={6} sm={4}>
                <StatCard
                  icon={ShowChart}
                  label="Mediana"
                  value={resp.stats.median.toFixed(2)}
                  color="warning.main"
                />
              </Grid>
              <Grid item xs={6} sm={4}>
                <StatCard
                  icon={Straighten}
                  label="Desv. Estándar"
                  value={resp.stats.std.toFixed(2)}
                  color="secondary.main"
                />
              </Grid>
              {/* <Grid item xs={6} sm={4}>
                <StatCard
                  icon={CheckCircle}
                  label="Píxeles Válidos"
                  value={`${resp.stats.count} (${resp.stats.valid_pct}%)`}
                  color="primary.main"
                />
              </Grid> */}
            </Grid>

            <Divider sx={{ my: 3 }} />

            {/* Sección para agregar más variables (placeholder visual) */}
            <Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                <Typography variant="subtitle1" fontWeight="medium">
                  Comparar con otras variables
                </Typography>
                <Tooltip title="Próximamente: agregar más variables para comparar">
                  <IconButton
                    size="small"
                    disabled
                    sx={{
                      border: "2px dashed",
                      borderColor: "divider",
                    }}
                  >
                    <AddIcon />
                  </IconButton>
                </Tooltip>
              </Box>

              {/* Vista previa de cómo se verían múltiples variables */}
              <Box
                sx={{
                  p: 2,
                  bgcolor: "action.hover",
                  borderRadius: 1,
                  border: "2px dashed",
                  borderColor: "divider",
                }}
              >
                <Typography variant="caption" color="text.secondary" sx={{ mb: 1.5, display: "block" }}>
                  💡 Ejemplo de visualización con múltiples variables:
                </Typography>

                {/* Mockup de tabla comparativa */}
                <Box sx={{ overflowX: "auto" }}>
                  <Box
                    component="table"
                    sx={{
                      width: "100%",
                      borderCollapse: "separate",
                      borderSpacing: "8px 4px",
                      opacity: 0.6,
                    }}
                  >
                    <thead>
                      <tr>
                        <Box component="th" sx={{ textAlign: "left", fontSize: "0.75rem", pb: 1 }}>
                          Estadística
                        </Box>
                        {["DBZH", "ZDR", "RHOHV"].map((field) => (
                          <Box
                            key={field}
                            component="th"
                            sx={{
                              textAlign: "center",
                              fontSize: "0.75rem",
                              pb: 1,
                              px: 1,
                            }}
                          >
                            <Chip
                              label={field}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: "0.7rem", height: "20px" }}
                            />
                          </Box>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: "Mín", values: ["-12.5", "-2.1", "0.85"] },
                        { label: "Máx", values: ["58.3", "8.2", "0.99"] },
                        { label: "Media", values: ["32.4", "1.8", "0.95"] },
                        { label: "Mediana", values: ["34.1", "1.5", "0.96"] },
                      ].map((row, idx) => (
                        <tr key={idx}>
                          <Box component="td" sx={{ fontSize: "0.7rem", fontWeight: "bold" }}>
                            {row.label}
                          </Box>
                          {row.values.map((val, i) => (
                            <Box
                              key={i}
                              component="td"
                              sx={{
                                textAlign: "center",
                                fontSize: "0.75rem",
                                bgcolor: "background.paper",
                                borderRadius: 0.5,
                                px: 1,
                                py: 0.5,
                              }}
                            >
                              {val}
                            </Box>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </Box>
                </Box>

                <Typography variant="caption" color="text.secondary" sx={{ mt: 1.5, display: "block", fontStyle: "italic" }}>
                  Al agregar variables, las estadísticas se mostrarían en columnas para fácil comparación
                </Typography>
              </Box>
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: "1px solid", borderColor: "divider" }}>
        <Button onClick={onClose} variant="contained">
          Cerrar
        </Button>
      </DialogActions>
    </Dialog>
  );
}
