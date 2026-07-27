import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getCacheStats, clearCache, setAuthToken } from "../api/backend";
import { useAuth } from "../contexts/AuthContext";
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
  CircularProgress,
  Divider,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  ThemeProvider,
  createTheme,
  Tooltip,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import RefreshIcon from "@mui/icons-material/Refresh";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AdminPanelSettingsIcon from "@mui/icons-material/AdminPanelSettings";

// ── Theme (same institutional palette) ────────────────────────────────────────

const cacheTheme = createTheme({
  palette: {
    primary: { main: "#1E3A5F", light: "#2563EB", dark: "#152C4A" },
    error: { main: "#DC2626" },
    success: { main: "#059669" },
    warning: { main: "#D97706" },
    background: { default: "#F8FAFC", paper: "#FFFFFF" },
    text: { primary: "#0F172A", secondary: "#64748B", disabled: "#94A3B8" },
    divider: "#E2E8F0",
    action: { hover: "rgba(30,58,95,0.04)" },
  },
  typography: {
    fontFamily: "'Fira Sans', 'Inter', system-ui, sans-serif",
    h6: { fontWeight: 600, letterSpacing: "-0.01em" },
    subtitle2: {
      fontWeight: 600,
      fontSize: "0.75rem",
      textTransform: "uppercase",
      letterSpacing: "0.07em",
    },
    button: { textTransform: "none", fontWeight: 500 },
    caption: { fontFamily: "'Fira Code', monospace", fontSize: "0.72rem" },
    body2: { fontSize: "0.85rem" },
  },
  shape: { borderRadius: 6 },
  shadows: [
    "none",
    "0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)",
    "0 2px 6px rgba(15,23,42,0.07)",
    ...Array(22).fill("0 4px 12px rgba(15,23,42,0.09)"),
  ],
  components: {
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: "none" },
        outlined: { borderColor: "#E2E8F0" },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { fontFamily: "'Fira Sans', sans-serif", borderRadius: 6 },
        contained: {
          boxShadow: "none",
          "&:hover": { boxShadow: "0 2px 6px rgba(30,58,95,0.2)" },
        },
        outlined: { borderColor: "#CBD5E1" },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          fontFamily: "'Fira Sans', sans-serif",
          fontSize: "13px",
          borderRadius: 6,
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          backgroundColor: "#F8FAFC",
          color: "#64748B",
          fontFamily: "'Fira Code', monospace",
          fontSize: "11px",
          fontWeight: 500,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          borderBottom: "1px solid #E2E8F0",
          padding: "9px 14px",
        },
        body: {
          borderBottom: "1px solid #F1F5F9",
          color: "#0F172A",
          padding: "8px 14px",
          fontSize: "0.84rem",
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          "&:hover td": { backgroundColor: "rgba(30,58,95,0.025)" },
          "&:last-child td": { borderBottom: "none" },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { borderRadius: 4, height: 6, backgroundColor: "#E2E8F0" },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontFamily: "'Fira Code', monospace",
          fontSize: "11px",
          height: 22,
          borderRadius: 4,
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 10,
          boxShadow: "0 8px 32px rgba(15,23,42,0.14)",
        },
      },
    },
  },
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function usagePct(used, max) {
  if (!max) return 0;
  return Math.min(100, (used / max) * 100);
}

function progressColor(pct) {
  if (pct < 55) return "success";
  if (pct < 80) return "warning";
  return "error";
}

function fmtSize(mb, bytes) {
  if (mb >= 0.01) return `${mb.toFixed(2)} MB`;
  return `${(bytes / 1024).toFixed(2)} KB`;
}

function fmtModified(ts) {
  return new Date(ts * 1000).toLocaleString("es-AR", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function CacheCard({
  title,
  subtitle,
  entries,
  usedMb,
  maxMb,
  extraRow,
  onClear,
}) {
  const pct = usagePct(usedMb, maxMb);
  const col = progressColor(pct);

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2.5,
        borderRadius: 2,
        display: "flex",
        flexDirection: "column",
        gap: 1.5,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
        }}
      >
        <Box>
          <Typography variant="subtitle2" color="text.secondary">
            {subtitle}
          </Typography>
          <Typography variant="h6" color="text.primary" sx={{ mt: 0.3 }}>
            {title}
          </Typography>
        </Box>
        {onClear && (
          <Tooltip title="Limpiar este cache">
            <Button
              size="small"
              variant="outlined"
              color="error"
              startIcon={<DeleteOutlineIcon sx={{ fontSize: 14 }} />}
              onClick={onClear}
              sx={{
                fontSize: 12,
                py: 0.4,
                borderColor: "rgba(220,38,38,0.3)",
                color: "#DC2626",
              }}
            >
              Limpiar
            </Button>
          </Tooltip>
        )}
      </Box>

      <Divider />

      {/* Stats */}
      <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Entradas
          </Typography>
          <Typography
            sx={{
              fontFamily: "'Fira Code', monospace",
              fontSize: 13,
              fontWeight: 500,
              color: "text.primary",
            }}
          >
            {entries}
          </Typography>
        </Box>

        {maxMb !== undefined && (
          <>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                Uso actual
              </Typography>
              <Typography
                sx={{
                  fontFamily: "'Fira Code', monospace",
                  fontSize: 13,
                  fontWeight: 500,
                  color: "text.primary",
                }}
              >
                {usedMb.toFixed(2)} MB
              </Typography>
            </Box>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                Límite
              </Typography>
              <Typography
                sx={{
                  fontFamily: "'Fira Code', monospace",
                  fontSize: 13,
                  color: "text.secondary",
                }}
              >
                {maxMb} MB
              </Typography>
            </Box>

            {/* Progress bar */}
            <Box sx={{ mt: 0.5 }}>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  mb: 0.6,
                }}
              >
                <Typography
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: 11,
                    color: "text.disabled",
                  }}
                >
                  {pct.toFixed(1)}% utilizado
                </Typography>
                <Chip
                  label={pct < 55 ? "Normal" : pct < 80 ? "Moderado" : "Alto"}
                  size="small"
                  color={col}
                  sx={{ height: 18, fontSize: 10 }}
                />
              </Box>
              <LinearProgress
                variant="determinate"
                value={pct}
                color={col}
                sx={{ "& .MuiLinearProgress-bar": { borderRadius: 4 } }}
              />
            </Box>
          </>
        )}

        {extraRow && (
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="body2" color="text.secondary">
              {extraRow.label}
            </Typography>
            <Typography
              sx={{
                fontFamily: "'Fira Code', monospace",
                fontSize: 12,
                color: "text.secondary",
              }}
            >
              {extraRow.value}
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function CacheStats() {
  const navigate = useNavigate();
  const { token } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null); // { msg, severity }
  const [polling, setPolling] = useState(false);
  const [confirm, setConfirm] = useState(null); // { type, label }

  // Sync Axios Bearer token when landing directly on /cache
  useEffect(() => {
    setAuthToken(token);
  }, [token]);

  const loadStats = async () => {
    try {
      setLoading(true);
      setStats(await getCacheStats());
      setError(null);
    } catch (err) {
      setError("Error al cargar estadísticas: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh every 10 s, paused when tab hidden
  useEffect(() => {
    let interval = null;
    const start = () => {
      if (!interval) {
        loadStats();
        interval = setInterval(loadStats, 10_000);
        setPolling(true);
      }
    };
    const stop = () => {
      if (interval) {
        clearInterval(interval);
        interval = null;
        setPolling(false);
      }
    };
    const onVisibility = () => (document.hidden ? stop() : start());
    start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const showToast = (msg, severity = "success") => {
    setToast({ msg, severity });
    setTimeout(() => setToast(null), 5000);
  };

  const CLEAR_LABELS = {
    all: "¿Limpiar TODOS los caches (RAM + Disco)?",
    grid2d: "¿Limpiar cache Grid 2D (RAM)?",
    w_operator_ram: "¿Limpiar cache W Operator (RAM)?",
    w_operator_disk: "¿Limpiar cache W Operator (Disco, archivos .npz)?",
  };

  const handleClear = async () => {
    const { type } = confirm;
    setConfirm(null);
    try {
      const res = await clearCache(type);
      showToast(
        `${res.cleared} entradas eliminadas del cache "${type}".`,
        "success",
      );
      loadStats();
    } catch (err) {
      showToast("Error al limpiar: " + err.message, "error");
    }
  };

  // ── Loading state ──
  if (loading && !stats) {
    return (
      <ThemeProvider theme={cacheTheme}>
        <Box
          sx={{
            minHeight: "100vh",
            bgcolor: "background.default",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <CircularProgress size={28} thickness={4} sx={{ color: "#1E3A5F" }} />
        </Box>
      </ThemeProvider>
    );
  }

  // ── Error state ──
  if (error && !stats) {
    return (
      <ThemeProvider theme={cacheTheme}>
        <Box
          sx={{
            minHeight: "100vh",
            bgcolor: "background.default",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            p: 3,
          }}
        >
          <Box sx={{ maxWidth: 440, width: "100%" }}>
            <Alert
              severity="error"
              action={
                <Button size="small" onClick={loadStats} color="error">
                  Reintentar
                </Button>
              }
            >
              {error}
            </Alert>
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

  const grid2d = stats?.grid2d_cache || {
    entries: 0,
    size_mb: 0,
    max_size_mb: 100,
  };
  const wRam = stats?.w_operator_cache_ram || {
    entries: 0,
    size_mb: 0,
    max_size_mb: 300,
  };
  const wDisk = stats?.w_operator_cache_disk || { files: 0, size_mb: 0 };

  return (
    <ThemeProvider theme={cacheTheme}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');`}</style>

      {/* ── Confirm dialog ── */}
      <Dialog
        open={!!confirm}
        onClose={() => setConfirm(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle
          sx={{
            fontFamily: "'Fira Sans', sans-serif",
            fontSize: 15,
            fontWeight: 600,
          }}
        >
          Confirmar acción
        </DialogTitle>
        <DialogContent>
          <DialogContentText
            sx={{ fontFamily: "'Fira Sans', sans-serif", fontSize: 14 }}
          >
            {confirm?.label}
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
          <Button size="small" onClick={() => setConfirm(null)}>
            Cancelar
          </Button>
          <Button
            size="small"
            variant="contained"
            color="error"
            onClick={handleClear}
            sx={{ boxShadow: "none" }}
          >
            Confirmar
          </Button>
        </DialogActions>
      </Dialog>

      <Box
        sx={{
          minHeight: "100vh",
          bgcolor: "background.default",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          py: { xs: 2, md: 4 },
          px: { xs: 2, md: 3 },
        }}
      >
        <Paper
          elevation={1}
          sx={{
            width: "100%",
            maxWidth: 960,
            borderRadius: 2,
            overflow: "hidden",
            border: "1px solid",
            borderColor: "divider",
          }}
        >
          {/* Navy accent bar */}
          <Box sx={{ height: 3, bgcolor: "#1E3A5F" }} />

          <Box sx={{ p: { xs: 2.5, md: 3 } }}>
            {/* ── Header ── */}
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}
            >
              <Tooltip title="Volver al mapa">
                <Button
                  size="small"
                  startIcon={<ArrowBackIcon sx={{ fontSize: 16 }} />}
                  onClick={() => navigate("/")}
                  sx={{ color: "text.secondary", minWidth: 0, px: 1 }}
                ></Button>
              </Tooltip>

              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="subtitle2"
                  color="text.secondary"
                  sx={{ mb: 0.1 }}
                >
                  Sistema
                </Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                  <Typography
                    variant="h6"
                    color="text.primary"
                    sx={{ lineHeight: 1.2 }}
                  >
                    Estadísticas de Cache
                  </Typography>
                  {polling && (
                    <Chip
                      label="En vivo"
                      size="small"
                      color="success"
                      sx={{
                        height: 20,
                        fontSize: 10,
                        fontFamily: "'Fira Code', monospace",
                      }}
                    />
                  )}
                </Box>
              </Box>

              <Button
                size="small"
                variant="outlined"
                startIcon={<AdminPanelSettingsIcon sx={{ fontSize: 15 }} />}
                onClick={() => navigate("/admin")}
                sx={{
                  fontSize: 13,
                  borderColor: "#CBD5E1",
                  color: "text.secondary",
                  display: { xs: "none", sm: "flex" },
                }}
              >
                Admin
              </Button>

              <Button
                size="small"
                variant="outlined"
                startIcon={<RefreshIcon sx={{ fontSize: 15 }} />}
                onClick={loadStats}
                sx={{
                  fontSize: 13,
                  borderColor: "#CBD5E1",
                  color: "text.secondary",
                }}
              >
                Actualizar
              </Button>
            </Box>

            <Divider sx={{ mb: 2.5 }} />

            {/* ── Toast ── */}
            {toast && (
              <Alert
                severity={toast.severity}
                sx={{ mb: 2.5 }}
                onClose={() => setToast(null)}
              >
                {toast.msg}
              </Alert>
            )}

            {/* ── Cache stat cards ── */}
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: {
                  xs: "1fr",
                  sm: "1fr 1fr",
                  md: "1fr 1fr 1fr",
                },
                gap: 2,
                mb: 3,
              }}
            >
              <CacheCard
                title="Grid 2D"
                subtitle="Cache RAM"
                entries={grid2d.entries}
                usedMb={grid2d.size_mb}
                maxMb={grid2d.max_size_mb}
                onClear={() =>
                  setConfirm({ type: "grid2d", label: CLEAR_LABELS.grid2d })
                }
              />
              <CacheCard
                title="W Operator"
                subtitle="Cache RAM"
                entries={wRam.entries}
                usedMb={wRam.size_mb}
                maxMb={wRam.max_size_mb}
                onClear={() =>
                  setConfirm({
                    type: "w_operator_ram",
                    label: CLEAR_LABELS.w_operator_ram,
                  })
                }
              />
              <CacheCard
                title="W Operator"
                subtitle="Cache Disco"
                entries={wDisk.files}
                usedMb={wDisk.size_mb}
                extraRow={{ label: "Ruta", value: "storage/cache/" }}
                onClear={() =>
                  setConfirm({
                    type: "w_operator_disk",
                    label: CLEAR_LABELS.w_operator_disk,
                  })
                }
              />
            </Box>

            {/* ── Danger zone ── */}
            <Paper
              variant="outlined"
              sx={{
                p: 2,
                borderRadius: 1.5,
                borderColor: "rgba(220,38,38,0.25)",
                bgcolor: "rgba(220,38,38,0.02)",
                mb: 3,
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 2,
                  flexWrap: "wrap",
                }}
              >
                <Box sx={{ flex: 1, minWidth: 200 }}>
                  <Typography
                    variant="subtitle2"
                    color="error"
                    sx={{ mb: 0.3 }}
                  >
                    Limpiar todo
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Elimina todos los caches de RAM y Disco simultáneamente.
                  </Typography>
                </Box>
                <Button
                  size="small"
                  variant="contained"
                  color="error"
                  startIcon={<DeleteOutlineIcon sx={{ fontSize: 15 }} />}
                  onClick={() =>
                    setConfirm({ type: "all", label: CLEAR_LABELS.all })
                  }
                  sx={{ flexShrink: 0, boxShadow: "none" }}
                >
                  Limpiar todo
                </Button>
              </Box>
            </Paper>

            {/* ── Files table ── */}
            <Box>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1.5,
                  mb: 1.5,
                }}
              >
                <Typography variant="subtitle2" color="text.secondary">
                  Archivos en cache de disco
                </Typography>
                <Chip
                  label={stats?.cache_files?.length ?? 0}
                  size="small"
                  sx={{
                    bgcolor: "#F1F5F9",
                    color: "text.secondary",
                    height: 20,
                    fontFamily: "'Fira Code', monospace",
                    fontSize: 11,
                  }}
                />
              </Box>

              {stats?.cache_files?.length > 0 ? (
                <TableContainer
                  component={Paper}
                  variant="outlined"
                  sx={{ borderRadius: 1.5 }}
                >
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Archivo</TableCell>
                        <TableCell align="right">Tamaño</TableCell>
                        <TableCell align="right">Modificado</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {stats.cache_files.map((f, i) => (
                        <TableRow key={i}>
                          <TableCell
                            sx={{
                              fontFamily: "'Fira Code', monospace",
                              fontSize: "11px",
                              maxWidth: 400,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                            title={f.name}
                          >
                            {f.name}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              fontFamily: "'Fira Code', monospace",
                              fontSize: "11px",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {fmtSize(f.size_mb, f.size_bytes)}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              fontFamily: "'Fira Code', monospace",
                              fontSize: "11px",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {fmtModified(f.modified_at)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Paper
                  variant="outlined"
                  sx={{ p: 4, borderRadius: 1.5, textAlign: "center" }}
                >
                  <Typography
                    variant="body2"
                    color="text.disabled"
                    sx={{ fontStyle: "italic" }}
                  >
                    No hay archivos de cache en disco
                  </Typography>
                </Paper>
              )}
            </Box>
          </Box>
        </Paper>

        <Typography variant="caption" color="text.disabled" sx={{ mt: 2.5 }}>
          Actualización automática cada 10 s · SIVAR v2.0
        </Typography>
      </Box>
    </ThemeProvider>
  );
}
