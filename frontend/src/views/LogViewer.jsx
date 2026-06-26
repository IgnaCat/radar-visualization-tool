import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { fetchLogs } from "../api/admin";
import { setAuthToken } from "../api/backend";
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Chip,
  ThemeProvider,
  createTheme,
  Switch,
  FormControlLabel,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import RefreshIcon from "@mui/icons-material/Refresh";

const logTheme = createTheme({
  palette: {
    primary: { main: "#1E3A5F", light: "#2563EB", dark: "#152C4A" },
    error: { main: "#DC2626" },
    success: { main: "#059669" },
    warning: { main: "#D97706" },
    background: { default: "#F8FAFC", paper: "#FFFFFF" },
    text: { primary: "#0F172A", secondary: "#64748B" },
    divider: "#E2E8F0",
  },
  typography: {
    fontFamily: "'Fira Sans', 'Inter', system-ui, sans-serif",
    h6: { fontWeight: 600, letterSpacing: "-0.01em" },
    button: { textTransform: "none", fontWeight: 500 },
  },
  shape: { borderRadius: 6 },
  shadows: [
    "none",
    "0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)",
    "0 2px 6px rgba(15,23,42,0.07)",
    ...Array(22).fill("0 4px 12px rgba(15,23,42,0.09)"),
  ],
});

const LEVEL_COLORS = {
  INFO: "#2563EB",
  WARNING: "#D97706",
  ERROR: "#DC2626",
  CRITICAL: "#7C2D12",
  DEBUG: "#64748B",
};

function colorizeLine(line) {
  for (const [level, color] of Object.entries(LEVEL_COLORS)) {
    if (line.includes(level)) {
      return color;
    }
  }
  return "#334155";
}

export default function LogViewer() {
  const navigate = useNavigate();
  const { token } = useAuth();
  const [lines, setLines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [total, setTotal] = useState(0);
  const [truncated, setTruncated] = useState(false);

  // Filters
  const [maxLines, setMaxLines] = useState(300);
  const [level, setLevel] = useState("");
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(false);

  const logEndRef = useRef(null);
  const intervalRef = useRef(null);

  // Sync Axios Bearer token when landing directly on /logs
  useEffect(() => {
    setAuthToken(token);
  }, [token]);

  const load = useCallback(async () => {
    try {
      setError(null);
      const data = await fetchLogs(token, {
        lines: maxLines,
        level: level || undefined,
        search: search || undefined,
      });
      setLines(data.lines || []);
      setTotal(data.total || 0);
      setTruncated(data.truncated || false);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [token, maxLines, level, search]);

  useEffect(() => {
    load();
  }, [load]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(load, 3000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, load]);

  // Scroll to bottom when lines change
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollTop = logEndRef.current.scrollHeight;
    }
  }, [lines]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setSearch(searchInput);
  };

  return (
    <ThemeProvider theme={logTheme}>
      <Box
        sx={{
          minHeight: "100vh",
          bgcolor: "background.default",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <Paper
          elevation={1}
          sx={{
            px: 3,
            py: 1.5,
            display: "flex",
            alignItems: "center",
            gap: 2,
            borderRadius: 0,
          }}
        >
          <Button
            size="small"
            startIcon={<ArrowBackIcon fontSize="small" />}
            onClick={() => navigate("/admin")}
            sx={{ color: "text.secondary", minWidth: "auto" }}
          >
            Admin
          </Button>

          <Typography variant="h6" sx={{ flex: 1 }}>
            Logs del Backend
          </Typography>

          <Chip
            label={`${total} líneas totales`}
            size="small"
            sx={{ bgcolor: "#F1F5F9", fontSize: 12 }}
          />
          {truncated && (
            <Chip
              label="truncado"
              size="small"
              color="warning"
              variant="outlined"
              sx={{ fontSize: 12 }}
            />
          )}

          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label={
              <Typography variant="body2" color="text.secondary">
                Auto-refresh
              </Typography>
            }
          />

          <Button
            size="small"
            variant="outlined"
            startIcon={<RefreshIcon fontSize="small" />}
            onClick={load}
            disabled={loading}
          >
            Refresh
          </Button>
        </Paper>

        {/* Filters */}
        <Box sx={{ px: 3, py: 1.5, display: "flex", gap: 2, alignItems: "center" }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Nivel</InputLabel>
            <Select
              value={level}
              label="Nivel"
              onChange={(e) => setLevel(e.target.value)}
            >
              <MenuItem value="">Todos</MenuItem>
              <MenuItem value="DEBUG">DEBUG</MenuItem>
              <MenuItem value="INFO">INFO</MenuItem>
              <MenuItem value="WARNING">WARNING</MenuItem>
              <MenuItem value="ERROR">ERROR</MenuItem>
              <MenuItem value="CRITICAL">CRITICAL</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel>Líneas</InputLabel>
            <Select
              value={maxLines}
              label="Líneas"
              onChange={(e) => setMaxLines(e.target.value)}
            >
              <MenuItem value={100}>100</MenuItem>
              <MenuItem value={300}>300</MenuItem>
              <MenuItem value={500}>500</MenuItem>
              <MenuItem value={1000}>1000</MenuItem>
              <MenuItem value={5000}>5000</MenuItem>
            </Select>
          </FormControl>

          <form onSubmit={handleSearchSubmit} style={{ display: "flex", gap: 8, flex: 1 }}>
            <TextField
              size="small"
              placeholder="Buscar en logs..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              sx={{ flex: 1, maxWidth: 400 }}
            />
            <Button type="submit" size="small" variant="contained">
              Buscar
            </Button>
            {search && (
              <Button
                size="small"
                onClick={() => {
                  setSearch("");
                  setSearchInput("");
                }}
              >
                Limpiar
              </Button>
            )}
          </form>
        </Box>

        {/* Error */}
        {error && (
          <Box sx={{ px: 3 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}

        {/* Log output */}
        <Box sx={{ flex: 1, px: 3, pb: 3, minHeight: 0 }}>
          <Paper
            ref={logEndRef}
            variant="outlined"
            sx={{
              height: "calc(100vh - 200px)",
              overflow: "auto",
              bgcolor: "#0F172A",
              p: 2,
              fontFamily: "'Fira Code', 'Consolas', monospace",
              fontSize: "12.5px",
              lineHeight: 1.65,
              whiteSpace: "pre",
              position: "relative",
            }}
          >
            {loading && lines.length === 0 ? (
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  height: "100%",
                }}
              >
                <CircularProgress size={28} sx={{ color: "#64748B" }} />
              </Box>
            ) : lines.length === 0 ? (
              <Typography sx={{ color: "#64748B", fontFamily: "inherit" }}>
                No hay logs disponibles.
              </Typography>
            ) : (
              lines.map((line, i) => (
                <div key={i} style={{ color: colorizeLine(line) }}>
                  {line}
                </div>
              ))
            )}
          </Paper>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
