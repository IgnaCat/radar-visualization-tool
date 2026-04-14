import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { setAuthToken } from "../api/backend";
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  ThemeProvider,
  createTheme,
  Divider,
} from "@mui/material";
import logoSrc from "../assets/lrsr_logo.png";

// ── Shared institutional theme ────────────────────────────────────────────────

export const institutionalTheme = createTheme({
  palette: {
    primary: { main: "#1E3A5F", light: "#2563EB", dark: "#152C4A" },
    error: { main: "#DC2626" },
    success: { main: "#059669" },
    warning: { main: "#D97706" },
    background: { default: "#F8FAFC", paper: "#FFFFFF" },
    text: { primary: "#0F172A", secondary: "#64748B", disabled: "#94A3B8" },
    divider: "#E2E8F0",
    action: { hover: "rgba(30,58,95,0.05)", selected: "rgba(30,58,95,0.08)" },
  },
  typography: {
    fontFamily: "'Fira Sans', 'Inter', system-ui, sans-serif",
    h4: { fontWeight: 700, letterSpacing: "-0.02em", lineHeight: 1.25 },
    h5: { fontWeight: 600, letterSpacing: "-0.01em", lineHeight: 1.3 },
    h6: { fontWeight: 600, letterSpacing: "-0.01em", lineHeight: 1.4 },
    subtitle1: { fontWeight: 500 },
    subtitle2: {
      fontWeight: 600,
      fontSize: "0.8rem",
      textTransform: "uppercase",
      letterSpacing: "0.06em",
    },
    button: { textTransform: "none", fontWeight: 500, fontSize: "0.9rem" },
    caption: { fontFamily: "'Fira Code', monospace", fontSize: "0.72rem" },
  },
  shape: { borderRadius: 6 },
  shadows: [
    "none",
    "0 1px 2px rgba(15,23,42,0.06)",
    "0 1px 4px rgba(15,23,42,0.08)",
    "0 2px 8px rgba(15,23,42,0.1)",
    ...Array(21).fill("0 4px 16px rgba(15,23,42,0.12)"),
  ],
  components: {
    MuiTextField: { defaultProps: { variant: "outlined", size: "small" } },
    MuiButton: {
      styleOverrides: {
        root: { fontFamily: "'Fira Sans', sans-serif" },
        contained: {
          boxShadow: "none",
          "&:hover": { boxShadow: "0 2px 8px rgba(30,58,95,0.22)" },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          backgroundColor: "#FAFBFD",
          transition: "border-color 0.15s, box-shadow 0.15s",
          "&:hover .MuiOutlinedInput-notchedOutline": {
            borderColor: "#94A3B8",
          },
          "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
            borderColor: "#1E3A5F",
            borderWidth: "2px",
          },
        },
        notchedOutline: { borderColor: "#E2E8F0" },
      },
    },
    MuiInputLabel: {
      styleOverrides: {
        root: { fontFamily: "'Fira Sans', sans-serif", fontSize: "14px" },
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
  },
});

// ── Radar rings decorative SVG ────────────────────────────────────────────────

function RadarRings({ size = 96, opacity = 0.22 }) {
  const s = size;
  const c = s / 2;
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} fill="none">
      {[0.9, 0.68, 0.46, 0.24].map((r, i) => (
        <circle
          key={i}
          cx={c}
          cy={c}
          r={c * r - 0.75}
          stroke={`rgba(255,255,255,${opacity + i * 0.04})`}
          strokeWidth="1.25"
        />
      ))}
      <circle
        cx={c}
        cy={c}
        r="3"
        fill={`rgba(255,255,255,${opacity + 0.55})`}
      />
      <line
        x1={c}
        y1={c}
        x2={c + c * 0.78}
        y2={c - c * 0.38}
        stroke={`rgba(255,255,255,${opacity + 0.35})`}
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <line
        x1={c}
        y1={4}
        x2={c}
        y2={s - 4}
        stroke={`rgba(255,255,255,${opacity * 0.5})`}
        strokeWidth="1"
      />
      <line
        x1={4}
        y1={c}
        x2={s - 4}
        y2={c}
        stroke={`rgba(255,255,255,${opacity * 0.5})`}
        strokeWidth="1"
      />
    </svg>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await login(username, password);
      setAuthToken(data.access_token);
      navigate(from, { replace: true });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Credenciales incorrectas. Verificá usuario y contraseña.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={institutionalTheme}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');`}</style>

      <Box sx={{ display: "flex", minHeight: "100vh" }}>
        {/* ── Left panel — institutional branding ── */}
        <Box
          sx={{
            display: { xs: "none", md: "flex" },
            flexDirection: "column",
            justifyContent: "space-between",
            width: "42%",
            flexShrink: 0,
            bgcolor: "#1E3A5F",
            p: 6,
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Dot grid texture */}
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              opacity: 0.045,
              backgroundImage:
                "radial-gradient(circle, rgba(255,255,255,1) 1px, transparent 1px)",
              backgroundSize: "26px 26px",
              pointerEvents: "none",
            }}
          />

          {/* Top — logo + name */}
          <Box sx={{ position: "relative", zIndex: 1 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 7 }}>
              <Box
                component="img"
                src={logoSrc}
                alt="Logo"
                sx={{ height: 38, width: "auto", objectFit: "contain" }}
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />
              <Box>
                <Typography
                  sx={{
                    fontFamily: "'Fira Sans', sans-serif",
                    fontSize: 18,
                    fontWeight: 700,
                    color: "#FFFFFF",
                    letterSpacing: "0.04em",
                    lineHeight: 1,
                  }}
                >
                  RADARG
                </Typography>
                <Typography
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: 10,
                    color: "rgba(255,255,255,0.5)",
                    letterSpacing: "0.06em",
                    mt: 0.4,
                  }}
                >
                  FAMAF · UNC
                </Typography>
              </Box>
            </Box>

            {/* Radar icon */}
            <Box sx={{ mb: 4 }}>
              <RadarRings size={88} opacity={0.18} />
            </Box>

            <Typography
              variant="h5"
              sx={{ color: "#FFFFFF", mb: 2, fontWeight: 600 }}
            >
              Sistema de Visualización de Radar Meteorológico
            </Typography>

            <Typography
              sx={{
                color: "rgba(255,255,255,0.6)",
                fontSize: 14,
                lineHeight: 1.75,
              }}
            >
              Procesamiento y visualización interactiva de datos radar. Soporta
              productos PPI, CAPPI, COLMAX y Pseudo-RHI desde archivos NetCDF y
              BUFR.
            </Typography>
          </Box>

          {/* Bottom — institution */}
          <Box sx={{ position: "relative", zIndex: 1 }}>
            <Divider sx={{ borderColor: "rgba(255,255,255,0.12)", mb: 2 }} />
            <Typography
              sx={{
                color: "rgba(255,255,255,0.35)",
                fontSize: 12,
                lineHeight: 1.6,
              }}
            >
              Proyecto de Tesis · FAMAF
              <br />
              Universidad Nacional de Córdoba
            </Typography>
          </Box>
        </Box>

        {/* ── Right panel — login form ── */}
        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "#FFFFFF",
            p: { xs: 3, sm: 5, md: 7 },
          }}
        >
          {/* Mobile-only compact brand bar */}
          <Box
            sx={{
              display: { xs: "flex", md: "none" },
              alignItems: "center",
              gap: 1.5,
              mb: 4,
              px: 2,
              py: 1.5,
              bgcolor: "#1E3A5F",
              borderRadius: 1.5,
              alignSelf: "stretch",
              justifyContent: "center",
            }}
          >
            <Box
              component="img"
              src={logoSrc}
              alt="Logo"
              sx={{ height: 28 }}
              onError={(e) => {
                e.target.style.display = "none";
              }}
            />
            <Typography
              sx={{
                fontFamily: "'Fira Sans', sans-serif",
                fontWeight: 700,
                fontSize: 15,
                color: "#FFFFFF",
                letterSpacing: "0.04em",
              }}
            >
              RADARG
            </Typography>
          </Box>

          {/* Form card */}
          <Box sx={{ width: "100%", maxWidth: 360 }}>
            <Typography variant="h5" color="text.primary" sx={{ mb: 0.75 }}>
              Acceso al sistema
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 3.5, lineHeight: 1.65 }}
            >
              Ingresá tus credenciales institucionales para continuar.
            </Typography>

            {error && (
              <Alert
                severity="error"
                sx={{ mb: 2.5 }}
                onClose={() => setError(null)}
              >
                {error}
              </Alert>
            )}

            <Box
              component="form"
              onSubmit={handleSubmit}
              sx={{ display: "flex", flexDirection: "column", gap: 2 }}
            >
              <TextField
                label="Usuario"
                fullWidth
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
                required
                autoComplete="username"
                inputProps={{ "aria-label": "Usuario" }}
              />
              <TextField
                label="Contraseña"
                type="password"
                fullWidth
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                inputProps={{ "aria-label": "Contraseña" }}
              />
              <Button
                type="submit"
                variant="contained"
                fullWidth
                size="large"
                disabled={loading}
                sx={{
                  mt: 0.5,
                  height: 44,
                  fontSize: 15,
                  fontWeight: 600,
                  bgcolor: "#1E3A5F",
                  color: "#FFFFFF",
                  "&:hover": { bgcolor: "#152C4A" },
                  "&.Mui-disabled": {
                    bgcolor: "rgba(30,58,95,0.35)",
                    color: "rgba(255,255,255,0.6)",
                  },
                }}
              >
                {loading ? (
                  <CircularProgress
                    size={20}
                    sx={{ color: "rgba(255,255,255,0.8)" }}
                  />
                ) : (
                  "Ingresar"
                )}
              </Button>
            </Box>

            <Typography
              variant="caption"
              color="text.disabled"
              sx={{
                display: "block",
                textAlign: "center",
                mt: 4,
                lineHeight: 1.6,
              }}
            >
              v2.0 · Acceso restringido a usuarios autorizados
            </Typography>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
