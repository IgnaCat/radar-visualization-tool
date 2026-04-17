import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import {
  fetchUsers,
  createUser,
  updateUser,
  fetchAccessLogs,
  fetchActiveSessions,
  forceCleanupUser,
} from "../api/admin";
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  CircularProgress,
  Tooltip,
  Switch,
  FormControlLabel,
  ThemeProvider,
  createTheme,
  Divider,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import EditIcon from "@mui/icons-material/Edit";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import RefreshIcon from "@mui/icons-material/Refresh";
import StorageIcon from "@mui/icons-material/Storage";
import logoSrc from "../assets/lrsr_logo.png";

// ── Institutional theme ───────────────────────────────────────────────────────

const adminTheme = createTheme({
  palette: {
    primary: { main: "#1E3A5F", light: "#2563EB", dark: "#152C4A" },
    error: { main: "#DC2626" },
    success: { main: "#059669" },
    warning: { main: "#D97706" },
    background: { default: "#F8FAFC", paper: "#FFFFFF" },
    text: { primary: "#0F172A", secondary: "#64748B", disabled: "#94A3B8" },
    divider: "#E2E8F0",
    action: { hover: "rgba(30,58,95,0.04)", selected: "rgba(30,58,95,0.08)" },
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
    "0 4px 12px rgba(15,23,42,0.09)",
    ...Array(21).fill("0 8px 24px rgba(15,23,42,0.1)"),
  ],
  components: {
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: "none" },
        outlined: { borderColor: "#E2E8F0" },
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
          padding: "10px 14px",
          whiteSpace: "nowrap",
        },
        body: {
          borderBottom: "1px solid #F1F5F9",
          color: "#0F172A",
          padding: "9px 14px",
          fontSize: "0.855rem",
        },
        stickyHeader: {
          backgroundColor: "#F8FAFC",
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          "&:hover td, &:hover th": { backgroundColor: "rgba(30,58,95,0.025)" },
          "&:last-child td": { borderBottom: "none" },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontFamily: "'Fira Sans', sans-serif",
          fontWeight: 500,
          fontSize: "14px",
          textTransform: "none",
          color: "#64748B",
          minHeight: 44,
          padding: "10px 18px",
          "&.Mui-selected": { color: "#1E3A5F", fontWeight: 600 },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: { backgroundColor: "#1E3A5F", height: 2 },
        root: { minHeight: 44 },
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
    MuiTextField: {
      defaultProps: { variant: "outlined", size: "small" },
      styleOverrides: {
        root: {
          "& .MuiInputBase-input": {
            fontFamily: "'Fira Sans', sans-serif",
            fontSize: "14px",
          },
          "& .MuiOutlinedInput-root": {
            "& fieldset": { borderColor: "#E2E8F0" },
            "&:hover fieldset": { borderColor: "#94A3B8" },
            "&.Mui-focused fieldset": { borderColor: "#1E3A5F" },
          },
          "& .MuiInputLabel-root": {
            fontFamily: "'Fira Sans', sans-serif",
            fontSize: "14px",
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: { fontFamily: "'Fira Sans', sans-serif", fontSize: "14px" },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: { boxShadow: "0 8px 32px rgba(15,23,42,0.14)" },
      },
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          fontFamily: "'Fira Sans', sans-serif",
          fontWeight: 600,
          fontSize: "16px",
          borderBottom: "1px solid #E2E8F0",
          pb: 1.5,
        },
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
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          "&:hover": { backgroundColor: "rgba(30,58,95,0.06)" },
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        switchBase: { "&.Mui-checked": { color: "#1E3A5F" } },
        track: {
          ".Mui-checked.Mui-checked + &": { backgroundColor: "#1E3A5F" },
        },
      },
    },
    MuiFormControlLabel: {
      styleOverrides: {
        label: { fontFamily: "'Fira Sans', sans-serif", fontSize: "14px" },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: { fontFamily: "'Fira Sans', sans-serif", fontSize: "14px" },
      },
    },
  },
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(isoStr) {
  if (!isoStr) return "—";
  const date = new Date(isoStr);
  // Restamos 3 horas para ajustar de UTC a hora local (Argentina)
  date.setHours(date.getHours() - 3);
  return date.toLocaleString("es-AR", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

// ── Sub-components ────────────────────────────────────────────────────────────

function TabPanel({ value, index, children }) {
  return value === index ? <Box sx={{ pt: 2 }}>{children}</Box> : null;
}

// ── Users tab ─────────────────────────────────────────────────────────────────

function UsersTab({ token, currentUserId }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [form, setForm] = useState({
    username: "",
    password: "",
    role: "user",
  });
  const [formActive, setFormActive] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setUsers(await fetchUsers(token));
    } catch {
      setError("No se pudo cargar la lista de usuarios.");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    load();
  }, [load]);

  const openCreate = () => {
    setEditingUser(null);
    setForm({ username: "", password: "", role: "user" });
    setFormActive(true);
    setFormError(null);
    setDialogOpen(true);
  };

  const openEdit = (u) => {
    setEditingUser(u);
    setForm({ username: u.username, password: "", role: u.role });
    setFormActive(u.is_active);
    setFormError(null);
    setDialogOpen(true);
  };

  const handleSave = async () => {
    setSaving(true);
    setFormError(null);
    try {
      if (editingUser) {
        const patch = { role: form.role, is_active: formActive };
        if (form.password) patch.password = form.password;
        await updateUser(token, editingUser.id, patch);
      } else {
        await createUser(token, {
          username: form.username,
          password: form.password,
          role: form.role,
        });
      }
      setDialogOpen(false);
      await load();
    } catch (err) {
      const detail = err.response?.data?.detail;
      setFormError(typeof detail === "string" ? detail : "Error al guardar.");
    } finally {
      setSaving(false);
    }
  };

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress size={26} thickness={4} />
      </Box>
    );
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <>
      <Box
        sx={{ display: "flex", justifyContent: "flex-end", mb: 1.5, gap: 1 }}
      >
        <Button
          size="small"
          startIcon={<RefreshIcon sx={{ fontSize: 16 }} />}
          onClick={load}
          sx={{ color: "text.secondary" }}
        >
          Actualizar
        </Button>
        <Button
          size="small"
          variant="contained"
          startIcon={<PersonAddIcon sx={{ fontSize: 16 }} />}
          onClick={openCreate}
          sx={{ bgcolor: "#1E3A5F", "&:hover": { bgcolor: "#152C4A" } }}
        >
          Nuevo usuario
        </Button>
      </Box>

      <TableContainer
        component={Paper}
        variant="outlined"
        sx={{ borderRadius: 1.5 }}
      >
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Usuario</TableCell>
              <TableCell>Rol</TableCell>
              <TableCell>Estado</TableCell>
              <TableCell>Creado</TableCell>
              <TableCell align="right">Acciones</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users.map((u) => (
              <TableRow key={u.id}>
                <TableCell sx={{ fontWeight: 500 }}>{u.username}</TableCell>
                <TableCell>
                  <Chip
                    label={u.role}
                    size="small"
                    color={u.role === "admin" ? "primary" : "default"}
                    sx={{
                      bgcolor:
                        u.role === "admin" ? "rgba(30,58,95,0.1)" : "#F1F5F9",
                      color: u.role === "admin" ? "#1E3A5F" : "#64748B",
                    }}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={u.is_active ? "Activo" : "Inactivo"}
                    size="small"
                    sx={{
                      bgcolor: u.is_active
                        ? "rgba(5,150,105,0.1)"
                        : "rgba(220,38,38,0.08)",
                      color: u.is_active ? "#059669" : "#DC2626",
                    }}
                  />
                </TableCell>
                <TableCell
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: "11px",
                    color: "text.secondary",
                  }}
                >
                  {fmtDate(u.created_at)}
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="Editar usuario">
                    <IconButton
                      size="small"
                      onClick={() => openEdit(u)}
                      disabled={u.id === currentUserId && u.role === "admin"}
                    >
                      <EditIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Create / Edit dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="xs"
        fullWidth
        PaperProps={{ sx: { borderRadius: 2 } }}
      >
        <DialogTitle
          sx={{
            fontFamily: "'Fira Sans', sans-serif",
            fontSize: 16,
            fontWeight: 600,
          }}
        >
          {editingUser ? "Editar usuario" : "Nuevo usuario"}
        </DialogTitle>
        <DialogContent
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            pt: "16px !important",
          }}
        >
          {formError && <Alert severity="error">{formError}</Alert>}
          <TextField
            label="Usuario"
            value={form.username}
            onChange={(e) =>
              setForm((f) => ({ ...f, username: e.target.value }))
            }
            disabled={!!editingUser}
            required={!editingUser}
            autoFocus
            fullWidth
          />
          <TextField
            label={
              editingUser
                ? "Nueva contraseña (dejar vacío para no cambiar)"
                : "Contraseña"
            }
            type="password"
            value={form.password}
            onChange={(e) =>
              setForm((f) => ({ ...f, password: e.target.value }))
            }
            required={!editingUser}
            fullWidth
          />
          <FormControl fullWidth size="small">
            <InputLabel>Rol</InputLabel>
            <Select
              label="Rol"
              value={form.role}
              onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}
            >
              <MenuItem value="user">user</MenuItem>
              <MenuItem value="admin">admin</MenuItem>
            </Select>
          </FormControl>
          {editingUser && (
            <FormControlLabel
              control={
                <Switch
                  checked={formActive}
                  onChange={(e) => setFormActive(e.target.checked)}
                  disabled={editingUser?.id === currentUserId}
                  size="small"
                />
              }
              label="Cuenta activa"
            />
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
          <Button onClick={() => setDialogOpen(false)} size="small">
            Cancelar
          </Button>
          <Button
            variant="contained"
            size="small"
            onClick={handleSave}
            disabled={saving}
            sx={{ bgcolor: "#1E3A5F", "&:hover": { bgcolor: "#152C4A" } }}
          >
            {saving ? (
              <CircularProgress size={16} sx={{ color: "white" }} />
            ) : (
              "Guardar"
            )}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Access Logs tab ───────────────────────────────────────────────────────────

function AccessLogsTab({ token }) {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setLogs(await fetchAccessLogs(token, { limit: 200 }));
    } catch {
      setError("No se pudo cargar el registro de accesos.");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress size={26} thickness={4} />
      </Box>
    );
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1.5 }}>
        <Button
          size="small"
          startIcon={<RefreshIcon sx={{ fontSize: 16 }} />}
          onClick={load}
          sx={{ color: "text.secondary" }}
        >
          Actualizar
        </Button>
      </Box>
      <TableContainer
        component={Paper}
        variant="outlined"
        sx={{ maxHeight: 480, borderRadius: 1.5 }}
      >
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Usuario</TableCell>
              <TableCell>IP</TableCell>
              <TableCell>Ciudad</TableCell>
              <TableCell>País</TableCell>
              <TableCell>Fecha / Hora</TableCell>
              <TableCell>User Agent</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {logs.map((log) => (
              <TableRow key={log.id}>
                <TableCell sx={{ fontWeight: 500 }}>
                  {log.username ?? log.user_id}
                </TableCell>
                <TableCell
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: "11px",
                  }}
                >
                  {log.ip_address}
                </TableCell>
                <TableCell>{log.city || "—"}</TableCell>
                <TableCell>{log.country || "—"}</TableCell>
                <TableCell
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: "11px",
                    whiteSpace: "nowrap",
                  }}
                >
                  {fmtDate(log.logged_in_at)}
                </TableCell>
                <TableCell
                  sx={{
                    maxWidth: 240,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    fontFamily: "'Fira Code', monospace",
                    fontSize: "10px",
                    color: "text.secondary",
                  }}
                  title={log.user_agent}
                >
                  {log.user_agent || "—"}
                </TableCell>
              </TableRow>
            ))}
            {logs.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={6}
                  align="center"
                  sx={{ color: "text.secondary", py: 4, fontStyle: "italic" }}
                >
                  Sin registros de acceso
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
}

// ── Active Sessions tab ───────────────────────────────────────────────────────

function ActiveSessionsTab({ token }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cleaningUp, setCleaningUp] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSessions(await fetchActiveSessions(token));
    } catch {
      setError("No se pudo cargar las sesiones activas.");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    load();
  }, [load]);

  const handleForceCleanup = async (userId) => {
    setCleaningUp(userId);
    try {
      await forceCleanupUser(token, userId);
      await load();
    } catch {
      /* ignore */
    } finally {
      setCleaningUp(null);
    }
  };

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress size={26} thickness={4} />
      </Box>
    );
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1.5 }}>
        <Button
          size="small"
          startIcon={<RefreshIcon sx={{ fontSize: 16 }} />}
          onClick={load}
          sx={{ color: "text.secondary" }}
        >
          Actualizar
        </Button>
      </Box>
      <TableContainer
        component={Paper}
        variant="outlined"
        sx={{ borderRadius: 1.5 }}
      >
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Usuario</TableCell>
              <TableCell>Session ID</TableCell>
              <TableCell>Inicio</TableCell>
              <TableCell align="right">Acciones</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sessions.map((s) => (
              <TableRow key={s.id}>
                <TableCell sx={{ fontWeight: 500 }}>
                  {s.username ?? s.user_id}
                </TableCell>
                <TableCell
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: "10px",
                    color: "text.secondary",
                  }}
                >
                  {s.session_id?.slice(0, 16)}…
                </TableCell>
                <TableCell
                  sx={{
                    fontFamily: "'Fira Code', monospace",
                    fontSize: "11px",
                  }}
                >
                  {fmtDate(s.created_at)}
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="Forzar limpieza de archivos">
                    <span>
                      <IconButton
                        size="small"
                        onClick={() => handleForceCleanup(s.user_id)}
                        disabled={cleaningUp === s.user_id}
                        sx={{
                          color: "#DC2626",
                          "&:hover": { bgcolor: "rgba(220,38,38,0.06)" },
                        }}
                      >
                        {cleaningUp === s.user_id ? (
                          <CircularProgress size={15} />
                        ) : (
                          <DeleteSweepIcon sx={{ fontSize: 17 }} />
                        )}
                      </IconButton>
                    </span>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
            {sessions.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={4}
                  align="center"
                  sx={{ color: "text.secondary", py: 4, fontStyle: "italic" }}
                >
                  Sin sesiones activas
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function AdminDashboard() {
  const { token, user } = useAuth();
  const navigate = useNavigate();
  const [tab, setTab] = useState(0);

  return (
    <ThemeProvider theme={adminTheme}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');`}</style>

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
            maxWidth: 1000,
            borderRadius: 2,
            overflow: "hidden",
            border: "1px solid",
            borderColor: "divider",
          }}
        >
          {/* ── Navy accent bar ── */}
          <Box sx={{ height: 3, bgcolor: "#1E3A5F" }} />

          <Box sx={{ p: { xs: 2.5, md: 3 } }}>
            {/* ── Header ── */}
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2.5 }}
            >
              <Tooltip title="Volver al mapa">
                <IconButton
                  size="small"
                  onClick={() => navigate("/")}
                  sx={{ color: "text.secondary" }}
                >
                  <ArrowBackIcon fontSize="small" />
                </IconButton>
              </Tooltip>

              <Box
                component="img"
                src={logoSrc}
                alt="Logo"
                sx={{
                  height: 28,
                  width: "auto",
                  objectFit: "contain",
                  display: { xs: "none", sm: "block" },
                }}
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />

              <Divider
                orientation="vertical"
                flexItem
                sx={{ display: { xs: "none", sm: "block" }, mx: 0.5 }}
              />

              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="subtitle2"
                  color="text.secondary"
                  sx={{ mb: 0.1 }}
                >
                  Administración
                </Typography>
                <Typography
                  variant="h6"
                  color="text.primary"
                  sx={{ lineHeight: 1.2 }}
                >
                  Panel de Administración
                </Typography>
              </Box>

              <Button
                size="small"
                variant="outlined"
                startIcon={<StorageIcon sx={{ fontSize: 15 }} />}
                onClick={() => navigate("/cache")}
                sx={{
                  fontSize: 13,
                  borderColor: "#CBD5E1",
                  color: "text.secondary",
                  display: { xs: "none", sm: "flex" },
                }}
              >
                Cache Stats
              </Button>

              {user?.username && (
                <Box
                  sx={{
                    px: 1.5,
                    py: 0.6,
                    bgcolor: "#F1F5F9",
                    borderRadius: 1,
                    border: "1px solid #E2E8F0",
                  }}
                >
                  <Typography
                    sx={{
                      fontFamily: "'Fira Code', monospace",
                      fontSize: 11,
                      color: "text.secondary",
                    }}
                  >
                    {user.username}
                  </Typography>
                </Box>
              )}
            </Box>

            <Divider sx={{ mb: 0 }} />

            {/* ── Tabs ── */}
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              sx={{
                borderBottom: "1px solid",
                borderColor: "divider",
                mb: 0.5,
              }}
            >
              <Tab label="Usuarios" />
              <Tab label="Registro de accesos" />
              <Tab label="Sesiones activas" />
            </Tabs>

            <TabPanel value={tab} index={0}>
              <UsersTab token={token} currentUserId={user?.id} />
            </TabPanel>
            <TabPanel value={tab} index={1}>
              <AccessLogsTab token={token} />
            </TabPanel>
            <TabPanel value={tab} index={2}>
              <ActiveSessionsTab token={token} />
            </TabPanel>
          </Box>
        </Paper>

        <Typography
          variant="caption"
          color="text.disabled"
          sx={{ mt: 2.5, fontFamily: "'Fira Code', monospace" }}
        >
          RADARG · FAMAF UNC · v2.0
        </Typography>
      </Box>
    </ThemeProvider>
  );
}
