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
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import EditIcon from "@mui/icons-material/Edit";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import RefreshIcon from "@mui/icons-material/Refresh";

// ── helpers ──────────────────────────────────────────────────────────────────

function fmtDate(isoStr) {
  if (!isoStr) return "—";
  return new Date(isoStr).toLocaleString("es-AR", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

// ── sub-components ───────────────────────────────────────────────────────────

function TabPanel({ value, index, children }) {
  return value === index ? <Box sx={{ pt: 2 }}>{children}</Box> : null;
}

// ── Users tab ────────────────────────────────────────────────────────────────

function UsersTab({ token, currentUserId }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Create/edit dialog state
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState(null); // null → create mode
  const [form, setForm] = useState({ username: "", password: "", role: "user" });
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

  useEffect(() => { load(); }, [load]);

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
        await createUser(token, { username: form.username, password: form.password, role: form.role });
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

  if (loading) return <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1, gap: 1 }}>
        <Button startIcon={<RefreshIcon />} onClick={load} size="small">Actualizar</Button>
        <Button variant="contained" startIcon={<PersonAddIcon />} onClick={openCreate} size="small">
          Nuevo usuario
        </Button>
      </Box>

      <TableContainer component={Paper} variant="outlined">
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
              <TableRow key={u.id} hover>
                <TableCell>{u.username}</TableCell>
                <TableCell>
                  <Chip
                    label={u.role}
                    size="small"
                    color={u.role === "admin" ? "primary" : "default"}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={u.is_active ? "Activo" : "Inactivo"}
                    size="small"
                    color={u.is_active ? "success" : "error"}
                  />
                </TableCell>
                <TableCell>{fmtDate(u.created_at)}</TableCell>
                <TableCell align="right">
                  <Tooltip title="Editar">
                    <IconButton
                      size="small"
                      onClick={() => openEdit(u)}
                      disabled={u.id === currentUserId && u.role === "admin"}
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Create / Edit dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>{editingUser ? "Editar usuario" : "Nuevo usuario"}</DialogTitle>
        <DialogContent sx={{ display: "flex", flexDirection: "column", gap: 2, pt: "16px !important" }}>
          {formError && <Alert severity="error">{formError}</Alert>}
          <TextField
            label="Usuario"
            value={form.username}
            onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))}
            disabled={!!editingUser}
            required={!editingUser}
            autoFocus
            fullWidth
          />
          <TextField
            label={editingUser ? "Nueva contraseña (dejar vacío para no cambiar)" : "Contraseña"}
            type="password"
            value={form.password}
            onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
            required={!editingUser}
            fullWidth
          />
          <FormControl fullWidth>
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
                />
              }
              label="Cuenta activa"
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancelar</Button>
          <Button variant="contained" onClick={handleSave} disabled={saving}>
            {saving ? <CircularProgress size={18} /> : "Guardar"}
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

  useEffect(() => { load(); }, [load]);

  if (loading) return <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1 }}>
        <Button startIcon={<RefreshIcon />} onClick={load} size="small">Actualizar</Button>
      </Box>
      <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 500 }}>
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
              <TableRow key={log.id} hover>
                <TableCell>{log.username ?? log.user_id}</TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: "11px" }}>{log.ip_address}</TableCell>
                <TableCell>{log.city || "—"}</TableCell>
                <TableCell>{log.country || "—"}</TableCell>
                <TableCell sx={{ whiteSpace: "nowrap" }}>{fmtDate(log.logged_in_at)}</TableCell>
                <TableCell
                  sx={{
                    maxWidth: 260,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    fontSize: "11px",
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
                <TableCell colSpan={6} align="center" sx={{ color: "text.secondary" }}>
                  Sin registros
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

  useEffect(() => { load(); }, [load]);

  const handleForceCleanup = async (userId) => {
    setCleaningUp(userId);
    try {
      await forceCleanupUser(token, userId);
      await load();
    } catch {
      // ignore
    } finally {
      setCleaningUp(null);
    }
  };

  if (loading) return <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1 }}>
        <Button startIcon={<RefreshIcon />} onClick={load} size="small">Actualizar</Button>
      </Box>
      <TableContainer component={Paper} variant="outlined">
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
              <TableRow key={s.id} hover>
                <TableCell>{s.username ?? s.user_id}</TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: "11px" }}>
                  {s.session_id?.slice(0, 16)}…
                </TableCell>
                <TableCell>{fmtDate(s.created_at)}</TableCell>
                <TableCell align="right">
                  <Tooltip title="Forzar limpieza de archivos">
                    <IconButton
                      size="small"
                      onClick={() => handleForceCleanup(s.user_id)}
                      disabled={cleaningUp === s.user_id}
                    >
                      {cleaningUp === s.user_id
                        ? <CircularProgress size={16} />
                        : <DeleteSweepIcon fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
            {sessions.length === 0 && (
              <TableRow>
                <TableCell colSpan={4} align="center" sx={{ color: "text.secondary" }}>
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
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "#1a1a2e",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        pt: 4,
        pb: 6,
        px: 2,
      }}
    >
      <Paper
        elevation={4}
        sx={{ width: "100%", maxWidth: 960, p: 3, borderRadius: 2 }}
      >
        {/* Header row */}
        <Box sx={{ display: "flex", alignItems: "center", mb: 2, gap: 1 }}>
          <Tooltip title="Volver al mapa">
            <IconButton onClick={() => navigate("/")} size="small">
              <ArrowBackIcon />
            </IconButton>
          </Tooltip>
          <Typography variant="h6" fontWeight={600} sx={{ flexGrow: 1 }}>
            Panel de administración
          </Typography>
          <Button
            size="small"
            variant="outlined"
            onClick={() => navigate("/cache")}
            sx={{ textTransform: "none" }}
          >
            📊 Cache Stats
          </Button>
          <Typography variant="body2" color="text.secondary">
            {user?.username}
          </Typography>
        </Box>

        {/* Tabs */}
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={{ borderBottom: 1, borderColor: "divider", mb: 1 }}
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
      </Paper>
    </Box>
  );
}
