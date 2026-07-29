import { Box, Button, Paper, Tooltip, Typography } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import LogoutIcon from "@mui/icons-material/Logout";
import AdminPanelSettingsIcon from "@mui/icons-material/AdminPanelSettings";
import { useNavigate } from "react-router-dom";
import logoSrc from "../../assets/lrsr_logo.png";

const btnSx = {
  backgroundColor: "rgba(255, 255, 255, 0.25)",
  color: "white",
  textTransform: "none",
  fontWeight: 500,
  fontSize: "12px",
  padding: "7px 14px",
  borderRadius: "6px",
  boxShadow: "none",
  border: "1px solid rgba(255, 255, 255, 0.3)",
  "&:hover": {
    backgroundColor: "rgba(255, 255, 255, 0.35)",
    boxShadow: "none",
    border: "1px solid rgba(255, 255, 255, 0.5)",
  },
};

export default function HeaderCard({ onUploadClick, onLogout, isAdmin, username }) {
  const navigate = useNavigate();

  return (
    <Paper
      className="header-card"
      elevation={3}
      sx={{
        position: "absolute",
        top: 12,
        left: 14,
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        gap: 1,
        padding: "8px 14px",
        backgroundColor: "rgba(74, 144, 226, 0.95)",
        backdropFilter: "blur(8px)",
        borderRadius: "8px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
      }}
    >
      {/* Logo */}
      <Box
        component="img"
        className="header-logo"
        src={logoSrc}
        alt="LRSR Logo"
        sx={{
          height: 36,
          width: "auto",
          objectFit: "contain",
          marginLeft: 1,
          marginRight: 1,
        }}
        onError={(e) => {
          e.target.style.display = "none";
        }}
      />

      {/* Botón de upload */}
      <Button
        variant="contained"
        startIcon={<CloudUploadIcon />}
        onClick={onUploadClick}
        sx={btnSx}
      >
        Subir archivos
      </Button>

      {/* Admin dashboard button — only visible to admins */}
      {isAdmin && (
        <Tooltip title="Panel de administración">
          <Button
            variant="contained"
            startIcon={<AdminPanelSettingsIcon />}
            onClick={() => navigate("/admin")}
            sx={btnSx}
          >
            Admin
          </Button>
        </Tooltip>
      )}

      {/* Spacer + username */}
      {username && (
        <Typography
          variant="caption"
          sx={{
            color: "rgba(255,255,255,0.85)",
            fontSize: "11px",
            mx: 0.5,
            whiteSpace: "nowrap",
          }}
        >
          {username}
        </Typography>
      )}

      {/* Logout button */}
      {onLogout && (
        <Tooltip title="Cerrar sesión">
          <Button
            variant="contained"
            startIcon={<LogoutIcon />}
            onClick={onLogout}
            sx={btnSx}
          >
            Salir
          </Button>
        </Tooltip>
      )}
    </Paper>
  );
}
