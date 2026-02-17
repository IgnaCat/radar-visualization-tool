import { useState, useMemo } from "react";
import {
  Box,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Typography,
  Chip,
  Tooltip,
  Divider,
  Paper,
  Collapse,
  Button,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CloseIcon from "@mui/icons-material/Close";

/**
 * Panel lateral para gestionar archivos subidos.
 * Permite ver metadata y eliminar archivos individuales del servidor.
 * Diseño Collapse+Paper similar a LayerManagerDialog.
 */
export default function FileManagerDialog({
  open,
  onClose,
  filesInfo = [],
  onRemoveFile,
}) {
  const [confirmDelete, setConfirmDelete] = useState(null); // filepath pendiente de confirmación

  // Agrupar archivos por radar para mejor visualización
  const groupedFiles = useMemo(() => {
    const groups = {};
    filesInfo.forEach((file) => {
      const filename = String(file.filepath || "")
        .split("/")
        .pop()
        .split("\\")
        .pop();
      const parts = filename.split("_");
      const radar = parts.length >= 1 ? parts[0] : "Desconocido";
      if (!groups[radar]) {
        groups[radar] = [];
      }
      groups[radar].push(file);
    });
    return groups;
  }, [filesInfo]);

  const handleDeleteClick = (filepath) => {
    setConfirmDelete(filepath);
  };

  const handleConfirmDelete = () => {
    if (confirmDelete) {
      onRemoveFile?.(confirmDelete);
      setConfirmDelete(null);
    }
  };

  const handleCancelDelete = () => {
    setConfirmDelete(null);
  };

  const handleClose = () => {
    setConfirmDelete(null);
    onClose();
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const extractTimestamp = (filepath) => {
    const filename = String(filepath || "")
      .split("/")
      .pop()
      .split("\\")
      .pop();
    const match = filename.match(/(\d{8}T\d{6})Z/i);
    if (!match) return null;
    const raw = match[1];
    const year = raw.slice(0, 4);
    const month = raw.slice(4, 6);
    const day = raw.slice(6, 8);
    const hour = raw.slice(9, 11);
    const min = raw.slice(11, 13);
    const sec = raw.slice(13, 15);
    return `${day}/${month}/${year} ${hour}:${min}:${sec}`;
  };

  const extractVolume = (filepath) => {
    const filename = String(filepath || "")
      .split("/")
      .pop()
      .split("\\")
      .pop();
    const parts = filename.split("_");
    return parts.length >= 3 ? parts[2] : null;
  };

  const extractStrategy = (filepath) => {
    const filename = String(filepath || "")
      .split("/")
      .pop()
      .split("\\")
      .pop();
    const parts = filename.split("_");
    return parts.length >= 2 ? parts[1] : null;
  };

  return (
    <Collapse
      in={open}
      orientation="horizontal"
      timeout={200}
      easing={{
        enter: "cubic-bezier(0.4, 0, 0.2, 1)",
        exit: "cubic-bezier(0.4, 0, 0.6, 1)",
      }}
    >
      <Paper
        elevation={3}
        sx={{
          position: "absolute",
          top: 70,
          left: 68,
          zIndex: 999,
          width: 360,
          maxHeight: "calc(100vh - 100px)",
          overflowY: "auto",
          backgroundColor: "rgba(255, 255, 255, 0.98)",
          backdropFilter: "blur(8px)",
          borderRadius: "8px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
          transition: "opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        {/* Header */}
        <Box
          sx={{
            padding: "12px 16px",
            borderBottom: "1px solid rgba(0, 0, 0, 0.08)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box display="flex" alignItems="center" gap={1}>
            <InsertDriveFileIcon sx={{ fontSize: 18, color: "primary.main" }} />
            <Typography
              variant="subtitle1"
              sx={{ fontWeight: 600, fontSize: "14px", color: "#212121" }}
            >
              Archivos Cargados
            </Typography>
            <Chip
              label={filesInfo.length}
              size="small"
              color="primary"
              variant="outlined"
              sx={{ height: 20, fontSize: "0.7rem" }}
            />
          </Box>
          <IconButton
            size="small"
            onClick={handleClose}
            sx={{ padding: "4px" }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Content */}
        <Box sx={{ padding: "12px 16px" }}>
          {filesInfo.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No hay archivos cargados
            </Typography>
          ) : (
            Object.entries(groupedFiles).map(([radar, files], groupIdx) => (
              <Box key={radar} mb={1.5}>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontWeight: 600, display: "block", mb: 0.5 }}
                >
                  Radar: {radar}
                </Typography>

                <List disablePadding dense>
                  {files.map((file) => {
                    const filename = String(file.filepath || "")
                      .split("/")
                      .pop()
                      .split("\\")
                      .pop();
                    const isDeleting = confirmDelete === file.filepath;
                    const timestamp = extractTimestamp(file.filepath);
                    const volume = extractVolume(file.filepath);
                    const strategy = extractStrategy(file.filepath);

                    return (
                      <ListItem
                        key={file.filepath}
                        sx={{
                          border: "1px solid",
                          borderColor: isDeleting ? "error.main" : "divider",
                          borderRadius: 1,
                          mb: 0.5,
                          padding: "6px 8px",
                          bgcolor: isDeleting ? "error.50" : "background.paper",
                          transition: "all 0.2s",
                        }}
                      >
                        <ListItemText
                          primary={
                            <Typography
                              variant="body2"
                              fontWeight={500}
                              noWrap
                              title={filename}
                              sx={{ fontSize: "0.8rem" }}
                            >
                              {filename}
                            </Typography>
                          }
                          secondary={
                            <Box
                              display="flex"
                              gap={0.5}
                              flexWrap="wrap"
                              mt={0.5}
                            >
                              {timestamp && (
                                <Chip
                                  label={timestamp}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 18, fontSize: "0.65rem" }}
                                />
                              )}
                              {volume && (
                                <Chip
                                  label={`Vol. ${volume}`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 18, fontSize: "0.65rem" }}
                                />
                              )}
                              {strategy && (
                                <Chip
                                  label={`Estr. ${strategy}`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 18, fontSize: "0.65rem" }}
                                />
                              )}
                            </Box>
                          }
                        />

                        <Box
                          sx={{ ml: 1, display: "flex", alignItems: "center" }}
                        >
                          {isDeleting ? (
                            <Box display="flex" gap={0.5} alignItems="center">
                              <Tooltip title="Confirmar eliminación">
                                <Button
                                  size="small"
                                  color="error"
                                  onClick={handleConfirmDelete}
                                  sx={{
                                    minWidth: "auto",
                                    fontSize: "0.65rem",
                                    textTransform: "none",
                                    padding: "2px 6px",
                                  }}
                                >
                                  Si
                                </Button>
                              </Tooltip>

                              <Button
                                size="small"
                                onClick={handleCancelDelete}
                                sx={{
                                  minWidth: "auto",
                                  fontSize: "0.65rem",
                                  textTransform: "none",
                                  padding: "2px 6px",
                                }}
                              >
                                No
                              </Button>
                            </Box>
                          ) : (
                            <Tooltip title="Eliminar archivo">
                              <IconButton
                                size="small"
                                onClick={() => handleDeleteClick(file.filepath)}
                                sx={{
                                  padding: "4px",
                                  color: "text.secondary",
                                  "&:hover": {
                                    color: "error.main",
                                    backgroundColor: "rgba(211, 47, 47, 0.08)",
                                  },
                                }}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Box>
                      </ListItem>
                    );
                  })}
                </List>

                {groupIdx < Object.keys(groupedFiles).length - 1 && (
                  <Divider sx={{ mt: 1 }} />
                )}
              </Box>
            ))
          )}
        </Box>
      </Paper>
    </Collapse>
  );
}
