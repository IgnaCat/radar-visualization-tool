import { useState, useEffect } from "react";
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  IconButton,
  Slider,
  Switch,
  Tooltip,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import GifIcon from "@mui/icons-material/Gif";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

/**
 * GifOptionsDialog - Opciones de contenido para exportar el GIF animado.
 *
 * Props:
 * - open: bool
 * - onClose: () => void
 * - onConfirm: (options) => void
 * - frameCount: number
 * - hasBasemap: bool   — hay mapa de fondo seleccionado
 * - hasColorbar: bool  — hay campo/colormap disponible
 * - hasMetadata: bool  — hay labels por frame
 */
export default function GifOptionsDialog({
  open,
  onClose,
  onConfirm,
  frameCount = 0,
  hasBasemap = false,
  hasColorbar = false,
  hasMetadata = false,
}) {
  const [includeBasemap, setIncludeBasemap] = useState(true);
  const [showColorbar, setShowColorbar] = useState(true);
  const [showMetadata, setShowMetadata] = useState(true);
  const [showLogo, setShowLogo] = useState(true);
  const [fps, setFps] = useState(1);

  // Resetear defaults cuando se abre
  useEffect(() => {
    if (open) {
      setIncludeBasemap(hasBasemap);
      setShowColorbar(hasColorbar);
      setShowMetadata(hasMetadata);
      setShowLogo(true);
      setFps(1);
    }
  }, [open, hasBasemap, hasColorbar, hasMetadata]);

  const handleConfirm = () => {
    onConfirm({ includeBasemap, showColorbar, showMetadata, showLogo, fps });
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xs"
      fullWidth
      PaperProps={{
        sx: { borderRadius: "12px", boxShadow: "0 8px 32px rgba(0,0,0,0.15)" },
      }}
    >
      {/* Header */}
      <DialogTitle sx={{ p: 0 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 2.5,
            pt: 2,
            pb: 1.5,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <GifIcon sx={{ color: "primary.main", fontSize: "1.1rem" }} />
            <Typography
              variant="subtitle1"
              sx={{ fontWeight: 600, fontSize: "13px" }}
            >
              Opciones del GIF animado
            </Typography>
          </Box>
          <IconButton
            onClick={onClose}
            size="small"
            sx={{ color: "text.secondary" }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>

      <Divider />

      <DialogContent sx={{ px: 2.5, py: 2 }}>
        {/* Preview visual */}
        <GifPreview
          includeBasemap={includeBasemap}
          showColorbar={showColorbar}
          showMetadata={showMetadata}
          showLogo={showLogo}
          frameCount={frameCount}
        />

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mb: 2 }}
        >
          {frameCount} frame{frameCount !== 1 ? "s" : ""} · {fps} fps ·{" "}
          {frameCount > 0 ? `≈ ${(frameCount / fps).toFixed(0)}s` : "—"}
        </Typography>

        {/* Opciones */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
          <OptionRow
            label="Mapa de fondo"
            description="Imagen del mapa base como fondo"
            value={includeBasemap}
            onChange={setIncludeBasemap}
            disabled={!hasBasemap}
            disabledReason="No hay mapa base seleccionado"
          />
          <OptionRow
            label="Barra de colores"
            description="Leyenda del campo en la esquina inferior izquierda"
            value={showColorbar}
            onChange={setShowColorbar}
            disabled={!hasColorbar}
            disabledReason="No hay campo/colormap disponible"
          />
          <OptionRow
            label="Metadata"
            description="Producto, radar y timestamp en la parte inferior"
            value={showMetadata}
            onChange={setShowMetadata}
            disabled={!hasMetadata}
            disabledReason="No hay información de metadata disponible"
          />
          <OptionRow
            label="Logo LSRS"
            description="Logo institucional en la esquina superior izquierda"
            value={showLogo}
            onChange={setShowLogo}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* FPS */}
        <Box>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mb: 0.5,
            }}
          >
            <Typography
              variant="caption"
              sx={{ fontWeight: 600, fontSize: "11px", color: "text.primary" }}
            >
              Velocidad (FPS)
            </Typography>
            <Typography
              variant="caption"
              sx={{ color: "primary.main", fontWeight: 600 }}
            >
              {fps} fps
            </Typography>
          </Box>
          <Slider
            value={fps}
            onChange={(_, v) => setFps(v)}
            min={1}
            max={10}
            step={1}
            marks={[
              { value: 1, label: "1" },
              { value: 5, label: "5" },
              { value: 10, label: "10" },
            ]}
            size="small"
            sx={{ color: "primary.main" }}
          />
        </Box>
      </DialogContent>

      <Divider />

      <DialogActions sx={{ px: 2.5, py: 1.5, gap: 1 }}>
        <Button
          onClick={onClose}
          size="small"
          color="inherit"
          sx={{ fontSize: "12px", textTransform: "none" }}
        >
          Cancelar
        </Button>
        <Button
          onClick={handleConfirm}
          variant="contained"
          size="small"
          startIcon={<GifIcon />}
          sx={{
            fontSize: "12px",
            borderRadius: "8px",
            textTransform: "none",
            boxShadow: "none",
            "&:hover": { boxShadow: "none" },
          }}
        >
          Generar y descargar
        </Button>
      </DialogActions>
    </Dialog>
  );
}

/* ── Subcomponentes ───────────────────────────────────────────────────── */

function OptionRow({
  label,
  description,
  value,
  onChange,
  disabled = false,
  disabledReason,
}) {
  const control = (
    <FormControlLabel
      control={
        <Switch
          checked={value && !disabled}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          size="small"
          sx={{ mr: 0 }}
        />
      }
      label={
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <Typography
            variant="body2"
            sx={{ fontSize: "12px", fontWeight: 500 }}
          >
            {label}
          </Typography>
          {disabled && disabledReason && (
            <Tooltip title={disabledReason} placement="right">
              <InfoOutlinedIcon
                sx={{ fontSize: "13px", color: "text.disabled" }}
              />
            </Tooltip>
          )}
        </Box>
      }
      labelPlacement="start"
      sx={{
        mx: 0,
        width: "100%",
        justifyContent: "space-between",
        opacity: disabled ? 0.5 : 1,
      }}
    />
  );

  return (
    <Box sx={{ py: 0.75 }}>
      {control}
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: "block", fontSize: "10.5px", lineHeight: 1.3, pl: 0.25 }}
      >
        {description}
      </Typography>
    </Box>
  );
}

/** Preview SVG del GIF con overlays */
function GifPreview({
  includeBasemap,
  showColorbar,
  showMetadata,
  showLogo,
  frameCount,
}) {
  // El mapa ocupa todo el canvas; el strip de metadata se agrega al fondo
  const W = 220;
  const STRIP = 28; // equivalente a _BOTTOM_STRIP del backend (escalado para el SVG)
  const MAP_H = 120;
  const H = MAP_H + (showMetadata ? STRIP : 0);

  const bgColor = includeBasemap ? "#c8dff0" : "#1a1a2e";
  const stripColor = includeBasemap ? "#fff" : "#141414";

  // Colorbar: termina justo encima del strip de metadata
  const cbBarH = 58;
  const cbCardH = cbBarH + 20;
  const cbBottomOffset = showMetadata ? STRIP : 0;
  const cbCardY = MAP_H - cbCardH + 24 - cbBottomOffset; // encima del strip

  return (
    <Box
      sx={{
        mb: 2,
        borderRadius: "8px",
        overflow: "hidden",
        border: "1px solid",
        borderColor: "divider",
      }}
    >
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        <defs>
          <linearGradient id="cbGrad2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#d73027" />
            <stop offset="30%" stopColor="#fc8d59" />
            <stop offset="50%" stopColor="#fee090" />
            <stop offset="70%" stopColor="#91bfdb" />
            <stop offset="100%" stopColor="#4575b4" />
          </linearGradient>
        </defs>

        {/* Mapa (ocupa todo el ancho/alto del radar) */}
        {includeBasemap ? (
          <>
            <rect width={W} height={MAP_H} fill={bgColor} />
            <line
              x1="0"
              y1={MAP_H / 2}
              x2={W}
              y2={MAP_H / 2}
              stroke="#a0b8cc"
              strokeWidth="0.7"
            />
            <line
              x1={W / 2}
              y1="0"
              x2={W / 2}
              y2={MAP_H}
              stroke="#a0b8cc"
              strokeWidth="0.7"
            />
            <line
              x1="0"
              y1="28"
              x2={W}
              y2="38"
              stroke="#b0c8d8"
              strokeWidth="0.5"
            />
            <line
              x1="0"
              y1="84"
              x2={W}
              y2="92"
              stroke="#b0c8d8"
              strokeWidth="0.5"
            />
          </>
        ) : (
          <rect width={W} height={MAP_H} fill={bgColor} />
        )}

        {/* Radar */}
        <ellipse
          cx={W / 2}
          cy={MAP_H / 2}
          rx="72"
          ry="46"
          fill="rgba(255,200,50,0.22)"
        />
        <ellipse
          cx={W / 2 - 8}
          cy={MAP_H / 2 - 4}
          rx="44"
          ry="28"
          fill="rgba(255,120,30,0.28)"
        />
        <ellipse
          cx={W / 2 - 14}
          cy={MAP_H / 2 - 8}
          rx="22"
          ry="16"
          fill="rgba(220,40,40,0.33)"
        />

        {/* Strip de metadata */}
        {showMetadata && (
          <rect y={MAP_H} width={W} height={STRIP} fill={stripColor} />
        )}

        {/* Logo (top-left, sobre el mapa) */}
        {showLogo && (
          <g>
            <rect x="8" y="7" width="52" height="24" rx="4" fill="#4a90e2" />
            <text
              x="34"
              y="22"
              textAnchor="middle"
              fill="white"
              fontSize="8"
              fontWeight="bold"
              fontFamily="sans-serif"
            >
              LSRS
            </text>
          </g>
        )}

        {/* Colorbar (bottom-left, sobre el mapa, encima del strip) */}
        {showColorbar && (
          <g>
            <rect
              x="8"
              y={cbCardY}
              width="34"
              height={cbCardH}
              rx="4"
              fill="rgba(0,0,0,0.72)"
            />
            <rect
              x="10"
              y={cbCardY + 10}
              width="10"
              height={cbBarH}
              rx="2"
              fill="url(#cbGrad2)"
            />
            <text
              x="23"
              y={cbCardY + 13}
              fill="white"
              fontSize="4.5"
              fontFamily="monospace"
            >
              60
            </text>
            <text
              x="23"
              y={cbCardY + 10 + cbBarH / 2}
              fill="white"
              fontSize="4.5"
              fontFamily="monospace"
            >
              30
            </text>
            <text
              x="23"
              y={cbCardY + 10 + cbBarH - 2}
              fill="white"
              fontSize="4.5"
              fontFamily="monospace"
            >
              0
            </text>
            <text
              x="15"
              y={cbCardY + cbCardH - 3}
              fill="#bbb"
              fontSize="4"
              fontFamily="sans-serif"
            >
              dBZ
            </text>
          </g>
        )}

        {/* Metadata (centrada en el strip inferior) */}
        {showMetadata &&
          (() => {
            const mW = W - 40;
            const mH = 14;
            const mX = (W - mW) / 2;
            const mY = MAP_H + (STRIP - mH) / 2;
            return (
              <g>
                <rect
                  x={mX}
                  y={mY}
                  width={mW}
                  height={mH}
                  rx="3"
                  fill="white"
                />
                <text
                  x={W / 2}
                  y={mY + mH / 2 + 1.5}
                  textAnchor="middle"
                  fill="#111"
                  fontSize="4.5"
                  fontFamily="sans-serif"
                >
                  {`PPI | RMA1 | 0315 | 19/08/2025 00:35 (1/${Math.max(frameCount || 2, 2)})`}
                </text>
              </g>
            );
          })()}
      </svg>
    </Box>
  );
}
