import { useRef, useState } from "react";
import { Marker, Popup, Tooltip, useMapEvent } from "react-leaflet";
import L from "leaflet";
import {
  Box,
  Typography,
  Button,
  Divider,
  TextField,
  InputAdornment,
  IconButton,
  Slider,
  Switch,
  FormControlLabel,
  GlobalStyles,
  Collapse,
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";
import DeleteIcon from "@mui/icons-material/Delete";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";

const DEFAULT_STYLE = { color: "#2563eb", size: 41 };
const DEFAULT_LABEL_STYLE = {
  visible: true,
  fontSize: 13,
  color: "#000000",
  bgColor: "#ffffff",
  hasBg: false,
  bold: true,
};

function getMarkerStyle(style = {}) {
  return {
    ...DEFAULT_STYLE,
    ...style,
    label: {
      ...DEFAULT_LABEL_STYLE,
      ...(style?.label || {}),
    },
  };
}

function makeMarkerIcon({
  color = DEFAULT_STYLE.color,
  size = DEFAULT_STYLE.size,
} = {}) {
  const w = Math.round(size * 0.61);
  const h = size;

  return L.divIcon({
    className: "",
    html: `<svg width="${w}" height="${h}" viewBox="0 0 26 40" xmlns="http://www.w3.org/2000/svg">
      <path d="M13 0C5.82 0 0 5.82 0 13c0 9.75 13 27 13 27S26 22.75 26 13C26 5.82 20.18 0 13 0z" fill="${color}"/>
      <circle cx="13" cy="13" r="5.5" fill="rgba(255,255,255,0.55)"/>
    </svg>`,
    iconSize: [w, h],
    iconAnchor: [w / 2, h],
    popupAnchor: [0, -h + 4],
  });
}

function MarkerLabel({ marker }) {
  const title = marker.name?.trim();
  const markerStyle = getMarkerStyle(marker.style);
  const labelStyle = markerStyle.label;
  const verticalOffset = -Math.max(30, Math.round(markerStyle.size * 0.8));
  const tooltipKey = `${marker.id}-${markerStyle.size}`;

  if (!title || !labelStyle.visible) {
    return null;
  }

  return (
    <Tooltip
      key={tooltipKey}
      permanent
      direction="top"
      offset={[0, verticalOffset]}
      opacity={1}
      className="marker-title-tooltip"
    >
      <Box
        component="span"
        sx={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          maxWidth: 220,
          px: labelStyle.hasBg ? 1 : 0,
          py: labelStyle.hasBg ? 0.35 : 0,
          borderRadius: 1,
          backgroundColor: labelStyle.hasBg
            ? labelStyle.bgColor
            : "transparent",
          color: labelStyle.color,
          fontSize: `${labelStyle.fontSize}px`,
          fontWeight: labelStyle.bold ? 700 : 400,
          lineHeight: 1.2,
          whiteSpace: "nowrap",
          textOverflow: "ellipsis",
          overflow: "hidden",
          border: labelStyle.hasBg
            ? "1px solid rgba(15, 23, 42, 0.18)"
            : "none",
          boxShadow: labelStyle.hasBg
            ? "0 2px 8px rgba(15, 23, 42, 0.18)"
            : "none",
          textShadow: labelStyle.hasBg
            ? "none"
            : "0 0 8px rgba(255,255,255,0.95), 0 0 4px rgba(255,255,255,0.95)",
        }}
      >
        {title}
      </Box>
    </Tooltip>
  );
}

function MarkerPopup({ marker, onRemove, onRename, onUpdate }) {
  const [editing, setEditing] = useState(false);
  const [textStylesOpen, setTextStylesOpen] = useState(false);
  const [draft, setDraft] = useState(marker.name || "Marcador");
  const inputRef = useRef(null);
  const style = getMarkerStyle(marker.style);
  const labelStyle = style.label;

  const updateStyle = (patch) => {
    onUpdate?.(marker.id, {
      style: {
        ...style,
        ...patch,
      },
    });
  };

  const updateLabelStyle = (patch) => {
    updateStyle({
      label: {
        ...labelStyle,
        ...patch,
      },
    });
  };

  const handleStartEdit = () => {
    setDraft(marker.name || "Marcador");
    setEditing(true);
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  const handleConfirm = () => {
    const trimmed = draft.trim() || "Marcador";
    setEditing(false);
    onRename?.(marker.id, trimmed);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleConfirm();
    if (e.key === "Escape") setEditing(false);
  };

  return (
    <Box sx={{ p: 1, minWidth: 215 }}>
      {editing ? (
        <TextField
          inputRef={inputRef}
          size="small"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={handleConfirm}
          onKeyDown={handleKeyDown}
          sx={{ mb: 1, width: "100%" }}
          slotProps={{
            input: {
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onMouseDown={(e) => {
                      e.preventDefault();
                      handleConfirm();
                    }}
                  >
                    <CheckIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              ),
            },
          }}
        />
      ) : (
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: "bold", flex: 1 }}>
            {marker.name || "Marcador"}
          </Typography>
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              handleStartEdit();
            }}
            sx={{ p: 0.25 }}
          >
            <EditIcon fontSize="small" />
          </IconButton>
        </Box>
      )}
      <Typography variant="body2" sx={{ mb: 0.5, fontSize: "0.85rem" }}>
        Lat: {marker.lat.toFixed(6)} deg
      </Typography>
      <Typography variant="body2" sx={{ mb: 1.5, fontSize: "0.85rem" }}>
        Lon: {marker.lon.toFixed(6)} deg
      </Typography>

      <Divider sx={{ mb: 1 }} />

      <Typography variant="caption" sx={{ display: "block", mb: 0.75 }}>
        Marcador
      </Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
        <Typography variant="caption" sx={{ minWidth: 40 }}>
          Color:
        </Typography>
        <input
          type="color"
          value={style.color}
          onChange={(e) => updateStyle({ color: e.target.value })}
          style={{
            cursor: "pointer",
            height: 22,
            width: 34,
            border: "none",
            borderRadius: 3,
            padding: 0,
          }}
        />
      </Box>
      <Typography variant="caption">Tamano: {style.size}px</Typography>
      <Slider
        size="small"
        min={16}
        max={56}
        step={2}
        value={style.size}
        onChange={(_, value) =>
          updateStyle({ size: typeof value === "number" ? value : style.size })
        }
        sx={{ mb: 1 }}
      />

      <Divider sx={{ mb: 1 }} />

      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 1,
          mb: 0.25,
        }}
      >
        <Typography variant="caption">Titulo en mapa</Typography>
        <IconButton
          size="small"
          onClick={() => setTextStylesOpen((open) => !open)}
          sx={{ p: 0.25 }}
        >
          {textStylesOpen ? (
            <ExpandLessIcon fontSize="small" />
          ) : (
            <ExpandMoreIcon fontSize="small" />
          )}
        </IconButton>
      </Box>
      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={labelStyle.visible}
            onChange={(e) => updateLabelStyle({ visible: e.target.checked })}
          />
        }
        label={<Typography variant="caption">Mostrar titulo</Typography>}
        sx={{ mb: textStylesOpen && labelStyle.visible ? 0.75 : 0 }}
      />

      <Collapse in={textStylesOpen && labelStyle.visible}>
          <Typography variant="caption">
            Tamano texto: {labelStyle.fontSize}px
          </Typography>
          <Slider
            size="small"
            min={10}
            max={24}
            step={1}
            value={labelStyle.fontSize}
            onChange={(_, value) =>
              updateLabelStyle({
                fontSize:
                  typeof value === "number" ? value : labelStyle.fontSize,
              })
            }
            sx={{ mb: 1 }}
          />

          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <Typography variant="caption" sx={{ minWidth: 72 }}>
              Texto:
            </Typography>
            <input
              type="color"
              value={labelStyle.color}
              onChange={(e) => updateLabelStyle({ color: e.target.value })}
              style={{
                cursor: "pointer",
                height: 22,
                width: 34,
                border: "none",
                borderRadius: 3,
                padding: 0,
              }}
            />
            <Box
              component="button"
              type="button"
              onClick={() => updateLabelStyle({ bold: !labelStyle.bold })}
              sx={{
                px: 0.9,
                py: 0.3,
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 1,
                cursor: "pointer",
                fontWeight: "bold",
                fontSize: "0.82rem",
                bgcolor: labelStyle.bold ? "primary.main" : "transparent",
                color: labelStyle.bold ? "#fff" : "text.primary",
                lineHeight: 1.4,
              }}
            >
              B
            </Box>
          </Box>

          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={labelStyle.hasBg}
                onChange={(e) => updateLabelStyle({ hasBg: e.target.checked })}
              />
            }
            label={<Typography variant="caption">Fondo</Typography>}
            sx={{ mb: labelStyle.hasBg ? 0.75 : 0 }}
          />

          {labelStyle.hasBg ? (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
              <Typography variant="caption" sx={{ minWidth: 72 }}>
                Fondo:
              </Typography>
              <input
                type="color"
                value={labelStyle.bgColor}
                onChange={(e) => updateLabelStyle({ bgColor: e.target.value })}
                style={{
                  cursor: "pointer",
                  height: 22,
                  width: 34,
                  border: "none",
                  borderRadius: 3,
                  padding: 0,
                }}
              />
            </Box>
          ) : null}
      </Collapse>

      <Divider sx={{ mb: 1 }} />
      <Button
        variant="outlined"
        color="error"
        size="small"
        fullWidth
        startIcon={<DeleteIcon />}
        onClick={() => onRemove?.(marker.id)}
        sx={{ textTransform: "none", fontSize: "0.85rem", py: 0.5 }}
      >
        Eliminar marcador
      </Button>
    </Box>
  );
}

export default function MarkersOverlay({
  enabled,
  markers = [],
  onAddMarker,
  onRemoveMarker,
  onRenameMarker,
  onUpdateMarker,
  onModeDeactivate,
}) {
  const [nextId, setNextId] = useState(1);

  useMapEvent("click", (e) => {
    if (!enabled) return;

    const newMarker = {
      id: nextId,
      lat: e.latlng.lat,
      lon: e.latlng.lng,
      style: { ...DEFAULT_STYLE },
    };

    setNextId((currentId) => currentId + 1);
    onAddMarker?.(newMarker);
    onModeDeactivate?.();
  });

  if (!markers || markers.length === 0) return null;

  return (
    <>
      <GlobalStyles
        styles={{
          ".marker-title-tooltip.leaflet-tooltip": {
            background: "transparent",
            border: "none",
            boxShadow: "none",
            padding: 0,
          },
          ".marker-title-tooltip.leaflet-tooltip::before": {
            display: "none",
          },
        }}
      />
      {markers.map((marker) => (
        <Marker
          key={marker.id}
          position={[marker.lat, marker.lon]}
          icon={makeMarkerIcon(marker.style)}
          draggable={false}
        >
          <MarkerLabel marker={marker} />
          <Popup
            closeButton={false}
            maxWidth={250}
            className="marker-context-popup"
          >
            <MarkerPopup
              marker={marker}
              onRemove={onRemoveMarker}
              onRename={onRenameMarker}
              onUpdate={onUpdateMarker}
            />
          </Popup>
        </Marker>
      ))}
    </>
  );
}
