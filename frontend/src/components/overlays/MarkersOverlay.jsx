import { useState, useRef } from "react";
import { Marker, Popup, useMapEvent } from "react-leaflet";
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
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";
import DeleteIcon from "@mui/icons-material/Delete";

const DEFAULT_STYLE = { color: "#2563eb", size: 41 };

function makeMarkerIcon({
  color = DEFAULT_STYLE.color,
  size = DEFAULT_STYLE.size,
} = {}) {
  const w = Math.round(size * 0.61); // preserves original 25x41 aspect ratio
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

/**
 * MarkersOverlay - Permite agregar marcadores en el mapa con click
 *
 * Props:
 * - enabled: boolean - Si está activo el modo de agregar marcadores
 * - markers: array - Array de marcadores { id, lat, lon, name?, style? }
 * - onAddMarker: function - Callback cuando se agrega un marcador
 * - onRemoveMarker: function - Callback cuando se elimina un marcador
 * - onRenameMarker: function - Callback cuando se renombra un marcador
 * - onUpdateMarker: function(id, patch) - Callback cuando se edita el estilo
 * - onModeDeactivate: function - Callback para desactivar el modo tras colocar un marcador
 */
function MarkerPopup({ marker, onRemove, onRename, onUpdate }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(marker.name || "Marcador");
  const inputRef = useRef(null);
  const style = { ...DEFAULT_STYLE, ...(marker.style || {}) };

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
        Lat: {marker.lat.toFixed(6)}°
      </Typography>
      <Typography variant="body2" sx={{ mb: 1.5, fontSize: "0.85rem" }}>
        Lon: {marker.lon.toFixed(6)}°
      </Typography>

      <Divider sx={{ mb: 1 }} />

      {/* Estilos */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
        <Typography variant="caption" sx={{ minWidth: 40 }}>
          Color:
        </Typography>
        <input
          type="color"
          value={style.color}
          onChange={(e) =>
            onUpdate?.(marker.id, {
              style: { ...style, color: e.target.value },
            })
          }
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
      <Typography variant="caption">Tamaño: {style.size}px</Typography>
      <Slider
        size="small"
        min={16}
        max={56}
        step={2}
        value={style.size}
        onChange={(_, v) =>
          onUpdate?.(marker.id, { style: { ...style, size: v } })
        }
        sx={{ mb: 1 }}
      />

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

  // Escuchar clicks en el mapa cuando está habilitado
  useMapEvent("click", (e) => {
    if (!enabled) return;

    const newMarker = {
      id: nextId,
      lat: e.latlng.lat,
      lon: e.latlng.lng,
      style: { ...DEFAULT_STYLE },
    };

    setNextId(nextId + 1);
    onAddMarker?.(newMarker);
    onModeDeactivate?.();
  });

  // Si no hay marcadores, no renderizar nada
  if (!markers || markers.length === 0) return null;

  return (
    <>
      {markers.map((marker) => (
        <Marker
          key={marker.id}
          position={[marker.lat, marker.lon]}
          icon={makeMarkerIcon(marker.style)}
          draggable={false}
        >
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
