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
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";

// Icono personalizado de ubicación (pin de mapa)
const markerIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

/**
 * MarkersOverlay - Permite agregar marcadores en el mapa con click
 *
 * Props:
 * - enabled: boolean - Si está activo el modo de agregar marcadores
 * - markers: array - Array de marcadores { id, lat, lon }
 * - onAddMarker: function - Callback cuando se agrega un marcador
 * - onRemoveMarker: function - Callback cuando se elimina un marcador
 * - onRenameMarker: function - Callback cuando se renombra un marcador
 */
function MarkerPopup({ marker, onRemove, onRename }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(marker.name || "Marcador");
  const inputRef = useRef(null);

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
    <Box sx={{ p: 1 }}>
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
      <Button
        variant="outlined"
        color="error"
        size="small"
        fullWidth
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
}) {
  const [nextId, setNextId] = useState(1);

  // Escuchar clicks en el mapa cuando está habilitado
  useMapEvent("click", (e) => {
    if (!enabled) return;

    const newMarker = {
      id: nextId,
      lat: e.latlng.lat,
      lon: e.latlng.lng,
    };

    setNextId(nextId + 1);
    onAddMarker?.(newMarker);
  });

  const handleRemoveMarker = (markerId) => {
    onRemoveMarker?.(markerId);
  };

  // Si no hay marcadores, no renderizar nada
  if (!markers || markers.length === 0) return null;

  return (
    <>
      {markers.map((marker) => (
        <Marker
          key={marker.id}
          position={[marker.lat, marker.lon]}
          icon={markerIcon}
          draggable={false}
        >
          <Popup
            closeButton={false}
            maxWidth={250}
            className="marker-context-popup"
          >
            <MarkerPopup
              marker={marker}
              onRemove={handleRemoveMarker}
              onRename={onRenameMarker}
            />
          </Popup>
        </Marker>
      ))}
    </>
  );
}
