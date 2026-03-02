import { useState, useEffect } from "react";
import { Marker, Popup, useMapEvent } from "react-leaflet";
import L from "leaflet";
import { Box, Typography, Button, Divider } from "@mui/material";

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
 */
export default function MarkersOverlay({
  enabled,
  markers = [],
  onAddMarker,
  onRemoveMarker,
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
          draggable={false} // No permitir mover el marcador
        >
          <Popup
            closeButton={false}
            maxWidth={250}
            className="marker-context-popup"
          >
            <Box sx={{ p: 1 }}>
              <Typography
                variant="subtitle2"
                sx={{ fontWeight: "bold", mb: 1 }}
              >
                Marcador
              </Typography>

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
                onClick={() => handleRemoveMarker(marker.id)}
                sx={{
                  textTransform: "none",
                  fontSize: "0.85rem",
                  py: 0.5,
                }}
              >
                Eliminar marcador
              </Button>
            </Box>
          </Popup>
        </Marker>
      ))}
    </>
  );
}
