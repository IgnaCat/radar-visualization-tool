import { useState, useCallback } from "react";
import { Box, IconButton, Paper } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import { sendUserLocation } from "../../api/backend";

export default function ZoomControls({
  map,
  bottomOffset = 22,
  sessionId = null,
  onLocationFound = null,
  locationActive = false,
}) {
  const [locating, setLocating] = useState(false);

  const handleZoomIn = () => {
    if (map) map.zoomIn();
  };

  const handleZoomOut = () => {
    if (map) map.zoomOut();
  };

  const handleLocateMe = useCallback(() => {
    if (!map || !("geolocation" in navigator)) return;

    setLocating(true);

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        map.setView([latitude, longitude], 12);
        setLocating(false);

        if (onLocationFound) {
          onLocationFound([latitude, longitude]);
        }

        if (sessionId) {
          sendUserLocation(sessionId, latitude, longitude);
        }
      },
      (err) => {
        console.warn("Geolocation denied or failed:", err.message);
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 },
    );
  }, [map, sessionId, onLocationFound]);

  const buttonSx = {
    width: 18,
    height: 18,
    borderRadius: "6px",
    margin: "2px",
    color: "#666",
    transition: "all 0.2s ease",
    border: "0px solid transparent",
    boxShadow: "none",
    "&:hover": {
      backgroundColor: "rgba(25, 118, 210, 0.08)",
      color: "#1976d2",
    },
  };

  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        bottom: bottomOffset,
        right: 12,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        backgroundColor: "rgba(255, 255, 255, 0.98)",
        backdropFilter: "blur(8px)",
        borderRadius: "8px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
        padding: "4px",
      }}
    >
      {/* Zoom In */}
      <IconButton onClick={handleZoomIn} sx={buttonSx}>
        <AddIcon fontSize="small" />
      </IconButton>

      {/* Divider */}
      <Box
        sx={{
          height: "1px",
          backgroundColor: "rgba(0, 0, 0, 0.08)",
          margin: "2px 8px",
        }}
      />

      {/* Zoom Out */}
      <IconButton onClick={handleZoomOut} sx={buttonSx}>
        <RemoveIcon fontSize="small" />
      </IconButton>

      {/* Divider */}
      <Box
        sx={{
          height: "1px",
          backgroundColor: "rgba(0, 0, 0, 0.08)",
          margin: "2px 8px",
        }}
      />

      {/* Locate Me */}
      <IconButton
        onClick={handleLocateMe}
        disabled={locating}
        title="Ir a mi ubicación"
        sx={{
          ...buttonSx,
          color: locationActive ? "#1976d2" : locating ? "#1976d2" : "#666",
          "&:hover": {
            backgroundColor: "rgba(25, 118, 210, 0.08)",
            color: "#1976d2",
          },
        }}
      >
        <MyLocationIcon fontSize="small" />
      </IconButton>
    </Paper>
  );
}
