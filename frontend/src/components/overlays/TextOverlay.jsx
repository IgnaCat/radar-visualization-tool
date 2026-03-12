import { useState } from "react";
import { Marker, Popup, useMapEvent } from "react-leaflet";
import L from "leaflet";
import {
  Box,
  Typography,
  TextField,
  Slider,
  Button,
  Divider,
  Switch,
  FormControlLabel,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";

function makeTextIcon(text, style) {
  const {
    fontSize = 14,
    color = "#000000",
    bgColor = "#ffffff",
    hasBg = false,
    bold = false,
    italic = false,
  } = style;

  const bgStyles = hasBg
    ? `background:${bgColor};padding:2px 8px;border-radius:3px;border:1px solid rgba(0,0,0,0.18);`
    : `text-shadow:0 0 4px #fff,0 0 4px #fff,0 0 4px #fff;`;

  return L.divIcon({
    className: "",
    html: `<div style="font-size:${fontSize}px;color:${color};font-weight:${bold ? "bold" : "normal"};font-style:${italic ? "italic" : "normal"};${bgStyles}white-space:nowrap;pointer-events:auto;cursor:pointer;line-height:1.3;">${text}</div>`,
    iconSize: null,
    iconAnchor: [0, 0],
  });
}

function TextStylePopup({ ann, onChange, onDelete }) {
  const [text, setText] = useState(ann.text);
  const [style, setStyle] = useState({ ...ann.style });

  const emitStyle = (patch) => {
    const next = { ...style, ...patch };
    setStyle(next);
    onChange({ text, style: next });
  };

  const emitText = (t) => {
    setText(t);
    onChange({ text: t, style });
  };

  return (
    <Box sx={{ p: 1, minWidth: 215 }}>
      <Typography variant="subtitle2" fontWeight="bold" mb={1}>
        Etiqueta de texto
      </Typography>

      <TextField
        size="small"
        fullWidth
        multiline
        maxRows={3}
        label="Texto"
        value={text}
        onChange={(e) => emitText(e.target.value)}
        sx={{ mb: 1.5 }}
      />

      <Typography variant="caption">Tamaño: {style.fontSize}px</Typography>
      <Slider
        size="small"
        min={8}
        max={52}
        step={1}
        value={style.fontSize}
        onChange={(_, v) => emitStyle({ fontSize: v })}
        sx={{ mb: 1 }}
      />

      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
        <Typography variant="caption">Color:</Typography>
        <input
          type="color"
          value={style.color}
          onChange={(e) => emitStyle({ color: e.target.value })}
          style={{
            cursor: "pointer",
            height: 24,
            width: 36,
            border: "none",
            borderRadius: 4,
            padding: 0,
          }}
        />
        {/* Bold / Italic mini-buttons */}
        <Box
          component="button"
          onClick={() => emitStyle({ bold: !style.bold })}
          sx={{
            px: 0.75,
            py: 0.25,
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 1,
            cursor: "pointer",
            fontWeight: "bold",
            fontSize: "0.82rem",
            bgcolor: style.bold ? "primary.main" : "transparent",
            color: style.bold ? "#fff" : "text.primary",
            lineHeight: 1.4,
          }}
        >
          B
        </Box>
        <Box
          component="button"
          onClick={() => emitStyle({ italic: !style.italic })}
          sx={{
            px: 0.75,
            py: 0.25,
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 1,
            cursor: "pointer",
            fontStyle: "italic",
            fontSize: "0.82rem",
            bgcolor: style.italic ? "primary.main" : "transparent",
            color: style.italic ? "#fff" : "text.primary",
            lineHeight: 1.4,
          }}
        >
          i
        </Box>
      </Box>

      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={style.hasBg}
            onChange={(e) => emitStyle({ hasBg: e.target.checked })}
          />
        }
        label={<Typography variant="caption">Fondo</Typography>}
        sx={{ mb: style.hasBg ? 0.75 : 0 }}
      />

      {style.hasBg && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
          <Typography variant="caption">Color fondo:</Typography>
          <input
            type="color"
            value={style.bgColor}
            onChange={(e) => emitStyle({ bgColor: e.target.value })}
            style={{
              cursor: "pointer",
              height: 24,
              width: 36,
              border: "none",
              borderRadius: 4,
              padding: 0,
            }}
          />
        </Box>
      )}

      <Divider sx={{ my: 1 }} />
      <Button
        variant="outlined"
        color="error"
        size="small"
        fullWidth
        startIcon={<DeleteIcon />}
        onClick={onDelete}
        sx={{ textTransform: "none", fontSize: "0.82rem" }}
      >
        Eliminar
      </Button>
    </Box>
  );
}

/**
 * TextOverlay - Permite colocar etiquetas de texto sobre el mapa con click.
 *
 * Props:
 * - enabled: boolean - Si está activo el modo de agregar texto
 * - annotations: array - Array de anotaciones de texto { id, lat, lon, text, style }
 * - onAdd: function - Callback al agregar una etiqueta
 * - onUpdate: function(id, patch) - Callback al editar una etiqueta
 * - onRemove: function(id) - Callback al eliminar una etiqueta
 */
export default function TextOverlay({
  enabled,
  annotations = [],
  onAdd,
  onUpdate,
  onRemove,
  onModeDeactivate,
}) {
  const [nextId, setNextId] = useState(1);

  useMapEvent("click", (e) => {
    if (!enabled) return;
    onAdd?.({
      id: nextId,
      lat: e.latlng.lat,
      lon: e.latlng.lng,
      text: "Texto",
      style: {
        fontSize: 14,
        color: "#000000",
        bgColor: "#ffffff",
        hasBg: false,
        bold: false,
        italic: false,
      },
    });
    setNextId((n) => n + 1);
    onModeDeactivate?.();
  });

  if (!annotations.length) return null;

  return (
    <>
      {annotations.map((ann) => (
        <Marker
          key={ann.id}
          position={[ann.lat, ann.lon]}
          icon={makeTextIcon(ann.text, ann.style)}
          zIndexOffset={1000}
        >
          <Popup closeButton={false} maxWidth={260}>
            <TextStylePopup
              ann={ann}
              onChange={({ text, style }) =>
                onUpdate?.(ann.id, { text, style })
              }
              onDelete={() => onRemove?.(ann.id)}
            />
          </Popup>
        </Marker>
      ))}
    </>
  );
}
