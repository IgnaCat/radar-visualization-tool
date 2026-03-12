import { useState, useEffect, useRef, useCallback } from "react";
import {
  useMap,
  useMapEvents,
  Polyline,
  Rectangle,
  Circle,
  Polygon,
  CircleMarker,
  Marker,
  Popup,
} from "react-leaflet";
import L from "leaflet";
import {
  Box,
  Typography,
  Slider,
  Button,
  Divider,
  Switch,
  FormControlLabel,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";

// ─── Default styles ────────────────────────────────────────────────────────────

const DEFAULT_STYLES = {
  line: { color: "#ff6b35", weight: 3, opacity: 1, dashArray: "" },
  arrow: { color: "#ff6b35", weight: 3, opacity: 1 },
  rect: {
    color: "#4a90e2",
    fillColor: "#4a90e2",
    fillOpacity: 0.15,
    weight: 2,
    opacity: 1,
  },
  circle: {
    color: "#4a90e2",
    fillColor: "#4a90e2",
    fillOpacity: 0.15,
    weight: 2,
    opacity: 1,
  },
  polygon: {
    color: "#4a90e2",
    fillColor: "#4a90e2",
    fillOpacity: 0.15,
    weight: 2,
    opacity: 1,
  },
};

// ─── Icons ─────────────────────────────────────────────────────────────────────

const finishIcon = L.divIcon({
  className: "",
  html: '<div style="width:14px;height:14px;background:#fff;border:3px solid #ff6b35;border-radius:2px;cursor:pointer;box-sizing:border-box;"></div>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

// ─── Helpers ───────────────────────────────────────────────────────────────────

function haversineMeters(a, b) {
  const R = 6371000;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const sinDLat = Math.sin(dLat / 2);
  const sinDLon = Math.sin(dLon / 2);
  const c =
    sinDLat * sinDLat +
    Math.cos((a.lat * Math.PI) / 180) *
      Math.cos((b.lat * Math.PI) / 180) *
      sinDLon *
      sinDLon;
  return R * 2 * Math.atan2(Math.sqrt(c), Math.sqrt(1 - c));
}

// ─── Color row shared between style popups ─────────────────────────────────────

function ColorRow({ label, value, onChange }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
      <Typography variant="caption" sx={{ minWidth: 85 }}>
        {label}:
      </Typography>
      <input
        type="color"
        value={value}
        onChange={(e) => onChange(e.target.value)}
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
  );
}

// ─── Style popup for line / arrow ──────────────────────────────────────────────

function LineStylePopup({ shape, onChange, onDelete }) {
  const [s, setS] = useState({ ...shape.style });
  const emit = (patch) => {
    const next = { ...s, ...patch };
    setS(next);
    onChange({ style: next });
  };

  return (
    <Box sx={{ p: 1, minWidth: 205 }}>
      <Typography variant="subtitle2" fontWeight="bold" mb={1}>
        {shape.type === "line" ? "Línea" : "Flecha"}
      </Typography>

      <ColorRow
        label="Color"
        value={s.color}
        onChange={(v) => emit({ color: v })}
      />

      <Typography variant="caption">Grosor: {s.weight}px</Typography>
      <Slider
        size="small"
        min={1}
        max={12}
        step={0.5}
        value={s.weight}
        onChange={(_, v) => emit({ weight: v })}
        sx={{ mb: 1 }}
      />

      <Typography variant="caption">
        Opacidad: {Math.round(s.opacity * 100)}%
      </Typography>
      <Slider
        size="small"
        min={0.1}
        max={1}
        step={0.05}
        value={s.opacity}
        onChange={(_, v) => emit({ opacity: v })}
        sx={{ mb: 1 }}
      />

      {shape.type === "line" && (
        <>
          <Typography variant="caption" sx={{ display: "block", mb: 0.5 }}>
            Estilo de línea:
          </Typography>
          <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", mb: 1 }}>
            {[
              { v: "", label: "—" },
              { v: "6,4", label: "- -" },
              { v: "2,4", label: "···" },
              { v: "10,4,2,4", label: "—·—" },
            ].map(({ v, label }) => (
              <Box
                key={v || "solid"}
                component="button"
                onClick={() => emit({ dashArray: v })}
                sx={{
                  px: 0.8,
                  py: 0.2,
                  border: "1px solid",
                  borderColor: s.dashArray === v ? "primary.main" : "divider",
                  borderRadius: 1,
                  cursor: "pointer",
                  fontSize: "0.78rem",
                  bgcolor: s.dashArray === v ? "primary.main" : "transparent",
                  color: s.dashArray === v ? "#fff" : "text.secondary",
                  letterSpacing: "0.05em",
                }}
              >
                {label}
              </Box>
            ))}
          </Box>
        </>
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

// ─── Style popup for rect / circle / polygon ───────────────────────────────────

function FillStylePopup({ shape, onChange, onDelete }) {
  const [s, setS] = useState({ ...shape.style });
  const emit = (patch) => {
    const next = { ...s, ...patch };
    setS(next);
    onChange({ style: next });
  };

  const typeLabel =
    shape.type === "rect"
      ? "Rectángulo"
      : shape.type === "circle"
        ? "Círculo"
        : "Polígono";

  return (
    <Box sx={{ p: 1, minWidth: 205 }}>
      <Typography variant="subtitle2" fontWeight="bold" mb={1}>
        {typeLabel}
      </Typography>

      <ColorRow
        label="Borde"
        value={s.color}
        onChange={(v) => emit({ color: v })}
      />
      <Typography variant="caption">Grosor borde: {s.weight}px</Typography>
      <Slider
        size="small"
        min={0}
        max={10}
        step={0.5}
        value={s.weight}
        onChange={(_, v) => emit({ weight: v })}
        sx={{ mb: 1 }}
      />

      <ColorRow
        label="Relleno"
        value={s.fillColor}
        onChange={(v) => emit({ fillColor: v })}
      />
      <Typography variant="caption">
        Opacidad relleno: {Math.round(s.fillOpacity * 100)}%
      </Typography>
      <Slider
        size="small"
        min={0}
        max={1}
        step={0.05}
        value={s.fillOpacity}
        onChange={(_, v) => emit({ fillOpacity: v })}
        sx={{ mb: 1 }}
      />

      <Typography variant="caption">
        Opacidad borde: {Math.round(s.opacity * 100)}%
      </Typography>
      <Slider
        size="small"
        min={0.1}
        max={1}
        step={0.05}
        value={s.opacity}
        onChange={(_, v) => emit({ opacity: v })}
        sx={{ mb: 1 }}
      />

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

// ─── Arrow head marker (recomputed on zoom) ────────────────────────────────────

function ArrowHead({ points, color, opacity }) {
  const map = useMap();
  const [icon, setIcon] = useState(null);
  const lastPt = points[points.length - 1];
  const prevPt = points[points.length - 2];

  const compute = useCallback(() => {
    if (!prevPt || !lastPt || !map) return;
    const p1 = map.latLngToContainerPoint([prevPt.lat, prevPt.lon]);
    const p2 = map.latLngToContainerPoint([lastPt.lat, lastPt.lon]);
    const angleDeg = (Math.atan2(p2.x - p1.x, -(p2.y - p1.y)) * 180) / Math.PI;
    setIcon(
      L.divIcon({
        className: "",
        html: `<svg viewBox="0 0 24 24" width="24" height="24" style="transform:rotate(${angleDeg}deg);transform-origin:12px 12px;opacity:${opacity};overflow:visible;display:block;">
          <polygon points="12,1 21,21 12,15.5 3,21" fill="${color}" stroke="${color}" stroke-width="1" stroke-linejoin="round"/>
        </svg>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      }),
    );
  }, [map, prevPt, lastPt, color, opacity]);

  useEffect(() => {
    compute();
    map.on("zoomend", compute);
    map.on("moveend", compute);
    return () => {
      map.off("zoomend", compute);
      map.off("moveend", compute);
    };
  }, [compute, map]);

  if (!icon || !lastPt) return null;
  return (
    <Marker
      position={[lastPt.lat, lastPt.lon]}
      icon={icon}
      interactive={false}
      zIndexOffset={500}
    />
  );
}

// ─── Single completed shape renderer ──────────────────────────────────────────

function RenderedShape({ shape, isDrawing, onUpdate, onRemove }) {
  const interactive = !isDrawing;
  const popupNode =
    shape.type === "line" || shape.type === "arrow" ? (
      <LineStylePopup
        shape={shape}
        onChange={(p) => onUpdate(shape.id, p)}
        onDelete={() => onRemove(shape.id)}
      />
    ) : (
      <FillStylePopup
        shape={shape}
        onChange={(p) => onUpdate(shape.id, p)}
        onDelete={() => onRemove(shape.id)}
      />
    );

  if (shape.type === "line") {
    return (
      <Polyline
        positions={shape.points.map((p) => [p.lat, p.lon])}
        pathOptions={{
          color: shape.style.color,
          weight: shape.style.weight,
          opacity: shape.style.opacity,
          dashArray: shape.style.dashArray || null,
          interactive,
        }}
      >
        <Popup closeButton={false} maxWidth={240}>
          {popupNode}
        </Popup>
      </Polyline>
    );
  }

  if (shape.type === "arrow") {
    return (
      <>
        <Polyline
          positions={shape.points.map((p) => [p.lat, p.lon])}
          pathOptions={{
            color: shape.style.color,
            weight: shape.style.weight,
            opacity: shape.style.opacity,
            interactive,
          }}
        >
          <Popup closeButton={false} maxWidth={240}>
            {popupNode}
          </Popup>
        </Polyline>
        <ArrowHead
          points={shape.points}
          color={shape.style.color}
          opacity={shape.style.opacity}
        />
      </>
    );
  }

  if (shape.type === "rect") {
    const lats = [shape.points[0].lat, shape.points[1].lat];
    const lons = [shape.points[0].lon, shape.points[1].lon];
    const bounds = [
      [Math.min(...lats), Math.min(...lons)],
      [Math.max(...lats), Math.max(...lons)],
    ];
    return (
      <Rectangle
        bounds={bounds}
        pathOptions={{
          color: shape.style.color,
          fillColor: shape.style.fillColor,
          fillOpacity: shape.style.fillOpacity,
          weight: shape.style.weight,
          opacity: shape.style.opacity,
          interactive,
        }}
      >
        <Popup closeButton={false} maxWidth={240}>
          {popupNode}
        </Popup>
      </Rectangle>
    );
  }

  if (shape.type === "circle") {
    return (
      <Circle
        center={[shape.center.lat, shape.center.lon]}
        radius={shape.radius}
        pathOptions={{
          color: shape.style.color,
          fillColor: shape.style.fillColor,
          fillOpacity: shape.style.fillOpacity,
          weight: shape.style.weight,
          opacity: shape.style.opacity,
          interactive,
        }}
      >
        <Popup closeButton={false} maxWidth={240}>
          {popupNode}
        </Popup>
      </Circle>
    );
  }

  if (shape.type === "polygon") {
    return (
      <Polygon
        positions={shape.points.map((p) => [p.lat, p.lon])}
        pathOptions={{
          color: shape.style.color,
          fillColor: shape.style.fillColor,
          fillOpacity: shape.style.fillOpacity,
          weight: shape.style.weight,
          opacity: shape.style.opacity,
          interactive,
        }}
      >
        <Popup closeButton={false} maxWidth={240}>
          {popupNode}
        </Popup>
      </Polygon>
    );
  }

  return null;
}

// ─── Main component ────────────────────────────────────────────────────────────

/**
 * ShapeAnnotationsOverlay - Herramienta de dibujo de formas sobre el mapa.
 *
 * Props:
 * - drawingMode: 'line' | 'arrow' | 'rect' | 'circle' | 'polygon' | null
 * - shapes: array - Array de formas completadas
 * - onAdd: function(shape) - Callback al completar una forma
 * - onUpdate: function(id, patch) - Callback al editar estilos
 * - onRemove: function(id) - Callback al eliminar una forma
 * - onModeDeactivate: function() - Callback para desactivar el modo de dibujo tras completar
 */
export default function ShapeAnnotationsOverlay({
  drawingMode,
  shapes = [],
  onAdd,
  onUpdate,
  onRemove,
  onModeDeactivate,
}) {
  const map = useMap();
  const [pts, setPts] = useState([]);
  const [mouseLL, setMouseLL] = useState(null);

  // Stable refs for use inside keyboard handler (which is set up once)
  const ptsRef = useRef([]);
  const drawingModeRef = useRef(drawingMode);
  const nextIdRef = useRef(1);
  const onAddRef = useRef(onAdd);
  const onModeDeactivateRef = useRef(onModeDeactivate);

  // Sync refs at render time (safe: render runs before any event handler)
  ptsRef.current = pts;
  drawingModeRef.current = drawingMode;
  onAddRef.current = onAdd;
  onModeDeactivateRef.current = onModeDeactivate;

  // Custom setPts that also keeps the ref in sync immediately
  const updatePts = useCallback((updater) => {
    setPts((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      ptsRef.current = next;
      return next;
    });
  }, []);

  // Reset drawing state when mode changes
  useEffect(() => {
    updatePts([]);
    setMouseLL(null);
  }, [drawingMode, updatePts]);

  // Change cursor while drawing
  useEffect(() => {
    const container = map.getContainer();
    if (drawingMode) container.style.cursor = "crosshair";
    return () => {
      container.style.cursor = "";
    };
  }, [drawingMode, map]);

  // Core finish drawing logic (reads from refs — safe from any closure)
  const finishDrawing = useCallback(
    (points, mode) => {
      if (!mode) return;
      const minPts = mode === "polygon" ? 3 : 2;
      if (points.length < minPts) return;

      const style = { ...DEFAULT_STYLES[mode] };
      const id = nextIdRef.current++;
      let newShape;

      if (mode === "line" || mode === "arrow") {
        newShape = { id, type: mode, points: [...points], style };
      } else if (mode === "rect") {
        newShape = {
          id,
          type: "rect",
          points: [points[0], points[1]],
          style,
        };
      } else if (mode === "circle") {
        newShape = {
          id,
          type: "circle",
          center: points[0],
          radius: haversineMeters(points[0], points[1]),
          style,
        };
      } else if (mode === "polygon") {
        newShape = { id, type: "polygon", points: [...points], style };
      }

      if (newShape) {
        onAddRef.current?.(newShape);
        updatePts([]);
        setMouseLL(null);
        onModeDeactivateRef.current?.();
      }
    },
    [updatePts],
  );

  // Keyboard shortcuts (set up once, uses refs)
  useEffect(() => {
    const handleKey = (e) => {
      const mode = drawingModeRef.current;
      if (!mode) return;
      if (e.key === "Escape") {
        updatePts([]);
        setMouseLL(null);
      } else if (e.key === "Delete" || e.key === "Backspace") {
        updatePts((prev) => prev.slice(0, -1));
      } else if (e.key === "Enter") {
        finishDrawing(ptsRef.current, mode);
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [finishDrawing, updatePts]);

  // Map event handlers
  useMapEvents({
    click(e) {
      if (!drawingMode) return;
      const pt = { lat: e.latlng.lat, lon: e.latlng.lng };
      if (drawingMode === "rect" || drawingMode === "circle") {
        if (pts.length === 0) {
          updatePts([pt]);
        } else {
          finishDrawing([pts[0], pt], drawingMode);
        }
      } else {
        // line, arrow, polygon: accumulate points
        updatePts((prev) => [...prev, pt]);
      }
    },
    mousemove(e) {
      if (!drawingMode || pts.length === 0) return;
      setMouseLL({ lat: e.latlng.lat, lon: e.latlng.lng });
    },
    dblclick(e) {
      // Prevent map zoom when double-clicking during drawing
      if (drawingMode) L.DomEvent.stopPropagation(e);
    },
  });

  // Finish marker click handler
  const handleFinishClick = (e) => {
    L.DomEvent.stopPropagation(e);
    finishDrawing(ptsRef.current, drawingModeRef.current);
  };

  // Preview point (mouse position or last placed point)
  const previewPt = mouseLL ?? (pts.length > 0 ? pts[pts.length - 1] : null);
  const previewColor = DEFAULT_STYLES[drawingMode]?.color ?? "#ff6b35";

  const canFinish =
    (drawingMode === "line" || drawingMode === "arrow") && pts.length >= 2;
  const canFinishPolygon = drawingMode === "polygon" && pts.length >= 3;

  // ─── Render ─────────────────────────────────────────────────────────────────

  return (
    <>
      {/* Completed shapes */}
      {shapes.map((shape) => (
        <RenderedShape
          key={shape.id}
          shape={shape}
          isDrawing={!!drawingMode}
          onUpdate={onUpdate}
          onRemove={onRemove}
        />
      ))}

      {/* In-progress drawing preview */}
      {drawingMode && pts.length > 0 && (
        <>
          {/* Preview polyline for line / arrow / polygon */}
          {(drawingMode === "line" ||
            drawingMode === "arrow" ||
            drawingMode === "polygon") &&
            previewPt && (
              <Polyline
                positions={[...pts, previewPt].map((p) => [p.lat, p.lon])}
                pathOptions={{
                  color: previewColor,
                  weight: 2.5,
                  opacity: 0.65,
                  dashArray: "6,5",
                  interactive: false,
                }}
              />
            )}

          {/* Preview rectangle while dragging second corner */}
          {drawingMode === "rect" && pts.length === 1 && previewPt && (
            <Rectangle
              bounds={[
                [
                  Math.min(pts[0].lat, previewPt.lat),
                  Math.min(pts[0].lon, previewPt.lon),
                ],
                [
                  Math.max(pts[0].lat, previewPt.lat),
                  Math.max(pts[0].lon, previewPt.lon),
                ],
              ]}
              pathOptions={{
                color: previewColor,
                fillColor: previewColor,
                fillOpacity: 0.08,
                weight: 2,
                dashArray: "6,5",
                interactive: false,
              }}
            />
          )}

          {/* Preview circle while dragging radius */}
          {drawingMode === "circle" && pts.length === 1 && previewPt && (
            <Circle
              center={[pts[0].lat, pts[0].lon]}
              radius={haversineMeters(pts[0], previewPt)}
              pathOptions={{
                color: previewColor,
                fillColor: previewColor,
                fillOpacity: 0.08,
                weight: 2,
                dashArray: "6,5",
                interactive: false,
              }}
            />
          )}

          {/* First point (green dot) */}
          <CircleMarker
            center={[pts[0].lat, pts[0].lon]}
            radius={5}
            pathOptions={{
              color: "#fff",
              weight: 2,
              fillColor: "#00c853",
              fillOpacity: 1,
              interactive: false,
            }}
          />

          {/* Intermediate point dots */}
          {(drawingMode === "line" ||
            drawingMode === "arrow" ||
            drawingMode === "polygon") &&
            pts.slice(1, -1).map((p, i) => (
              <CircleMarker
                key={i}
                center={[p.lat, p.lon]}
                radius={4}
                pathOptions={{
                  color: "#fff",
                  weight: 2,
                  fillColor: previewColor,
                  fillOpacity: 0.9,
                  interactive: false,
                }}
              />
            ))}

          {/* Last point: finish marker (white square) or regular dot */}
          {canFinish || canFinishPolygon ? (
            <Marker
              position={[pts[pts.length - 1].lat, pts[pts.length - 1].lon]}
              icon={finishIcon}
              eventHandlers={{ click: handleFinishClick }}
              zIndexOffset={1000}
            />
          ) : (
            pts.length > 0 && (
              <CircleMarker
                center={[pts[pts.length - 1].lat, pts[pts.length - 1].lon]}
                radius={5}
                pathOptions={{
                  color: "#fff",
                  weight: 2,
                  fillColor: previewColor,
                  fillOpacity: 0.9,
                  interactive: false,
                }}
              />
            )
          )}

          {/* Live arrow preview head */}
          {drawingMode === "arrow" && pts.length >= 1 && previewPt && (
            <ArrowHead
              points={[...pts, previewPt]}
              color={previewColor}
              opacity={0.65}
            />
          )}
        </>
      )}
    </>
  );
}
