import { useEffect, useRef, useState } from "react";
import { useMap, Polyline, CircleMarker, Marker } from "react-leaflet";
import L from "leaflet";

// Crear icono personalizado para el marcador final (cuadrado blanco)
const finishMarkerIcon = L.divIcon({
  className: "finish-marker",
  html: '<div style="width: 16px; height: 16px; background: white; border: 3px solid #ff6b35; cursor: pointer;"></div>',
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

/**
 * Overlay para dibujar una línea poligonal en el mapa.
 * Este componente solo se encarga del dibujo y la edición básica.
 * El hover sincronizado con gráficos vive en un overlay aparte para poder reutilizarlo.
 */
export default function LineDrawOverlay({
  enabled,
  points: externalPoints,
  onComplete,
  onPointsChange,
}) {
  const map = useMap();
  const [points, setPoints] = useState([]);
  const onCompleteRef = useRef(onComplete);
  const onPointsChangeRef = useRef(onPointsChange);

  useEffect(() => {
    if (externalPoints !== undefined) {
      setPoints(externalPoints);
    }
  }, [externalPoints]);

  useEffect(() => {
    onCompleteRef.current = onComplete;
    onPointsChangeRef.current = onPointsChange;
  }, [onComplete, onPointsChange]);

  useEffect(() => {
    if (enabled && points.length > 0) {
      onPointsChangeRef.current?.(points);
    }
  }, [points, enabled]);

  useEffect(() => {
    if (!map || !enabled) {
      return;
    }

    map.getContainer().style.cursor = "crosshair";

    const handleMapClick = (e) => {
      const newPoint = { lat: e.latlng.lat, lon: e.latlng.lng };
      setPoints((prev) => [...prev, newPoint]);
    };

    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        setPoints([]);
        onPointsChangeRef.current?.([]);
      }

      if (e.key === "Enter") {
        setPoints((prevPoints) => {
          if (prevPoints.length >= 2) {
            onCompleteRef.current?.(prevPoints);
          }
          return prevPoints;
        });
      }

      if (e.key === "Delete") {
        setPoints((prev) => {
          if (prev.length === 0) return prev;
          const newPoints = prev.slice(0, -1);
          onPointsChangeRef.current?.(newPoints);
          return newPoints;
        });
      }
    };

    map.on("click", handleMapClick);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      map.off("click", handleMapClick);
      document.removeEventListener("keydown", handleKeyDown);
      map.getContainer().style.cursor = "";
    };
  }, [map, enabled]);

  const handleFinishClick = () => {
    if (points.length >= 2) {
      onCompleteRef.current?.(points);
    }
  };

  if (points.length === 0) return null;

  const positions = points.map((p) => [p.lat, p.lon]);

  return (
    <>
      {points.length > 1 && (
        <Polyline
          positions={positions}
          color="#ff6b35"
          weight={3}
          opacity={0.8}
          dashArray="5, 10"
          interactive={false}
        />
      )}

      {points.slice(0, -1).map((point, idx) => (
        <CircleMarker
          key={idx}
          center={[point.lat, point.lon]}
          radius={6}
          fillColor={idx === 0 ? "#00ff00" : "#ff6b35"}
          fillOpacity={0.9}
          color="#fff"
          weight={2}
        />
      ))}

      {!enabled && points.length >= 1 && (
        <CircleMarker
          center={[
            points[points.length - 1].lat,
            points[points.length - 1].lon,
          ]}
          radius={6}
          fillColor="#ff6b35"
          fillOpacity={0.9}
          color="#fff"
          weight={2}
        />
      )}

      {enabled && points.length >= 1 && (
        <Marker
          position={[
            points[points.length - 1].lat,
            points[points.length - 1].lon,
          ]}
          icon={finishMarkerIcon}
          eventHandlers={{
            click: handleFinishClick,
          }}
        />
      )}
    </>
  );
}
