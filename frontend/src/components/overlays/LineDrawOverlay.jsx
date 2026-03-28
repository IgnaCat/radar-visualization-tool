import { useEffect, useRef, useState } from "react";
import { useMap, Polyline, CircleMarker, Marker } from "react-leaflet";
import L from "leaflet";

function distanceToSegment(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;

  if (dx === 0 && dy === 0) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }

  const t = Math.max(
    0,
    Math.min(
      1,
      ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy),
    ),
  );

  const projectedX = start.x + t * dx;
  const projectedY = start.y + t * dy;
  return Math.hypot(point.x - projectedX, point.y - projectedY);
}

// Crear icono personalizado para el marcador final (cuadrado blanco)
const finishMarkerIcon = L.divIcon({
  className: "finish-marker",
  html: '<div style="width: 16px; height: 16px; background: white; border: 3px solid #ff6b35; cursor: pointer;"></div>',
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

/**
 * Overlay para dibujar una línea poligonal en el mapa.
 * El usuario hace click para agregar puntos.
 * El último punto es un cuadrado blanco, al hacer click en él se finaliza el dibujo.
 *
 * Props:
 *  - enabled: boolean
 *  - points: {lat, lon}[] - puntos externos (controlado desde padre)
 *  - onComplete: (coordinates: {lat, lon}[]) => void - cuando el usuario termina de dibujar
 *  - onPointsChange: (coordinates: {lat, lon}[]) => void - cuando se agregan/quitan puntos
 */
export default function LineDrawOverlay({
  enabled,
  points: externalPoints,
  onComplete,
  onPointsChange,
  profilePoints = [],
  onHoverPoint,
}) {
  const map = useMap();
  const [points, setPoints] = useState([]);
  const onCompleteRef = useRef(onComplete);
  const onPointsChangeRef = useRef(onPointsChange);
  const hoverFrameRef = useRef(null);
  const lastHoveredRef = useRef(undefined);

  // Sincronizar con puntos externos (reseteo desde padre)
  useEffect(() => {
    if (externalPoints !== undefined) {
      setPoints(externalPoints);
    }
  }, [externalPoints]);

  // Limpiar estado visual cuando se desactiva
  useEffect(() => {
    if (!enabled && points.length > 0) {
      // Si se desactiva con puntos existentes, mantenerlos visibles
      // pero no permitir edición
    }
  }, [enabled, points]);

  // Mantener refs actualizadas sin causar re-renders
  useEffect(() => {
    onCompleteRef.current = onComplete;
    onPointsChangeRef.current = onPointsChange;
  }, [onComplete, onPointsChange]);

  // Notificar cambios en los puntos (fuera del render)
  useEffect(() => {
    if (enabled && points.length > 0) {
      onPointsChangeRef.current?.(points);
    }
  }, [points, enabled]);

  useEffect(() => {
    if (!map || !enabled) {
      return;
    }

    // Cambiar cursor para indicar modo dibujo
    map.getContainer().style.cursor = "crosshair";

    const handleMapClick = (e) => {
      // Agregar punto
      const newPoint = { lat: e.latlng.lat, lon: e.latlng.lng };
      setPoints((prev) => [...prev, newPoint]);
    };

    const handleKeyDown = (e) => {
      // ESC para cancelar
      if (e.key === "Escape") {
        setPoints([]);
        onPointsChangeRef.current?.([]);
      }
      // Enter para terminar (si hay al menos 2 puntos)
      if (e.key === "Enter") {
        setPoints((prevPoints) => {
          if (prevPoints.length >= 2) {
            onCompleteRef.current?.(prevPoints);
          }
          return prevPoints; // Mantener los puntos visibles
        });
      }
      // Delete último punto con tecla Delete
      if (e.key === "Delete") {
        setPoints((prev) => {
          if (prev.length > 0) {
            const newPoints = prev.slice(0, -1);
            // Notificar al padre del cambio
            onPointsChangeRef.current?.(newPoints);
            return newPoints;
          }
          return prev;
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

  useEffect(() => {
    if (
      !map ||
      enabled ||
      points.length < 2 ||
      !Array.isArray(profilePoints) ||
      profilePoints.length === 0
    ) {
      return;
    }

    const hoverTolerancePx = 18;

    const handleMapMouseMove = (e) => {
      if (hoverFrameRef.current != null) {
        cancelAnimationFrame(hoverFrameRef.current);
      }

      hoverFrameRef.current = requestAnimationFrame(() => {
        const mousePoint = map.latLngToContainerPoint(e.latlng);
        const linePixels = points.map((point) =>
          map.latLngToContainerPoint([point.lat, point.lon]),
        );

        let nearLine = false;
        for (let i = 0; i < linePixels.length - 1; i += 1) {
          if (
            distanceToSegment(mousePoint, linePixels[i], linePixels[i + 1]) <=
            hoverTolerancePx
          ) {
            nearLine = true;
            break;
          }
        }

        if (!nearLine) {
          map.getContainer().style.cursor = "";
          if (lastHoveredRef.current !== null) {
            lastHoveredRef.current = null;
            onHoverPoint?.(null);
          }
          hoverFrameRef.current = null;
          return;
        }

        let nearestPoint = null;
        let nearestDistance = Infinity;

        profilePoints.forEach((point) => {
          if (!Number.isFinite(point?.lat) || !Number.isFinite(point?.lon)) {
            return;
          }

          const pointPx = map.latLngToContainerPoint([point.lat, point.lon]);
          const distance = Math.hypot(
            mousePoint.x - pointPx.x,
            mousePoint.y - pointPx.y,
          );
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestPoint = point;
          }
        });

        map.getContainer().style.cursor = "pointer";
        if (
          lastHoveredRef.current?.lat !== nearestPoint?.lat ||
          lastHoveredRef.current?.lon !== nearestPoint?.lon
        ) {
          lastHoveredRef.current = nearestPoint;
          onHoverPoint?.(nearestPoint);
        }
        hoverFrameRef.current = null;
      });
    };

    const handleMapMouseOut = () => {
      if (hoverFrameRef.current != null) {
        cancelAnimationFrame(hoverFrameRef.current);
        hoverFrameRef.current = null;
      }
      map.getContainer().style.cursor = "";
      if (lastHoveredRef.current !== null) {
        lastHoveredRef.current = null;
        onHoverPoint?.(null);
      }
    };

    map.on("mousemove", handleMapMouseMove);
    map.on("mouseout", handleMapMouseOut);

    return () => {
      if (hoverFrameRef.current != null) {
        cancelAnimationFrame(hoverFrameRef.current);
        hoverFrameRef.current = null;
      }
      map.off("mousemove", handleMapMouseMove);
      map.off("mouseout", handleMapMouseOut);
      map.getContainer().style.cursor = "";
    };
  }, [map, enabled, points, profilePoints, onHoverPoint]);

  // Función para finalizar el dibujo al hacer click en el marcador final
  const handleFinishClick = () => {
    if (points.length >= 2) {
      onCompleteRef.current?.(points);
      // No limpiamos los puntos - se mantienen visibles hasta cerrar el diálogo
    }
  };

  if (points.length === 0) return null;

  // Convertir puntos para Leaflet (formato [lat, lon])
  const positions = points.map((p) => [p.lat, p.lon]);

  return (
    <>
      {/* Línea conectando los puntos */}
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

      {/* Marcadores en cada punto (excepto el último) */}
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

      {/* Punto final visible cuando el dibujo ya terminó */}
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

      {/* Marcador final clickeable solo mientras el modo dibujo está activo */}
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
