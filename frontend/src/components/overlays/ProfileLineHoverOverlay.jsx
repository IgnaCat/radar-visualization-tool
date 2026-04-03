import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";

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

/**
 * Overlay genérico para sincronizar una línea del mapa con un perfil muestreado.
 * No dibuja nada: solo escucha el mouse del mapa y resuelve qué punto del perfil
 * corresponde al tramo más cercano al cursor.
 */
export default function ProfileLineHoverOverlay({
  enabled = false,
  linePoints = [],
  profilePoints = [],
  onHoverPoint,
  hoverTolerancePx = 18,
}) {
  const map = useMap();
  const hoverFrameRef = useRef(null);
  const lastHoveredRef = useRef(undefined);

  useEffect(() => {
    if (
      !map ||
      !enabled ||
      !Array.isArray(linePoints) ||
      linePoints.length < 2 ||
      !Array.isArray(profilePoints) ||
      profilePoints.length === 0
    ) {
      return;
    }

    const handleMapMouseMove = (e) => {
      if (hoverFrameRef.current != null) {
        cancelAnimationFrame(hoverFrameRef.current);
      }

      hoverFrameRef.current = requestAnimationFrame(() => {
        const mousePoint = map.latLngToContainerPoint(e.latlng);

        // Reproyectamos la línea en cada frame para que el hover siga funcionando
        // incluso si el usuario hace zoom o pan mientras mueve el mouse.
        const linePixels = linePoints.map((point) =>
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

        // Buscamos la muestra del perfil más cercana al cursor en pantalla.
        // Esto mantiene sincronizados mapa y gráfico sin depender del estilo visual de la línea.
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

    const clearHover = () => {
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
    map.on("mouseout", clearHover);

    return () => {
      clearHover();
      map.off("mousemove", handleMapMouseMove);
      map.off("mouseout", clearHover);
    };
  }, [map, enabled, linePoints, profilePoints, onHoverPoint, hoverTolerancePx]);

  return null;
}
