import { useEffect } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";

/* Control para mostrar la escala del mapa */
export default function ScaleControl({
  position = "bottomright",
  maxWidth = 120,
  metric = true,
  imperial = false,
}) {
  const map = useMap();

  useEffect(() => {
    const scaleControl = L.control.scale({
      position,
      maxWidth,
      metric,
      imperial,
    });

    scaleControl.addTo(map);

    return () => {
      scaleControl.remove();
    };
  }, [imperial, map, maxWidth, metric, position]);

  return null;
}
