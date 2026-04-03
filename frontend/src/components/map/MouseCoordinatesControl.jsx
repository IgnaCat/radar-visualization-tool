import { useEffect } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";

const DEFAULT_PRECISION = 5;

function formatCoordinate(value, precision) {
  return Number.isFinite(value) ? value.toFixed(precision) : "-";
}

function formatLatLng(latlng, precision) {
  if (!latlng) {
    return "Lat: -, Lon: -";
  }

  return `Lat: ${formatCoordinate(latlng.lat, precision)}, Lon: ${formatCoordinate(latlng.lng, precision)}`;
}

export default function MouseCoordinatesControl({
  position = "bottomright",
  precision = DEFAULT_PRECISION,
}) {
  const map = useMap();

  useEffect(() => {
    const coordinatesControl = L.control({ position });

    coordinatesControl.onAdd = () => {
      const container = L.DomUtil.create(
        "div",
        "leaflet-control leaflet-control-coordinates",
      );

      container.setAttribute("aria-live", "polite");
      container.textContent = formatLatLng(null, precision);

      return container;
    };

    coordinatesControl.addTo(map);

    const updateCoordinates = (latlng) => {
      const container = coordinatesControl.getContainer();
      if (!container) return;

      container.textContent = formatLatLng(latlng, precision);
    };

    const handleMouseMove = (event) => {
      updateCoordinates(event.latlng);
    };

    const handleMouseLeave = () => {
      updateCoordinates(null);
    };

    map.on("mousemove", handleMouseMove);
    map.on("mouseout", handleMouseLeave);

    return () => {
      map.off("mousemove", handleMouseMove);
      map.off("mouseout", handleMouseLeave);
      coordinatesControl.remove();
    };
  }, [map, position, precision]);

  return null;
}
