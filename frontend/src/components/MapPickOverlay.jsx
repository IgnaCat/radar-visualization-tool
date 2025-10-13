import { useEffect } from "react";
import { Marker, Polyline, useMapEvent } from "react-leaflet";

export default function MapPickOverlay({
  enabled,
  radarSite,
  pickedPoint,
  onPick,
}) {
  useMapEvent("click", (e) => {
    if (!enabled) return;
    onPick?.({ lat: e.latlng.lat, lon: e.latlng.lng });
  });

  if (!enabled && !pickedPoint) return null;

  const siteLatLng = radarSite ? [radarSite.lat, radarSite.lon] : null;
  const pp = pickedPoint ? [pickedPoint.lat, pickedPoint.lon] : null;

  return (
    <>
      {pp && <Marker position={pp} />}
      {pp && siteLatLng && <Polyline positions={[siteLatLng, pp]} />}
    </>
  );
}
