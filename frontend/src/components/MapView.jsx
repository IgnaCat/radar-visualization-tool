import { useMemo, useEffect } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import parseGeoraster from "georaster";
import GeoRasterLayer from "georaster-layer-for-leaflet";

function GeoTIFFLayerComponent({ url }) {
  const map = useMap();

  useEffect(() => {
    if (!url) return;

    let layer;

    fetch(url)
      .then((res) => res.arrayBuffer())
      .then(parseGeoraster)
      .then((georaster) => {
        layer = new GeoRasterLayer({
          georaster,
          opacity: 1.0,
          resolution: 128,
        });
        layer.addTo(map);
        map.fitBounds(layer.getBounds()); // ajusta el mapa a la extensión del TIFF
      });

    return () => {
      if (layer) map.removeLayer(layer);
    };
  }, [url, map]);

  return null;
}

export default function MapView({ overlayData }) {
  const center = useMemo(() => [-31.4, -64.2], []);

  return (
    <MapContainer
      center={center}
      zoom={6}
      style={{ height: "100vh", width: "100%" }}
    >
      {/* base map */}
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {/* capa con geotiff */}
      {overlayData?.image_url && (
        <GeoTIFFLayerComponent
          url={"http://localhost:8000/" + overlayData.image_url}
        />
      )}
    </MapContainer>
  );
}
