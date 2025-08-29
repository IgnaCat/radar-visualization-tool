import { useMemo, useEffect, useRef } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import parseGeoraster from "georaster";
import GeoRasterLayer from "georaster-layer-for-leaflet";

function GeoTIFFLayerComponent({ url }) {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    if (!url) return;

    // limpiamos la capa anterior
    if (layerRef.current) {
      map.removeLayer(layerRef.current);
      layerRef.current = null;
    }

    let active = true;

    fetch(url)
      .then((res) => res.arrayBuffer())
      .then(parseGeoraster)
      .then((georaster) => {
        if (!active) return;

        const newLayer = new GeoRasterLayer({
          georaster,
          opacity: 1,
          resolution: 128,
        });
        newLayer.addTo(map);
        map.fitBounds(newLayer.getBounds());
        layerRef.current = newLayer;
      });

    return () => {
      active = false;
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
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
          key={overlayData.image_url}
          url={"http://localhost:8000/" + overlayData.image_url}
        />
      )}
    </MapContainer>
  );
}
