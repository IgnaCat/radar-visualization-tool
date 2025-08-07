import { MapContainer, TileLayer, ImageOverlay } from "react-leaflet";
import "leaflet/dist/leaflet.css";

export default function MapView({ overlayData }) {
  return (
    <MapContainer
      center={[-34.6, -58.4]}
      zoom={5}
      style={{ height: "100vh", width: "100%" }}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {overlayData?.animation ? (
        // Si hay animación, mostrarla ocupando toda la región de Argentina
        <ImageOverlay
          url={`http://localhost:5000${overlayData.overlays[0].image_url}`}
          bounds={overlayData.overlays[0].bounds}
          opacity={0.8}
        />
      ) : (
        // Si no hay animación, mostrar cada overlay individual
        overlayData?.overlays?.map((overlay, idx) => (
          <ImageOverlay
            key={idx}
            url={`http://localhost:5000${overlay.image_url}`}
            bounds={overlay.bounds}
            opacity={0.8}
          />
        ))
      )}
    </MapContainer>
  );
}
