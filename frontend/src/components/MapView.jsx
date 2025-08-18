import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function TileJsonLayer({ tilejsonUrl, opacity = 0.85 }) {
  const [template, setTemplate] = useState(null);
  const [bounds, setBounds] = useState(null);
  const map = useMap();

  useEffect(() => {
    let abort = false;
    (async () => {
      if (!tilejsonUrl) return;

      const resp = await fetch(tilejsonUrl, { mode: "cors" });

      // Clonar para poder inspeccionar texto sin “gastar” el body
      const clone = resp.clone();
      const ct = resp.headers.get("content-type") || "";

      if (!resp.ok) {
        const txt = await clone.text();
        console.error("TileJSON HTTP error", resp.status, txt.slice(0, 200));
        return;
      }

      // Si no es JSON, logueo los primeros bytes para ver qué vino
      if (!ct.includes("application/json")) {
        const txt = await clone.text();
        console.error(
          "TileJSON no es JSON. CT:",
          ct,
          "Body:",
          txt.slice(0, 200)
        );
        return;
      }

      // Ahora sí parseo el original
      const tj = await resp.json();
      if (abort) return;

      if (!tj.tiles?.length) {
        console.error("TileJSON sin 'tiles':", tj);
        return;
      }

      let url = tj.tiles?.[0];
      if (url) {
        // fallback por si viene sin el prefijo
        url = url.replace("/tiles/", "/cog/tiles/");
        setTemplate(url);
      }

      // Ajustar zooms del mapa al rango del TileJSON
      if (tj.minzoom != null) map.setMinZoom(tj.minzoom);
      if (tj.maxzoom != null) map.setMaxZoom(tj.maxzoom);

      // Fit al bbox
      if (tj.bounds?.length === 4) {
        const [w, s, e, n] = tj.bounds;
        const llb = [
          [s, w],
          [n, e],
        ];
        setBounds(llb);
        map.fitBounds(llb, { padding: [20, 20] });
        // limita pan para que Leaflet no pida tiles fuera
        map.setMaxBounds(llb);
      }

      // Si trae center, úsalo (formato [lon, lat, zoom])
      if (Array.isArray(tj.center) && tj.center.length === 3) {
        const [lon, lat, z] = tj.center;
        map.setView([lat, lon], z);
      }
    })();

    return () => {
      abort = true;
    };
  }, [tilejsonUrl]);

  // ajustar vista cuando tengamos bounds
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [bounds, map]);

  return template ? (
    <TileLayer
      url={template}
      opacity={opacity}
      noWrap={true}
      bounds={bounds ?? undefined}
      minZoom={map.getMinZoom()}
      maxZoom={map.getMaxZoom()}
      updateWhenZooming={true}
    />
  ) : null;
}

export default function MapView({ overlayData }) {
  const center = useMemo(() => [-34.6, -58.4], []);

  return (
    <MapContainer
      center={center}
      zoom={5}
      style={{ height: "100vh", width: "105%" }}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {/* Si hay dato, pintamos el layer de TiTiler */}
      {overlayData?.tilejson_url && (
        <TileJsonLayer
          tilejsonUrl={overlayData.tilejson_url}
          opacity={0.9}
          noWrap={true}
        />
      )}
    </MapContainer>
  );
}
