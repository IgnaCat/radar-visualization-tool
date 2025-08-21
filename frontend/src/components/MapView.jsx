import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function COGTile({ tilejsonUrl, opacity = 0.9 }) {
  const map = useMap();
  const [template, setTemplate] = useState(null);
  const [llb, setLLB] = useState(null);
  const [nativeZooms, setNativeZooms] = useState({ min: 0, max: 22 });
  const abortRef = useRef(null);

  useEffect(() => {
    if (!tilejsonUrl) return;

    // cancelar fetch previo si cambia rápido
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    (async () => {
      try {
        const r = await fetch(tilejsonUrl, { signal: ctrl.signal });
        if (!r.ok) {
          const txt = await r.text();
          console.error("TileJSON error", r.status, txt.slice(0, 200));
          return;
        }
        const tj = await r.json();

        let url = tj.tiles?.[0];
        if (!url) {
          console.error("TileJSON sin 'tiles':", tj);
          return;
        }
        // prefijo /cog si falta
        if (url.includes("/tiles/") && !url.includes("/cog/tiles/")) {
          url = url.replace("/tiles/", "/cog/tiles/");
        }
        // cache-buster para que no mezcle tiles entre productos
        url +=
          (url.includes("?") ? "&" : "?") +
          "v=" +
          Date.now().toString().slice(-6);

        // zooms nativos
        const minN = Number.isFinite(tj.minzoom) ? tj.minzoom : 0;
        const maxN = Number.isFinite(tj.maxzoom) ? tj.maxzoom : 22;
        setNativeZooms({ min: minN, max: maxN });

        // bounds / center
        if (Array.isArray(tj.bounds) && tj.bounds.length === 4) {
          if (Array.isArray(tj.center) && tj.center.length === 3) {
            const [lon, lat, z] = tj.center;
            map.setView([lat, lon], z);
          }
          const [w, s, e, n] = tj.bounds;
          const bounds = [
            [s, w],
            [n, e],
          ];
          setLLB(bounds);

          // map.fitBounds(bounds, { padding: [20, 20] });

          map.setMaxBounds(bounds);
        }

        setTemplate(url);
      } catch (e) {
        if (e.name !== "AbortError") console.error("TileJSON fetch fail:", e);
      }
    })();

    return () => {
      ctrl.abort();
      // liberar maxBounds al cambiar de producto
      try {
        map.setMaxBounds(null);
      } catch {}
    };
  }, [tilejsonUrl, map]);

  // clave estable para forzar desmontaje limpio de la capa previa
  const layerKey = template
    ? `${template}|${nativeZooms.min}|${nativeZooms.max}`
    : "none";

  return template ? (
    <TileLayer
      key={layerKey}
      url={template}
      opacity={opacity}
      noWrap={true}
      bounds={llb}
      minNativeZoom={nativeZooms.min}
      maxNativeZoom={nativeZooms.max}
      // límites UI: permitir acercar sin pedir z fuera
      minZoom={Math.min(0, nativeZooms.min)}
      maxZoom={nativeZooms.max + 6}
      // suaviza animaciones y evita “fantasmas”
      updateWhenZooming={true}
      reuseTiles={false}
      keepBuffer={1}
      zIndex={500}
      // tile transparente en caso de error puntual
      errorTileUrl="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
      eventHandlers={{
        tileerror: (e) => console.warn("tileerror", e.coords, e.error),
      }}
      crossOrigin={"anonymous"}
    />
  ) : null;
}

export default function MapView({ overlayData }) {
  const center = useMemo(() => [-31.4, -64.2], []);
  return (
    <MapContainer
      center={center}
      zoom={6}
      style={{ height: "100vh", width: "100%" }}
      worldCopyJump={false}
      preferCanvas={false}
      fadeAnimation={false}
      zoomAnimation={true}
      markerZoomAnimation={true}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {overlayData?.tilejson_url && (
        <COGTile tilejsonUrl={overlayData.tilejson_url} opacity={0.95} />
      )}
    </MapContainer>
  );
}
