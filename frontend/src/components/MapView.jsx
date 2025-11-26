import { useEffect, useMemo, useRef, useState } from "react";
import {
  MapContainer,
  TileLayer,
  useMap,
  CircleMarker,
  Tooltip,
  Polyline,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import MapPickOverlay from "./MapPickOverlay";
import AreaDrawOverlay from "./AreaDrawOverlay";
import UsePixelStatClick from "./UsePixelStatClick";

function COGTile({ tilejsonUrl, opacity, zIndex = 500 }) {
  const map = useMap();
  const [template, setTemplate] = useState(null);
  const [llb, setLLB] = useState(null);
  const [nativeZooms, setNativeZooms] = useState({ min: 0, max: 22 });
  const abortRef = useRef(null);
  const didCenter = useRef(false);

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
          if (
            Array.isArray(tj.center) &&
            tj.center.length === 3 &&
            !didCenter.current
          ) {
            const [lon, lat, z] = tj.center;
            map.setView([lat, lon], z);
            didCenter.current = true;
          }
          const [w, s, e, n] = tj.bounds;
          const bounds = [
            [s, w],
            [n, e],
          ];
          setLLB(bounds);

          // map.fitBounds(bounds, { padding: [20, 20] })
          // map.setMaxBounds(bounds);
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
      // suaviza animaciones y evita “fantasmas”
      updateWhenZooming={true}
      reuseTiles={false}
      keepBuffer={1}
      zIndex={zIndex}
      // tile transparente en caso de error puntual
      errorTileUrl="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
      eventHandlers={{
        tileerror: (e) => console.warn("tileerror", e.coords, e.error),
      }}
      crossOrigin={"anonymous"}
    />
  ) : null;
}

export default function MapView({
  overlayData,
  opacities = [0.95],
  opacityByField = {},
  pickPointMode = false,
  radarSite = null,
  pickedPoint = null,
  onPickPoint,
  drawAreaMode = false,
  onAreaComplete,
  pixelStatMode = false,
  onPixelStatClick,
  pixelStatMarker = null,
  lineOverlay = null,
  onClearLineOverlay,
  rhiEndpoints = null, // { start: {lat, lon}, end: {lat, lon} }
}) {
  const center = useMemo(() => [-31.4, -64.2], []);
  const baseZ = 500;
  // overlayData ahora puede ser un array de capas de distintos radares para el frame actual
  const n = overlayData?.length ?? 0;

  // Si pickedPoint se limpia, avisar al padre para limpiar la línea
  useEffect(() => {
    if (!pickedPoint && typeof onClearLineOverlay === "function") {
      onClearLineOverlay();
    }
  }, [pickedPoint]);

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

      {/* Mostrar todas las capas del frame actual (pueden ser de distintos radares) */}
      {Array.isArray(overlayData) &&
        overlayData.map((L, idx) => {
          const keyField = String(L.field || L.label || "").toUpperCase();
          const fieldOpacity =
            typeof opacityByField[keyField] === "number"
              ? opacityByField[keyField]
              : opacities[idx] ?? 1;
          return (
            <COGTile
              key={`${L.field || "layer"}|${L.tilejson_url}`}
              tilejsonUrl={L.tilejson_url}
              opacity={fieldOpacity}
              zIndex={baseZ + (n - 1 - idx) * 10}
            />
          );
        })}
      <MapPickOverlay
        enabled={pickPointMode}
        radarSite={radarSite}
        pickedPoint={pickedPoint}
        onPick={onPickPoint}
      />
      <AreaDrawOverlay
        enabled={drawAreaMode}
        onComplete={onAreaComplete}
        modes={{ polygon: true, rectangle: true }}
      />
      <UsePixelStatClick
        enabled={pixelStatMode}
        onPixelStatClick={onPixelStatClick}
      />
      {pixelStatMarker &&
        Number.isFinite(pixelStatMarker.lat) &&
        Number.isFinite(pixelStatMarker.lon) && (
          <CircleMarker
            center={[pixelStatMarker.lat, pixelStatMarker.lon]}
            radius={6}
            pathOptions={{ color: "#ff3b30", weight: 2, fillOpacity: 0.7 }}
          >
            <Tooltip direction="top" offset={[0, -6]} permanent>
              {pixelStatMarker.value == null
                ? "masked"
                : String(pixelStatMarker.value)}
            </Tooltip>
          </CircleMarker>
        )}
      {/* Marcadores persistentes para los puntos de RHI (inicio/fin) */}
      {rhiEndpoints?.start &&
        Number.isFinite(rhiEndpoints.start.lat) &&
        Number.isFinite(rhiEndpoints.start.lon) && (
          <CircleMarker
            center={[rhiEndpoints.start.lat, rhiEndpoints.start.lon]}
            radius={6}
            pathOptions={{ color: "#00aaff", weight: 2, fillOpacity: 0.7 }}
          />
        )}
      {rhiEndpoints?.end &&
        Number.isFinite(rhiEndpoints.end.lat) &&
        Number.isFinite(rhiEndpoints.end.lon) && (
          <CircleMarker
            center={[rhiEndpoints.end.lat, rhiEndpoints.end.lon]}
            radius={6}
            pathOptions={{ color: "#00aaff", weight: 2, fillOpacity: 0.7 }}
          />
        )}
      {Array.isArray(lineOverlay) && lineOverlay.length === 2 && (
        <Polyline
          positions={lineOverlay}
          pathOptions={{ color: "#00aaff", weight: 3, opacity: 0.9 }}
        />
      )}
      {/* Origen del radar (solo al elegir puntos para pseudo-RHI) */}
      {pickPointMode && radarSite && (
        <CircleMarker
          center={[radarSite.lat, radarSite.lon]}
          radius={7}
          pathOptions={{
            color: "#ff9800",
            weight: 3,
            fillOpacity: 0.9,
            fillColor: "#ff9800",
          }}
        >
          <Tooltip direction="top" offset={[0, -6]} permanent>
            Origen radar
          </Tooltip>
        </CircleMarker>
      )}
    </MapContainer>
  );
}
