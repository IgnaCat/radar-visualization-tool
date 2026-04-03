import { useEffect } from "react";
import { useMap } from "react-leaflet";

const LEAFLET_PREFIX_WITHOUT_FLAG =
  '<a href="https://leafletjs.com" title="A JavaScript library for interactive maps">Leaflet</a>';

/* Control para establecer el prefijo de atribución del mapa 
  Esconder bandera nativa de Leaflet modularmente*/
export default function AttributionPrefixControl({
  prefix = LEAFLET_PREFIX_WITHOUT_FLAG,
}) {
  const map = useMap();

  useEffect(() => {
    map.attributionControl?.setPrefix(prefix);
  }, [map, prefix]);

  return null;
}
