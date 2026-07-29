import { cleanupClose, getApiBaseUrl, getAuthToken } from "./backend";

export function registerCleanupAxios(getPayload) {
  const handler = () => {
    try {
      const payload = getPayload?.();
      if (!payload) return;

      const baseUrl = getApiBaseUrl().replace(/\/+$/, "");
      const url = `${baseUrl}/cleanup/close`;
      const token = getAuthToken();

      // Prefer fetch con keepalive porque soporta headers (Auth)
      if (typeof window !== "undefined" && window.fetch) {
        const headers = { "Content-Type": "application/json" };
        if (token) {
          headers["Authorization"] = `Bearer ${token}`;
        }

        fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
          keepalive: true,
        }).catch(() => {});
        return;
      }

      // Fallback: axios (puede cortarse si el navegador cierra muy rápido)
      // No await: fire-and-forget
      cleanupClose(payload).catch(() => {});
    } catch {
      /* ignore */
    }
  };

  // cubrir más casos de cierre
  window.addEventListener("pagehide", handler);
  window.addEventListener("beforeunload", handler);

  return () => {
    window.removeEventListener("pagehide", handler);
    window.removeEventListener("beforeunload", handler);
  };
}

// toma outputs = overlayData.outputs y devuelve rutas en FS para borrar
export function cogFsPaths(outputs) {
  return (outputs || [])
    .map((o) => o?.image_url) // "static/tmp/radar_...tif"
    .filter(Boolean)
    .map((rel) => {
      // convertir URL estática a ruta de FS
      const file = rel.replace(/^static\/tmp\//, "");
      return `app/storage/tmp/${file}`; // ruta que el server puede borrar
    });
}
