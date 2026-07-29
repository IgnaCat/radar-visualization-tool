import { useEffect, useRef, useState } from "react";
import { getApiBaseUrl } from "../api/backend";

// Timeout largo porque el operador W puede tardar hasta ~60 s.
// Si el servidor crashea, fetch() falla al instante (ECONNREFUSED),
// así que este timeout solo aplica al caso "servidor vivo pero ocupado".
const REQUEST_TIMEOUT_MS = 65_000;

// Intervalo entre pings. Al ser mayor que el timeout normal de respuesta
// evitamos acumular requests cuando el servidor está rápido.
const POLL_INTERVAL_MS = 10_000;

// Fallos consecutivos necesarios para declarar el servidor offline.
// Un crash real produce ECONNREFUSED instantáneo → se detecta en ~20 s.
// El operador W tarda hasta 60 s → un solo ping cuelga pero no falla → no dispara.
const FAILURE_THRESHOLD = 2;

/**
 * Polls GET /health every POLL_INTERVAL_MS.
 *
 * Returns { isOnline: boolean }
 *
 * Side-effect: cuando el servidor vuelve después de un corte confirmado
 * (FAILURE_THRESHOLD fallos consecutivos), recarga la página para re-sincronizar
 * el estado de la sesión con el servidor reiniciado.
 */
export function useBackendHealth() {
  const [isOnline, setIsOnline] = useState(true);
  const [failCount, setFailCount] = useState(0); // fallos consecutivos visibles en UI
  const wasOfflineRef = useRef(false);
  const failCountRef = useRef(0);
  const pingInProgressRef = useRef(false); // evita solapamiento de requests

  useEffect(() => {
    let cancelled = false;

    async function ping() {
      // Si el ping anterior todavía está colgado (servidor ocupado), saltamos
      // este ciclo para no acumular requests pendientes.
      if (pingInProgressRef.current) {
        console.warn(
          "[health] Ping anterior aún en curso (servidor ocupado) — se omite este ciclo"
        );
        return;
      }

      pingInProgressRef.current = true;
      const url = `${getApiBaseUrl()}/health`;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      try {
        const res = await fetch(url, { signal: controller.signal });
        clearTimeout(timer);

        if (cancelled) return;

        if (res.ok) {
          const wasDown = wasOfflineRef.current;
          if (failCountRef.current > 0) {
            console.info(
              `[health] Servidor recuperado tras ${failCountRef.current} fallo(s) — reconectado`
            );
          }
          failCountRef.current = 0;
          setFailCount(0);

          if (wasDown) {
            console.info("[health] Servidor vuelto a estar online — recargando página");
            window.location.reload();
            return;
          }
          setIsOnline(true);
        } else {
          throw new Error(`HTTP ${res.status}`);
        }
      } catch (err) {
        clearTimeout(timer);
        if (cancelled) return;

        const isTimeout = err?.name === "AbortError";
        const reason = isTimeout
          ? `timeout (>${REQUEST_TIMEOUT_MS / 1000}s) — posiblemente construyendo operador W`
          : err?.message ?? "error de red";

        failCountRef.current += 1;
        setFailCount(failCountRef.current);
        console.warn(
          `[health] Fallo consecutivo #${failCountRef.current} (umbral: ${FAILURE_THRESHOLD}): ${reason}`
        );

        if (failCountRef.current >= FAILURE_THRESHOLD) {
          if (!wasOfflineRef.current) {
            console.error(
              `[health] Servidor declarado OFFLINE tras ${failCountRef.current} fallos consecutivos`
            );
          }
          wasOfflineRef.current = true;
          setIsOnline(false);
        }
      } finally {
        pingInProgressRef.current = false;
      }
    }

    ping();
    const id = setInterval(ping, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return { isOnline, failCount };
}
