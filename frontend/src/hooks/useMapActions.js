import { useState, useCallback, useEffect } from "react";
import domtoimage from "dom-to-image-more";
import html2canvas from "html2canvas";

/**
 * Custom hook para manejar acciones del mapa
 * - Captura de pantalla (dom-to-image-more + html2canvas para tiles COG)
 * - Impresión
 * - Pantalla completa
 */
export function useMapActions() {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const isIgnoredCaptureElement = useCallback((node) => {
    if (!node?.classList) return false;

    return (
      node.classList.contains("no-print") ||
      node.classList.contains("leaflet-control-zoom")
    );
  }, []);

  // Detectar cambios en fullscreen (por ESC u otros métodos)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  /**
   * Espera a que todos los tiles del mapa se carguen
   * @param {HTMLElement} container - Contenedor del mapa
   * @param {number} timeout - Timeout en ms
   * @returns {Promise<void>}
   */
  const waitForTiles = useCallback((container, timeout = 3000) => {
    return new Promise((resolve) => {
      const startTime = Date.now();

      const checkTiles = () => {
        const tiles = container.querySelectorAll("img.leaflet-tile");
        const allLoaded = Array.from(tiles).every((tile) => tile.complete);

        if (allLoaded || Date.now() - startTime > timeout) {
          resolve();
        } else {
          setTimeout(checkTiles, 100);
        }
      };

      checkTiles();
    });
  }, []);

  const waitForPaint = useCallback(() => {
    return new Promise((resolve) => {
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(resolve);
      });
    });
  }, []);

  const captureWithDomToImage = useCallback(
    async (container) => {
      return await domtoimage.toPng(container, {
        quality: 1.0,
        bgcolor: "#ffffff",
        cacheBust: true,
        filter: (node) => !isIgnoredCaptureElement(node),
      });
    },
    [isIgnoredCaptureElement],
  );

  const captureWithHtml2Canvas = useCallback(
    async (container) => {
      const canvas = await html2canvas(container, {
        backgroundColor: "#ffffff",
        useCORS: true,
        allowTaint: false,
        logging: false,
        removeContainer: true,
        scale: Math.max(window.devicePixelRatio || 1, 2),
        ignoreElements: (element) => isIgnoredCaptureElement(element),
        onclone: (clonedDocument) => {
          clonedDocument
            .querySelectorAll(".no-print, .leaflet-control-zoom")
            .forEach((element) => {
              element.style.display = "none";
            });
        },
      });

      return canvas.toDataURL("image/png");
    },
    [isIgnoredCaptureElement],
  );

  const shouldPreferHtml2Canvas = useCallback((container) => {
    return Boolean(
      container.querySelector('img.leaflet-tile[src*="/cog/"]'),
    );
  }, []);

  const captureMapImage = useCallback(
    async (containerId = "map-container") => {
      try {
        const container = document.getElementById(containerId);
        if (!container) {
          throw new Error(`Contenedor "${containerId}" no encontrado`);
        }

        await waitForTiles(container);
        await waitForPaint();

        const elementsToHide = container.querySelectorAll(
          ".no-print, .leaflet-control-zoom",
        );
        elementsToHide.forEach((el) => {
          el.dataset.originalDisplay = el.style.display;
          el.style.display = "none";
        });

        try {
          if (shouldPreferHtml2Canvas(container)) {
            return await captureWithHtml2Canvas(container);
          }

          try {
            return await captureWithDomToImage(container);
          } catch (error) {
            console.warn(
              "dom-to-image no pudo exportar el mapa; usando html2canvas",
              error,
            );
            return await captureWithHtml2Canvas(container);
          }
        } finally {
          elementsToHide.forEach((el) => {
            el.style.display = el.dataset.originalDisplay || "";
            delete el.dataset.originalDisplay;
          });
        }
      } catch (error) {
        console.error("Error capturando imagen del mapa:", error);
        throw error;
      }
    },
    [
      captureWithDomToImage,
      captureWithHtml2Canvas,
      shouldPreferHtml2Canvas,
      waitForPaint,
      waitForTiles,
    ],
  );

  /**
   * Captura una pantalla del mapa usando dom-to-image-more
   * Mejor manejo de CORS y tiles dinámicos que leaflet-image
   * @param {L.Map} mapRef - Referencia al mapa de Leaflet (no usado, mantenido por compatibilidad)
   * @param {string} containerId - ID del contenedor a capturar (default: "map-container")
   */
  const handleScreenshot = useCallback(
    async (mapRef, containerId = "map-container") => {
      try {
        const dataUrl = await captureMapImage(containerId);

        const link = document.createElement("a");
        const timestamp = new Date()
          .toISOString()
          .replace(/[:.]/g, "-")
          .replace("T", "_")
          .substring(0, 19);
        link.download = `radar-map_${timestamp}.png`;
        link.href = dataUrl;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (error) {
        console.error("Error capturando pantalla:", error);
        throw error;
      }
    },
    [captureMapImage],
  );

  /**
   * Imprime el contenido del mapa
   */
  const handlePrint = useCallback(() => {
    try {
      window.print();
    } catch (error) {
      console.error("Error al imprimir:", error);
      throw error;
    }
  }, []);

  /**
   * Alterna el modo pantalla completa
   */
  const handleFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (error) {
      console.error("Error al cambiar pantalla completa:", error);
      throw error;
    }
  }, []);

  return {
    isFullscreen,
    captureMapImage,
    handleScreenshot,
    handlePrint,
    handleFullscreen,
  };
}
