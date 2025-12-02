import { useState, useCallback, useEffect } from "react";
import html2canvas from "html2canvas";

/**
 * Custom hook para manejar acciones del mapa
 * - Captura de pantalla
 * - Impresión
 * - Pantalla completa
 */
export function useMapActions() {
    const [isFullscreen, setIsFullscreen] = useState(false);

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
     * Captura una pantalla del contenedor especificado
     * @param {string} containerId - ID del elemento a capturar
     */
    const handleScreenshot = useCallback(async (containerId = "map-container") => {
        try {
            const element = document.getElementById(containerId);
            if (!element) {
                console.error(`Elemento con ID "${containerId}" no encontrado`);
                return;
            }

            const canvas = await html2canvas(element, {
                useCORS: true,
                allowTaint: true,
                backgroundColor: "#ffffff",
                scale: 2, // Mayor calidad
            });

            // Crear enlace de descarga
            const link = document.createElement("a");
            const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
            link.download = `radar-screenshot-${timestamp}.png`;
            link.href = canvas.toDataURL("image/png");
            link.click();
        } catch (error) {
            console.error("Error capturando pantalla:", error);
            throw error;
        }
    }, []);

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
                // Entrar en pantalla completa
                await document.documentElement.requestFullscreen();
                setIsFullscreen(true);
            } else {
                // Salir de pantalla completa
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
        handleScreenshot,
        handlePrint,
        handleFullscreen,
    };
}
