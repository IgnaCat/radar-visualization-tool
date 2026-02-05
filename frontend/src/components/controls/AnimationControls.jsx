import { useEffect, useState, useMemo } from "react";
import { Box, Slider, IconButton } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";

export default function AnimationControls({
  overlayData,
  currentIndex,
  setCurrentIndex,
  showPlayButton = false,
}) {
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let interval = null;
    if (isPlaying && overlayData.outputs.length > 1) {
      interval = setInterval(() => {
        setCurrentIndex((prev) =>
          prev < overlayData.outputs.length - 1 ? prev + 1 : 0,
        );
      }, 1300);
    } else if (!isPlaying && interval !== null) {
      clearInterval(interval);
    }

    return () => clearInterval(interval);
  }, [isPlaying, overlayData]);

  // currentOverlay ahora es un array de capas (de distintos radares) para el frame actual
  const currentOverlay = overlayData.outputs[currentIndex];

  // Extraer información del frame actual
  const frameInfo = useMemo(() => {
    if (!Array.isArray(currentOverlay) || currentOverlay.length === 0) {
      return {
        timestamp: null,
        product: null,
        radars: [],
        strategies: [],
        volumes: [],
      };
    }

    // Timestamp (el más temprano del frame)
    const timestamp = currentOverlay
      .map((l) => l.timestamp)
      .filter(Boolean)
      .sort()[0];

    // Producto del overlayData
    const product = overlayData.product || null;

    // Extraer radar de cada layer (ya viene anotado desde mergeRadarFrames)
    const radars = [
      ...new Set(currentOverlay.map((l) => l.radar).filter(Boolean)),
    ];

    // Extraer estrategia y volumen del source_file (formato: RADAR_ESTRATEGIA_VOLUMEN_TIMESTAMP.nc)
    const strategies = new Set();
    const volumes = new Set();

    currentOverlay.forEach((layer) => {
      if (layer.source_file) {
        const filename = layer.source_file.split("/").pop().split("\\").pop();
        const parts = filename.split("_");
        if (parts.length >= 4) {
          strategies.add(parts[1]); // estrategia
          volumes.add(parts[2]); // volumen
        }
      }
    });

    return {
      timestamp,
      product,
      radars,
      strategies: Array.from(strategies),
      volumes: Array.from(volumes),
    };
  }, [currentOverlay, overlayData.product]);

  // Buscar el timestamp más representativo del frame (el menor, si hay varios)
  let frameTimestamp = null;
  if (Array.isArray(currentOverlay) && currentOverlay.length > 0) {
    // Preferir el timestamp más temprano del frame
    frameTimestamp = currentOverlay
      .map((l) => l.timestamp)
      .filter(Boolean)
      .sort()[0];
  }

  useEffect(() => {
    // Solo resetear si la cantidad de frames cambió
    setCurrentIndex(0);
    // eslint-disable-next-line
  }, [overlayData.outputs?.length]);

  return (
    <Box
      position="absolute"
      display="flex"
      alignItems="center"
      justifyContent="center"
      bottom={0}
      left="50%"
      sx={{
        transform: "translateX(-50%)",
        width: "50%",
        py: 5,
        zIndex: 900,
      }}
    >
      {showPlayButton && (
        <Box
          width={"35%"}
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          mb={1}
        >
          <Slider
            value={currentIndex}
            onChange={(e, val) => setCurrentIndex(val)}
            step={1}
            min={0}
            max={overlayData.outputs.length - 1}
            marks
          />

          {overlayData.animation && (
            <Box mt={1} gap={2} display="flex">
              <IconButton
                onClick={() =>
                  setCurrentIndex((prev) => (prev > 0 ? prev - 1 : 0))
                }
                sx={{
                  backgroundColor: "#42A5F5",
                  color: "white",
                  "&:hover": { backgroundColor: "#1E88E5" },
                }}
              >
                <SkipPreviousIcon />
              </IconButton>
              <IconButton
                onClick={() => setIsPlaying(!isPlaying)}
                sx={{
                  backgroundColor: "#42A5F5",
                  color: "white",
                  "&:hover": { backgroundColor: "#1E88E5" },
                }}
              >
                {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
              </IconButton>
              <IconButton
                onClick={() =>
                  setCurrentIndex((prev) =>
                    prev < overlayData.outputs.length - 1 ? prev + 1 : 0,
                  )
                }
                sx={{
                  backgroundColor: "#42A5F5",
                  color: "white",
                  "&:hover": { backgroundColor: "#1E88E5" },
                }}
              >
                <SkipNextIcon />
              </IconButton>
            </Box>
          )}
        </Box>
      )}
      <Box
        position="absolute"
        bottom={0}
        left={0}
        width="100%"
        bgcolor="white"
        py={1}
        px={2}
        textAlign="center"
        fontSize="0.875rem"
        fontFamily="Roboto, sans-serif"
        boxShadow="0 -1px 4px rgba(0,0,0,0.2)"
        borderRadius={3}
        color="black"
        zIndex={999}
      >
        <Box>
          {frameInfo.product && (
            <Box mt={0.5}>
              Vista: {frameInfo.product.toUpperCase()}
              {frameInfo.radars.length > 0 && (
                <> | Radar: {frameInfo.radars.join(", ")}</>
              )}
              {frameInfo.strategies.length > 0 && (
                <> | Estrategia: {frameInfo.strategies.join(", ")}</>
              )}
              {frameInfo.volumes.length > 0 && (
                <> | Volumen: {frameInfo.volumes.join(", ")}</>
              )}
            </Box>
          )}
          Fecha: {frameTimestamp || `Imagen ${currentIndex + 1}`} (Frame{" "}
          {currentIndex + 1} de {overlayData.outputs.length})
        </Box>
      </Box>
    </Box>
  );
}
