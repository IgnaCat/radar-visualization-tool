import { useState, useEffect } from "react";
import MapView from "./components/MapView";
import UploadButton from "./components/UploadButton";
import FloatingMenu from "./components/FloatingMenu";
import Alerts from "./components/Alerts";
import ColorLegend from "./components/ColorLegend";
import Loader from "./components/Loader";
import { uploadFile, processFile } from "./api/backend";
import { Box, Slider } from "@mui/material";
import { IconButton } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";

export default function App() {
  const [overlayData, setOverlayData] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [alert, setAlert] = useState({
    open: false,
    message: "",
    severity: "info",
  });
  const [loading, setLoading] = useState(false);

  const handleFileUpload = () => {
    document.getElementById("upload-file").click();
  };

  const handleFilesSelected = async (files) => {
    try {
      setLoading(true);
      const uploadResp = await uploadFile(files);
      const filepaths = uploadResp.data.filepaths;

      const processResp = await processFile(filepaths);
      if (!processResp.data || !processResp.data.image_url) {
        setAlert({
          open: true,
          message: "No se encontraron imágenes procesadas",
          severity: "warning",
        });
      }
      console.log(processResp.data + "response");
      setOverlayData(processResp.data);
      setCurrentIndex(0);
      setAlert({
        open: true,
        message: "Archivos procesados correctamente",
        severity: "success",
      });
    } catch (err) {
      setAlert({
        open: true,
        message: err.response?.data?.error || "Error",
        severity: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  const currentOverlay = overlayData.outputs?.[currentIndex] || null;

  useEffect(() => {
    if (overlayData && overlayData.outputs && overlayData.outputs.length > 0) {
      setCurrentIndex(0);
    }
  }, [overlayData]);

  useEffect(() => {
    let interval = null;
    if (isPlaying && overlayData.outputs.length > 1) {
      interval = setInterval(() => {
        setCurrentIndex((prev) =>
          prev < overlayData.outputs.length - 1 ? prev + 1 : 0
        );
      }, 1000); // cada 1 segundo
    } else if (!isPlaying && interval !== null) {
      clearInterval(interval);
    }

    return () => clearInterval(interval);
  }, [isPlaying, overlayData]);

  return (
    <>
      <MapView overlayData={currentOverlay} />
      <ColorLegend />
      <FloatingMenu onUploadClick={handleFileUpload} />
      <UploadButton onFilesSelected={handleFilesSelected} />

      {/* Slider para múltiples imágenes */}
      {overlayData.outputs && overlayData.outputs.length > 1 && (
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
            {/* Botones de reproducción y pausa */}
            {overlayData.animation && (
              <Box mt={1} gap={2} display="flex">
                <IconButton
                  color="primary"
                  component="span"
                  onClick={() =>
                    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : 0))
                  }
                  sx={{
                    backgroundColor: "#42A5F5", // fondo azul claro
                    color: "white", // color del ícono
                    "&:hover": {
                      backgroundColor: "#1E88E5", // color más oscuro al hacer hover
                    },
                  }}
                >
                  <SkipPreviousIcon />
                </IconButton>
                <IconButton
                  color="primary"
                  component="span"
                  onClick={() => setIsPlaying(!isPlaying)}
                  sx={{
                    backgroundColor: "#42A5F5", // fondo azul claro
                    color: "white", // color del ícono
                    "&:hover": {
                      backgroundColor: "#1E88E5", // color más oscuro al hacer hover
                    },
                  }}
                >
                  {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                </IconButton>
                <IconButton
                  color="primary"
                  component="span"
                  onClick={() =>
                    setCurrentIndex((prev) =>
                      prev < overlayData.outputs.length - 1 ? prev + 1 : 0
                    )
                  }
                  sx={{
                    backgroundColor: "#42A5F5", // fondo azul claro
                    color: "white", // color del ícono
                    "&:hover": {
                      backgroundColor: "#1E88E5", // color más oscuro al hacer hover
                    },
                  }}
                >
                  <SkipNextIcon />
                </IconButton>
              </Box>
            )}
          </Box>

          {/* Texto del timestamp en un bloque abajo del todo */}
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
            Mostrando:{" "}
            {currentOverlay?.timestamp || `Imagen ${currentIndex + 1}`} (Frame{" "}
            {currentIndex + 1} de {overlayData.outputs.length})
          </Box>
        </Box>
      )}

      <Alerts
        open={alert.open}
        message={alert.message}
        severity={alert.severity}
        onClose={() => setAlert({ ...alert, open: false })}
      />
      <Loader open={loading} />
    </>
  );
}
