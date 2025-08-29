import { useState } from "react";
import MapView from "./components/MapView";
import UploadButton from "./components/UploadButton";
import FloatingMenu from "./components/FloatingMenu";
import Alerts from "./components/Alerts";
import ColorLegend from "./components/ColorLegend";
import Loader from "./components/Loader";
import { uploadFile, processFile } from "./api/backend";
import AnimationControls from "./components/AnimationControls";

export default function App() {
  const [overlayData, setOverlayData] = useState({
    outputs: [],
    animation: false,
  });
  const [currentIndex, setCurrentIndex] = useState(0);
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
      const filepaths = uploadResp.data.filepaths || [];
      const warnings = uploadResp.data.warnings || [];

      if (filepaths.length === 0) {
        setAlert({
          open: true,
          message: "No se encontraron archivos válidos\n" + warnings.join("\n"),
          severity: "warning",
        });
        return;
      }
      if (warnings.length > 0) {
        setAlert({
          open: true,
          message: warnings.join("\n"),
          severity: "warning",
        });
      }

      const processResp = await processFile(filepaths);
      if (
        !processResp.data ||
        !processResp.data.outputs ||
        processResp.data.outputs.length === 0
      ) {
        setAlert({
          open: true,
          message: "No se encontraron imágenes procesadas",
          severity: "warning",
        });
      }

      // Mantenemos datos previos
      setOverlayData((prev) => ({
        ...prev,
        animation: processResp.data.animation ?? prev.animation,
        outputs: [...(prev.outputs || []), ...(processResp.data.outputs || [])],
      }));

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

  var currentOverlay = overlayData.outputs?.[currentIndex] || null;

  return (
    <>
      <MapView overlayData={currentOverlay} />
      <ColorLegend />
      <FloatingMenu onUploadClick={handleFileUpload} />
      <UploadButton onFilesSelected={handleFilesSelected} />

      {/* Slider para múltiples imágenes */}
      {overlayData.outputs && overlayData.outputs.length > 1 && (
        <AnimationControls
          overlayData={overlayData}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
        />
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
