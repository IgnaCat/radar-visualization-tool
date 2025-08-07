import { useState } from "react";
import MapView from "./components/MapView";
import UploadButton from "./components/UploadButton";
import FloatingMenu from "./components/FloatingMenu";
import Alerts from "./components/Alerts";
import ColorLegend from "./components/ColorLegend";
import Loader from "./components/Loader";
import { uploadFile, processFile } from "./api/backend";

export default function App() {
  const [overlayData, setOverlayData] = useState(null);
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
      setOverlayData({
        overlays: processResp.data.outputs || [],
        animation: processResp.data.animation || null,
      });

      setAlert({
        open: true,
        message: "Archivo procesado correctamente",
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

  return (
    <>
      <MapView overlayData={overlayData} />
      <ColorLegend />
      <FloatingMenu onUploadClick={handleFileUpload} />
      <UploadButton onFilesSelected={handleFilesSelected} />
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
