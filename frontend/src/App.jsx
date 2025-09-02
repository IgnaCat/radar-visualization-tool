import { useState } from "react";
import { uploadFile, processFile } from "./api/backend";
import MapView from "./components/MapView";
import UploadButton from "./components/UploadButton";
import FloatingMenu from "./components/FloatingMenu";
import Alerts from "./components/Alerts";
import ColorLegend from "./components/ColorLegend";
import Loader from "./components/Loader";
import AnimationControls from "./components/AnimationControls";
import ProductSelectorDialog from "./components/ProductSelectorDialog";

export default function App() {
  const [overlayData, setOverlayData] = useState({
    outputs: [],
    animation: false,
  });
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [currentProduct, setCurrentProduct] = useState("PPI");

  const [alert, setAlert] = useState({
    open: false,
    message: "",
    severity: "info",
  });

  const handleFileUpload = () => {
    document.getElementById("upload-file").click();
  };

  const handleFilesSelected = async (files) => {
    try {
      setLoading(true);
      const uploadResp = await uploadFile(files);
      const warnings = uploadResp.data.warnings || [];
      const filepaths = uploadResp.data.filepaths || [];

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
      setUploadedFiles((prev) => [...prev, ...filepaths]);
      setSelectorOpen(true);
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

  const handleProductChosen = async (product) => {
    if (!uploadedFiles || uploadedFiles.length === 0) {
      setAlert({
        open: true,
        message: "No hay archivos para procesar",
        severity: "error",
      });
      return;
    }
    try {
      setLoading(true);

      const processResp = await processFile({
        filepaths: uploadedFiles,
        product,
      });
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

      setOverlayData(processResp.data);

      setCurrentIndex(0);

      setAlert({
        open: true,
        message: `Mostrando ${product.toUpperCase()}`,
        severity: "success",
      });
    } catch (err) {
      setAlert({
        open: true,
        message: "Error al procesar producto",
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
      <FloatingMenu
        onUploadClick={handleFileUpload}
        onChangeProductClick={() => setSelectorOpen(true)}
      />
      <UploadButton onFilesSelected={handleFilesSelected} />

      {/* Slider para múltiples imágenes */}
      {overlayData.outputs && overlayData.outputs.length > 1 && (
        <AnimationControls
          overlayData={overlayData}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
        />
      )}

      <ProductSelectorDialog
        open={selectorOpen}
        onClose={() => setSelectorOpen(false)}
        onConfirm={handleProductChosen}
      />

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
