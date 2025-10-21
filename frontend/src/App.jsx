import { useState, useEffect, useRef } from "react";
import { uploadFile, processFile, generatePseudoRHI } from "./api/backend";
import { registerCleanupAxios, cogFsPaths } from "./api/registerCleanupAxios";
import MapView from "./components/MapView";
import UploadButton from "./components/UploadButton";
import FloatingMenu from "./components/FloatingMenu";
import Alerts from "./components/Alerts";
import ColorLegend from "./components/ColorLegend";
import Loader from "./components/Loader";
import AnimationControls from "./components/AnimationControls";
import ProductSelectorDialog from "./components/ProductSelectorDialog";
import PseudoRHIDialog from "./components/PseudoRHIDialog";

export default function App() {
  const [overlayData, setOverlayData] = useState({
    outputs: [],
    animation: false,
    metadata: {},
  });
  const [opacity, setOpacity] = useState([0.95]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0); // índice de la imagen activa
  const [loading, setLoading] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [fieldsUsed, setFieldsUsed] = useState("DBZH");
  const [filesInfo, setFilesInfo] = useState([]);
  const [savedLayers, setSavedLayers] = useState([]); // layers / variables usadas
  const allCogsRef = useRef(new Set());
  const [showPlayButton, setShowPlayButton] = useState(false); // animacion
  var currentOverlay = overlayData.outputs?.[currentIndex] || null;

  const [alert, setAlert] = useState({
    open: false,
    message: "",
    severity: "info",
  });

  const [rhiOpen, setRhiOpen] = useState(false);
  const [pickPointMode, setPickPointMode] = useState(false);
  const [pickedPoint, setPickedPoint] = useState(null); // { lat, lon } seleccionado
  var radarSite = overlayData?.metadata?.site || null;

  // Registrar cleanup en cierre de pestaña/ventana
  useEffect(() => {
    const unregister = registerCleanupAxios(() => ({
      uploads: uploadedFiles,
      cogs: Array.from(allCogsRef.current),
      delete_cache: false,
    }));
    return unregister;
  }, [uploadedFiles, overlayData]);

  const handleFileUpload = () => {
    document.getElementById("upload-file").click();
  };

  const handleFilesSelected = async (files) => {
    try {
      setLoading(true);
      const uploadResp = await uploadFile(files);
      const warnings = uploadResp.data.warnings || [];
      const filesInfo = uploadResp.data.files || [];
      const filepaths = filesInfo.map((f) => f.filepath);

      if (filesInfo.length === 0) {
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
      setFilesInfo(filesInfo);
      setUploadedFiles((prev) => {
        const merged = [...prev, ...filepaths];
        // elimina duplicados preservando el orden
        return Array.from(new Set(merged));
      });
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

  const handleProductChosen = async (data) => {
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

      const files = uploadedFiles;
      const layers = data.layers;
      const product = data.product;
      const height = data.height;
      const elevation = data.elevation;
      const filters = data.filters;
      const enabledLayers = layers.filter((l) => l.enabled).map((l) => l.label);
      const opacities = layers.filter((l) => l.enabled).map((l) => l.opacity);

      setOpacity(opacities);
      setFieldsUsed(enabledLayers);
      setSavedLayers(data.layers);

      const processResp = await processFile({
        files,
        layers: enabledLayers,
        product,
        height,
        elevation,
        filters,
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

      console.log("Process response:", processResp.data);

      setOverlayData(processResp.data);
      setCurrentIndex(0);
      setShowPlayButton(
        processResp.data?.outputs && processResp.data.outputs.length > 1
      );

      // Guardar todos los cogs para el cleanup
      // const fromOutputs = cogFsPaths(processResp.data?.outputs || []);
      // fromOutputs.forEach((p) => allCogsRef.current.add(p));

      setAlert({
        open: true,
        message: `Mostrando ${data.product.toUpperCase()}`,
        severity: "success",
      });
    } catch (err) {
      console.error(err);
      setAlert({
        open: true,
        message: "Error al procesar producto",
        severity: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleOpenRHI = () => setRhiOpen(true);

  const handleRequestPickPoint = () => {
    setPickedPoint(null);
    setPickPointMode(true);
  };
  const handlePickPoint = (pt) => {
    setPickedPoint(pt);
    setPickPointMode(false); // se desactiva al elegir
  };
  const handleClearPickedPoint = () => setPickedPoint(null);

  const handleGenerateRHI = async ({
    filepath,
    field,
    end_lat,
    end_lon,
    // max_length_km,
    // elevation,
    // filters,
  }) => {
    const resp = await generatePseudoRHI({
      filepath,
      field,
      end_lat,
      end_lon,
    });
    // devolvemos lo que el dialog espera
    return resp.data;
  };

  return (
    <>
      <MapView
        overlayData={currentOverlay}
        opacities={opacity}
        pickPointMode={pickPointMode}
        radarSite={radarSite}
        pickedPoint={pickedPoint}
        onPickPoint={handlePickPoint}
      />
      <ColorLegend fields={fieldsUsed} />
      <FloatingMenu
        onUploadClick={handleFileUpload}
        onChangeProductClick={() => setSelectorOpen(true)}
        onPseudoRhiClick={handleOpenRHI}
      />
      <UploadButton onFilesSelected={handleFilesSelected} />

      {/* Slider para múltiples imágenes */}
      {overlayData?.outputs && overlayData?.outputs.length > 0 && (
        <AnimationControls
          overlayData={overlayData}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
          showPlayButton={showPlayButton}
        />
      )}

      <ProductSelectorDialog
        open={selectorOpen}
        fields_present={Array.from(
          new Set(filesInfo.map((f) => f.metadata.fields_present).flat())
        )}
        elevations={Array.from(
          new Set(filesInfo.map((f) => f.metadata.elevations).flat())
        )}
        initialLayers={savedLayers}
        onClose={() => setSelectorOpen(false)}
        onConfirm={handleProductChosen}
      />

      {/* Dialog Pseudo-RHI */}
      <PseudoRHIDialog
        open={rhiOpen}
        onClose={() => setRhiOpen(false)}
        filepath={uploadedFiles[currentIndex]}
        radarSite={radarSite}
        fields_present={
          Array.from(
            new Set(filesInfo.map((f) => f.metadata.fields_present).flat())
          ) || ["DBZH", "KDP", "RHOHV", "ZDR"]
        }
        onRequestPickPoint={handleRequestPickPoint}
        pickedPoint={pickedPoint}
        onClearPickedPoint={handleClearPickedPoint}
        onGenerate={handleGenerateRHI}
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
