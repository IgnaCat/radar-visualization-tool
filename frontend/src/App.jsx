import { useState, useEffect, useRef } from "react";
import { useSnackbar } from "notistack";
import {
  uploadFile,
  processFile,
  generatePseudoRHI,
  generateAreaStats,
  generatePixelStat,
} from "./api/backend";
import { registerCleanupAxios, cogFsPaths } from "./api/registerCleanupAxios";
import stableStringify from "json-stable-stringify";
import MapView from "./components/MapView";
import UploadButton from "./components/UploadButton";
import FloatingMenu from "./components/FloatingMenu";
import Alerts from "./components/Alerts";
import ColorLegend from "./components/ColorLegend";
import Loader from "./components/Loader";
import AnimationControls from "./components/AnimationControls";
import ProductSelectorDialog from "./components/ProductSelectorDialog";
import PseudoRHIDialog from "./components/PseudoRHIDialog";
import WarningPanel from "./components/WarningPanel";
import AreaStatsDialog from "./components/AreaStatsDialog";

// Utilidad para combinar frames de múltiples radares por timestamp
function mergeRadarFrames(results) {
  // results: [{radar, outputs: [[LayerResult, ...], ...]}, ...]
  // Salida: [{ timestamp, layers: [LayerResult, ...] } ...] sincronizado por tiempo
  const frameMap = new Map(); // key: timestamp.toISOString() || 'null', value: [LayerResult, ...]

  results.forEach((radarResult) => {
    radarResult.outputs.forEach((layers, idx) => {
      // Tomar timestamp del primer layer (todos los de un frame tienen el mismo)
      const ts = layers[0]?.timestamp || null;
      const key = ts || `null_${idx}_${radarResult.radar}`;
      if (!frameMap.has(key)) frameMap.set(key, []);
      frameMap.get(key).push(...layers);
    });
  });

  // Ordenar por timestamp (nulls al final)
  const sorted = Array.from(frameMap.entries()).sort((a, b) => {
    if (a[0].startsWith("null")) return 1;
    if (b[0].startsWith("null")) return -1;
    return new Date(a[0]) - new Date(b[0]);
  });
  // Salida: array de arrays de LayerResult (cada frame puede tener varias capas de distintos radares)
  return sorted.map(([key, layers]) => layers);
}

function buildComputeKey({
  files,
  product,
  fields,
  elevation,
  height,
  filters,
  selectedVolumes,
}) {
  return stableStringify({
    files,
    product,
    fields,
    elevation,
    height,
    filters,
    selectedVolumes,
  });
}

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
  const [volumes, setVolumes] = useState([]); // lista de volúmenes cargados sin repetidos
  const [savedLayers, setSavedLayers] = useState([]); // layers / variables usadas
  const [filtersUsed, setFiltersUsed] = useState([]); // filtros aplicados
  const [activeElevation, setActiveElevation] = useState(null);
  const [activeHeight, setActiveHeight] = useState(null);
  const allCogsRef = useRef(new Set());
  const [showPlayButton, setShowPlayButton] = useState(false); // animacion
  const [computeKey, setComputeKey] = useState("");
  const [warnings, setWarnings] = useState([]);
  // var currentOverlay = overlayData.outputs?.[currentIndex] || null;

  const [alert, setAlert] = useState({
    open: false,
    message: "",
    severity: "info",
  });

  const { enqueueSnackbar } = useSnackbar();
  const [pixelStatMode, setPixelStatMode] = useState(false);
  const [pixelStatMarker, setPixelStatMarker] = useState(null);
  const [rhiOpen, setRhiOpen] = useState(false);
  const [pickPointMode, setPickPointMode] = useState(false);
  const [pickedPoint, setPickedPoint] = useState(null); // { lat, lon } seleccionado
  const [areaDrawMode, setAreaDrawMode] = useState(false);
  const [areaPolygon, setAreaPolygon] = useState(null);
  const [areaStatsOpen, setAreaStatsOpen] = useState(false);
  const drawnLayerRef = useRef(null); // referencia a la capa dibujada
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
      setVolumes((prev) => {
        const merged = [...prev, ...uploadResp.data.volumes];
        return Array.from(new Set(merged));
      });
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
      const selectedVolumes = data.selectedVolumes;
      const enabledLayers = layers.filter((l) => l.enabled).map((l) => l.label);
      const opacities = layers.filter((l) => l.enabled).map((l) => l.opacity);

      setOpacity(opacities);
      setFieldsUsed(enabledLayers);
      setSavedLayers(data.layers);
      setFiltersUsed(filters);
      if (elevation !== undefined) setActiveElevation(elevation);
      if (height !== undefined) setActiveHeight(height);

      const nextKey = buildComputeKey({
        files: uploadedFiles,
        product,
        fields: enabledLayers,
        elevation,
        height,
        filters,
        selectedVolumes,
      });

      // Si solo cambió UI (opacidad/orden), no reproceses:
      if (nextKey === computeKey) {
        return;
      }

      const processResp = await processFile({
        files,
        layers: enabledLayers,
        product,
        height,
        elevation,
        filters,
        selectedVolumes,
      });
      console.log("Respuesta de process:", processResp.data);
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
      setWarnings(processResp.data.warnings || []);
      setCurrentIndex(0);
      setComputeKey(nextKey);
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
    filters,
    // max_length_km,
    // elevation,
  }) => {
    const resp = await generatePseudoRHI({
      filepath,
      field,
      end_lat,
      end_lon,
      filters,
    });
    // devolvemos lo que el dialog espera
    return resp.data;
  };

  const handleOpenAreaStatsMode = () => {
    setAreaPolygon(null);
    setAreaDrawMode(true);
  };

  const handleAreaComplete = (gj, layer) => {
    drawnLayerRef.current = layer;
    setAreaDrawMode(false);
    setAreaPolygon(gj);
    setAreaStatsOpen(true);
  };

  const handleCloseAreaStats = () => {
    // al cerrar el diálogo, removemos la capa del mapa
    try {
      drawnLayerRef.current?.remove();
    } catch {
      console.log("Error");
    }
    drawnLayerRef.current = null;
    setAreaStatsOpen(false);
  };

  const handleAreaStatsRequest = async (payload) => {
    // backend espera: filepath, field, product, elevation?, height?, filters?, polygon
    const r = await generateAreaStats(payload);
    return r.data;
  };

  const handleTogglePixelStat = () => {
    setPixelStatMode((v) => {
      const next = !v;
      if (!next) setPixelStatMarker(null);
      return next;
    });
  };

  const handleMapClickPixelStat = async (latlng) => {
    try {
      const payload = {
        filepath: uploadedFiles[currentIndex],
        field: fieldsUsed?.[0] || "DBZH",
        product: overlayData?.product || "PPI",
        elevation: activeElevation,
        height: activeHeight,
        filters: filtersUsed,
        lat: latlng.lat,
        lon: latlng.lng,
      };
      const resp = await generatePixelStat(payload);
      const v = resp.data?.value;
      if (resp.data.masked || v == null) {
        enqueueSnackbar("Sin dato (masked / fuera de cobertura)", {
          variant: "warning",
        });
      } else {
        enqueueSnackbar(`${fieldsUsed?.[0] || "DBZH"}: ${v}`, {
          variant: "success",
        });
        setPixelStatMarker({
          lat: resp.data?.lat,
          lon: resp.data?.lon,
          value: v,
        });
      }
    } catch (e) {
      enqueueSnackbar(e?.response?.data?.detail || "Error", {
        variant: "error",
      });
    }
  };

  // ADAPTACIÓN MULTI-RADAR
  // Si la respuesta tiene results (multi-radar), combinamos los frames por timestamp
  let mergedOutputs = [];
  let animation = false;
  let product = overlayData.product;
  if (overlayData.results) {
    mergedOutputs = mergeRadarFrames(overlayData.results);
    animation = mergedOutputs.length > 1;
  } else {
    mergedOutputs = overlayData.outputs || [];
    animation = overlayData.animation;
  }
  var currentOverlay = mergedOutputs[currentIndex] || null;

  return (
    <>
      <MapView
        overlayData={currentOverlay}
        opacities={opacity}
        pickPointMode={pickPointMode}
        radarSite={radarSite}
        pickedPoint={pickedPoint}
        onPickPoint={handlePickPoint}
        drawAreaMode={areaDrawMode}
        onAreaComplete={handleAreaComplete}
        pixelStatMode={pixelStatMode}
        onPixelStatClick={handleMapClickPixelStat}
        pixelStatMarker={pixelStatMarker}
      />
      <ColorLegend fields={fieldsUsed} />
      <FloatingMenu
        onUploadClick={handleFileUpload}
        onChangeProductClick={() => setSelectorOpen(true)}
        onPseudoRhiClick={handleOpenRHI}
        onAreaStatsClick={handleOpenAreaStatsMode}
        onPixelStatToggle={handleTogglePixelStat}
      />
      <UploadButton onFilesSelected={handleFilesSelected} />

      {/* Slider para múltiples imágenes */}
      {mergedOutputs.length > 0 && (
        <AnimationControls
          overlayData={{ outputs: mergedOutputs, animation }}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
          showPlayButton={animation}
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
        volumes={volumes}
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

      <AreaStatsDialog
        open={areaStatsOpen}
        onClose={handleCloseAreaStats}
        requestFn={handleAreaStatsRequest}
        payload={{
          filepath: uploadedFiles[currentIndex],
          field: fieldsUsed?.[0] || "DBZH",
          product: overlayData?.product || "PPI",
          elevation: activeElevation,
          height: activeHeight,
          filters: filtersUsed,
          polygon: areaPolygon,
        }}
      />

      <Alerts
        open={alert.open}
        message={alert.message}
        severity={alert.severity}
        onClose={() => setAlert({ ...alert, open: false })}
      />
      <WarningPanel warnings={warnings} />
      <Loader open={loading} />
    </>
  );
}
