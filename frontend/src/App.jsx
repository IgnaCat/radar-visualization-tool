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
import ActiveLayerPicker from "./components/ActiveLayerPicker";
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
function mergeRadarFrames(results, toleranceSec = 240) {
  // results: [{ radar, outputs: [[LayerResult,...], ...] }, ...]

  // 1) aplanamos a una lista de {tsISO, radar, layers}
  const shots = [];
  for (const r of results || []) {
    for (const layers of r.outputs || []) {
      const rawTs = layers?.[0]?.timestamp ?? null; // todos los layers del frame comparten ts
      const tsISO = rawTs
        ? new Date(rawTs).toISOString() // normalizamos SIEMPRE a ISO
        : null;
      shots.push({ tsISO, radar: r.radar, layers });
    }
  }

  // 2) agrupamos por "bucket temporal" con tolerancia
  const buckets = []; // [{ center: Date, members: [layers,...] }]
  const tolMs = toleranceSec * 1000;

  const tryPutInBucket = (ts, layers) => {
    if (!ts) {
      buckets.push({ center: null, members: [layers] }); // sin timestamp: bucket propio
      return;
    }
    const t = new Date(ts).getTime();
    for (const b of buckets) {
      if (b.center === null) continue;
      const dt = Math.abs(b.center.getTime() - t);
      if (dt <= tolMs) {
        b.members.push(layers);
        return;
      }
    }
    // no encontró bucket compatible → crear uno nuevo
    buckets.push({ center: new Date(t), members: [layers] });
  };

  for (const s of shots) {
    tryPutInBucket(s.tsISO, s.layers);
  }

  // 3) ordenar buckets por tiempo (los null al final)
  buckets.sort((a, b) => {
    if (a.center === null && b.center === null) return 0;
    if (a.center === null) return 1;
    if (b.center === null) return -1;
    return a.center - b.center;
  });

  // 4) salida: cada bucket → un “frame” con todas las capas mezcladas (multi-radar)
  //    mantenemos el orden interno de cada frame por 'order' (ya viene en cada LayerResult)
  return buckets.map((b) =>
    b.members
      .flat()
      .slice()
      .sort((L, R) => (L.order ?? 0) - (R.order ?? 0))
  );
}

function buildComputeKey({
  files,
  product,
  fields,
  elevation,
  height,
  filters,
  selectedVolumes,
  selectedRadars,
}) {
  return stableStringify({
    files,
    product,
    fields,
    elevation,
    height,
    filters,
    selectedVolumes,
    selectedRadars,
  });
}

export default function App() {
  const [overlayData, setOverlayData] = useState({
    outputs: [],
    animation: false,
    metadata: {},
  });
  const [opacity, setOpacity] = useState([0.95]);
  const [opacityByField, setOpacityByField] = useState({});
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0); // índice de la imagen activa
  const [loading, setLoading] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [fieldsUsed, setFieldsUsed] = useState("DBZH");
  const [filesInfo, setFilesInfo] = useState([]);
  const [volumes, setVolumes] = useState([]); // lista de volúmenes cargados sin repetidos
  const [selectedRadars, setSelectedRadars] = useState([]);
  const [savedLayers, setSavedLayers] = useState([]); // layers / variables usadas
  const [filtersUsed, setFiltersUsed] = useState([]); // filtros aplicados
  const [activeElevation, setActiveElevation] = useState(null);
  const [activeHeight, setActiveHeight] = useState(null);
  const allCogsRef = useRef(new Set());
  // animación controlada por variable 'animation' derivada de outputs
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
  const [rhiLinePreview, setRhiLinePreview] = useState({
    start: null,
    end: null,
  });
  const [areaDrawMode, setAreaDrawMode] = useState(false);
  const [areaPolygon, setAreaPolygon] = useState(null);
  const [areaStatsOpen, setAreaStatsOpen] = useState(false);
  const drawnLayerRef = useRef(null); // referencia a la capa dibujada
  // archivo seleccionado manualmente para herramientas (pixel/área/RHI)
  const [activeToolFile, setActiveToolFile] = useState(null);
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
      setSelectedRadars((prev) => {
        const merged = [...prev, ...uploadResp.data.radars];
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
      const selectedRadars = data.selectedRadars;
      const enabledLayers = layers.filter((l) => l.enabled).map((l) => l.label);
      const enabledLayerObjs = layers.filter((l) => l.enabled);
      const opacities = enabledLayerObjs.map((l) => l.opacity);

      // Build field-based opacity map so all radars for the same field share opacity
      const opacityMap = Object.fromEntries(
        enabledLayerObjs.map((l) => [
          String(l.label || l.field).toUpperCase(),
          Number(l.opacity ?? 1),
        ])
      );

      setOpacity(opacities);
      setOpacityByField(opacityMap);
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
        selectedRadars,
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
        selectedRadars,
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
      // Animación se calcula dinámicamente más abajo

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
    // Activar modo pick point en el mapa para el RHI
    setPickedPoint(null);
    setPickPointMode(true);
  };
  const handlePickPoint = (pt) => {
    setPickedPoint(pt);
    setPickPointMode(false); // se desactiva al elegir
    // Si aún no hay punto inicial para el RHI, fijarlo para que el marcador persista
    setRhiLinePreview((prev) => {
      if (!prev?.start) {
        return {
          start: { lat: pt.lat, lon: pt.lng ?? pt.lon },
          end: prev?.end || null,
        };
      }
      return prev;
    });
  };
  const handleClearPickedPoint = () => {
    setPickedPoint(null);
    setRhiLinePreview({ start: null, end: null });
  };

  // Callback para limpiar la línea cuando se limpian los puntos
  const handleClearLineOverlay = () => {
    setRhiLinePreview({ start: null, end: null });
  };

  const handleGenerateRHI = async ({
    filepath,
    field,
    end_lat,
    end_lon,
    start_lat,
    start_lon,
    filters,
    // max_length_km,
    // elevation,
  }) => {
    const resp = await generatePseudoRHI({
      filepath,
      field,
      end_lat,
      end_lon,
      start_lat,
      start_lon,
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
        filepath: activeToolFile || uploadedFiles[currentIndex],
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
  // sino dejamos la respuesta vieja
  let mergedOutputs = [];
  let animation = false;
  // const product = overlayData.product;
  if (overlayData.results) {
    mergedOutputs = mergeRadarFrames(overlayData.results);
    animation = mergedOutputs.length > 1;
  } else {
    mergedOutputs = overlayData.outputs || [];
    animation = overlayData.animation;
  }
  var currentOverlay = mergedOutputs[currentIndex] || null;

  // Sincronizar archivo activo con las capas visibles del frame actual
  useEffect(() => {
    const sources = Array.isArray(currentOverlay)
      ? currentOverlay.map((L) => L?.source_file).filter(Boolean)
      : [];
    if (sources.length === 0) return;
    setActiveToolFile((prev) =>
      prev && sources.includes(prev) ? prev : sources[0]
    );
  }, [currentOverlay]);

  return (
    <>
      <MapView
        overlayData={currentOverlay}
        opacities={opacity}
        opacityByField={opacityByField}
        pickPointMode={pickPointMode}
        radarSite={radarSite}
        pickedPoint={pickedPoint}
        onPickPoint={handlePickPoint}
        drawAreaMode={areaDrawMode}
        onAreaComplete={handleAreaComplete}
        pixelStatMode={pixelStatMode}
        onPixelStatClick={handleMapClickPixelStat}
        pixelStatMarker={pixelStatMarker}
        lineOverlay={
          rhiLinePreview?.start && rhiLinePreview?.end
            ? [
                [rhiLinePreview.start.lat, rhiLinePreview.start.lon],
                [rhiLinePreview.end.lat, rhiLinePreview.end.lon],
              ]
            : null
        }
        onClearLineOverlay={handleClearLineOverlay}
        rhiEndpoints={{ start: rhiLinePreview.start, end: rhiLinePreview.end }}
      />
      {/* Selector de capa activa para herramientas (cuando hay varias capas a la vez) */}
      <ActiveLayerPicker
        layers={Array.isArray(currentOverlay) ? currentOverlay : []}
        value={activeToolFile}
        onChange={setActiveToolFile}
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
        radars={selectedRadars}
        initialLayers={savedLayers}
        onClose={() => setSelectorOpen(false)}
        onConfirm={handleProductChosen}
      />

      {/* Dialog Pseudo-RHI */}
      <PseudoRHIDialog
        open={rhiOpen}
        onClose={() => setRhiOpen(false)}
        filepath={activeToolFile || uploadedFiles[currentIndex]}
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
        onLinePreviewChange={setRhiLinePreview}
      />

      <AreaStatsDialog
        open={areaStatsOpen}
        onClose={handleCloseAreaStats}
        requestFn={handleAreaStatsRequest}
        payload={{
          filepath: activeToolFile || uploadedFiles[currentIndex],
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
