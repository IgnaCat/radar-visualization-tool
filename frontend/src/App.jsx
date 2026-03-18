import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useSnackbar } from "notistack";
import {
  uploadFile,
  processFile,
  generatePseudoRHI,
  generateAreaStats,
  generatePixelStat,
  generateElevationProfile,
  removeFiles,
} from "./api/backend";
import { registerCleanupAxios } from "./api/registerCleanupAxios";
import stableStringify from "json-stable-stringify";
import { useMapActions } from "./hooks/useMapActions";
import { useDownloads } from "./hooks/useDownloads";
import "./print.css";

import { generateSessionId } from "./utils/session";
import UploadButton from "./components/ui/UploadButton";
import HeaderCard from "./components/ui/HeaderCard";
import Alerts from "./components/ui/Alerts";
import Loader from "./components/ui/Loader";
import SplitScreenContainer from "./components/layout/SplitScreenContainer";
import SettingsDialog from "./components/dialogs/SettingsDialog";
import DownloadLayersDialog from "./components/dialogs/DownloadLayersDialog";

// Utilidad para combinar frames de múltiples radares por timestamp
function mergeRadarFrames(results, toleranceSec = 0) {
  // results: [{ radar, outputs: [[LayerResult,...], ...] }, ...]

  // 1) aplanamos a una lista de {tsISO, radar, layers}
  const shots = [];
  for (const r of results || []) {
    for (const layers of r.outputs || []) {
      const rawTs = layers?.[0]?.timestamp ?? null; // todos los layers del frame comparten ts
      const tsISO = rawTs
        ? new Date(rawTs).toISOString() // normalizamos SIEMPRE a ISO
        : null;
      // Anotar el radar en cada layer para uso posterior
      const layersWithRadar = layers.map((layer) => ({
        ...layer,
        radar: r.radar,
      }));
      shots.push({ tsISO, radar: r.radar, layers: layersWithRadar });
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
      .sort((L, R) => (L.order ?? 0) - (R.order ?? 0)),
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
  colormap_overrides,
  weightFunc,
  maxNeighbors,
  smoothing,
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
    colormap_overrides,
    weightFunc,
    maxNeighbors,
    smoothing,
  });
}

export default function App() {
  const [overlayData, setOverlayData] = useState({
    outputs: [],
    animation: false,
    metadata: {},
  });
  const [opacity, setOpacity] = useState([0.95]); // LEGACY: array posicional de opacidades por índice de layer
  const [opacityByField, setOpacityByField] = useState({});
  // Opacidad por capa individual (key: "FIELD::source_file"). Prioridad máxima en MapView.
  const [opacityByLayer, setOpacityByLayer] = useState({});
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0); // índice de la imagen activa
  const [loading, setLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [downloadLayersDialogOpen, setDownloadLayersDialogOpen] =
    useState(false);
  const [deltaT, setDeltaT] = useState(0);
  const [interpSettings, setInterpSettings] = useState({
    weightFunc: "nearest",
    maxNeighbors: 1,
    smoothing: {
      enabled: false,
      sigma: 0.8,
      only_when_nearest: true,
    },
  });
  const [settingsApplyVersion, setSettingsApplyVersion] = useState(0);
  const lastProcessedSettingsVersionRef = useRef(0);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [fieldsUsed, setFieldsUsed] = useState(["DBZH"]);
  const [filesInfo, setFilesInfo] = useState([]);
  const [volumes, setVolumes] = useState([]); // lista de volúmenes cargados sin repetidos
  const [availableRadars, setAvailableRadars] = useState([]); // todos los radares presentes en archivos subidos
  const [savedLayers, setSavedLayers] = useState([]); // layers / variables usadas
  const [filtersUsed, setFiltersUsed] = useState([]); // filtros globales aplicados
  const [filtersPerField, setFiltersPerField] = useState({}); // filtros por campo del LayerManager
  const [selectedVolumesUsed, setSelectedVolumesUsed] = useState([]); // últimos volúmenes procesados
  const [selectedRadarsUsed, setSelectedRadarsUsed] = useState([]); // últimos radares procesados
  const [activeElevation, setActiveElevation] = useState(null);
  const [activeHeight, setActiveHeight] = useState(null);
  const allCogsRef = useRef(new Set());
  // animación controlada por variable 'animation' derivada de outputs
  const [computeKey, setComputeKey] = useState("");
  const [warnings, setWarnings] = useState([]);
  const [mapInstance, setMapInstance] = useState(null); // Referencia al mapa Leaflet

  const [alert, setAlert] = useState({
    open: false,
    message: "",
    severity: "info",
  });

  const { enqueueSnackbar } = useSnackbar();

  // Session ID único para esta pestaña/ventana del navegador
  const [sessionId] = useState(() => generateSessionId());

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
  // Estado para el selector de mapas base
  const [mapSelectorOpen, setMapSelectorOpen] = useState(false);
  // Estado para el perfil de elevación
  const [elevationProfileOpen, setElevationProfileOpen] = useState(false);
  const [lineDrawMode, setLineDrawMode] = useState(false);
  const [drawnLineCoords, setDrawnLineCoords] = useState([]);
  const [lineDrawingFinished, setLineDrawingFinished] = useState(false);
  const [highlightedPoint, setHighlightedPoint] = useState(null);
  // Estado para modo de marcadores
  const [markerMode, setMarkerMode] = useState(false);
  const [markers, setMarkers] = useState([]);
  const [selectedBaseMap, setSelectedBaseMap] = useState({
    id: "osm",
    name: "Argenmap",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  });
  // Estado para paletas de colores personalizadas por campo
  const [selectedColormaps, setSelectedColormaps] = useState({});
  const [initialColormaps, setInitialColormaps] = useState({});
  const [paletteSelectorOpen, setPaletteSelectorOpen] = useState(false);
  const [layerManagerOpen, setLayerManagerOpen] = useState(false);
  const [fileManagerOpen, setFileManagerOpen] = useState(false);
  const [hiddenLayers, setHiddenLayers] = useState(new Set());

  // Estado para split screen
  const [splitScreenActive, setSplitScreenActive] = useState(false);

  // Hook para acciones del mapa (screenshot, print, fullscreen)
  const { isFullscreen, handleScreenshot, handlePrint, handleFullscreen } =
    useMapActions();

  // Hook para gestión de descargas
  const { generateFilename } = useDownloads();

  // Registrar cleanup en cierre de pestaña/ventana
  useEffect(() => {
    const unregister = registerCleanupAxios(() => ({
      uploads: uploadedFiles,
      cogs: Array.from(allCogsRef.current),
      delete_cache: true,
      session_id: sessionId,
    }));
    return unregister;
  }, [uploadedFiles, overlayData, sessionId]);

  // Seteamos currentOverlay
  // Si la respuesta tiene results (multi-radar), combinamos los frames por timestamp
  // sino dejamos la respuesta vieja
  let mergedOutputs = [];
  let animation = false;
  let product = overlayData.product || "PPI";
  if (overlayData.results) {
    mergedOutputs = mergeRadarFrames(overlayData.results, deltaT);
    animation = mergedOutputs.length > 1;
  } else {
    mergedOutputs = overlayData.outputs || [];
    animation = overlayData.animation;
  }
  var currentOverlay = mergedOutputs[currentIndex] || null;

  // Filtrar capas ocultas para el renderizado del mapa
  const visibleOverlay = useMemo(() => {
    if (!Array.isArray(currentOverlay) || hiddenLayers.size === 0)
      return currentOverlay;
    const filtered = currentOverlay.filter(
      (l) => !hiddenLayers.has(`${l.field}::${l.source_file}`),
    );
    return filtered.length > 0 ? filtered : [];
  }, [currentOverlay, hiddenLayers]);

  // Derivar sitio del radar a partir de la capa visible de mayor prioridad.
  const radarSite = useMemo(() => {
    const topVisibleLayerFile =
      Array.isArray(visibleOverlay) && visibleOverlay.length > 0
        ? visibleOverlay[0]?.source_file
        : null;
    const fileToUse = topVisibleLayerFile || uploadedFiles[currentIndex];
    if (!fileToUse) return null;
    const fi = filesInfo.find((f) => f.filepath === fileToUse);
    const md = fi?.metadata;
    if (!md) return null;
    const site = md.radar_site || md.site; // soportar ambos nombres
    if (!site || site.lat == null || site.lon == null) return null;
    return {
      lat: Number(site.lat),
      lon: Number(site.lon),
      alt_m: site.alt_m ?? site.alt ?? null,
    };
  }, [visibleOverlay, uploadedFiles, currentIndex, filesInfo]);

  // Función para descargar COGs de las capas actuales
  // layersOverride: si se pasa, se descargan esas capas directamente (desde el dialog de selección)
  const handleDownloadCOGs = useCallback(
    async (layersOverride) => {
      let layersToDownload;

      if (layersOverride) {
        layersToDownload = layersOverride;
      } else {
        if (!currentOverlay || currentOverlay.length === 0) {
          enqueueSnackbar("No hay capas disponibles para descargar", {
            variant: "warning",
          });
          return;
        }
        layersToDownload = currentOverlay.filter((layer) => layer.image_url);
      }

      if (layersToDownload.length === 0) {
        enqueueSnackbar("No hay archivos COG para descargar", {
          variant: "warning",
        });
        return;
      }

      try {
        // Crear todos los links de descarga
        const baseUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
        const downloadLinks = [];

        for (const layer of layersToDownload) {
          const filename =
            layer.image_url.split("/").pop() || generateFilename("cog", ".tif");
          const cogUrl = `${baseUrl}/${layer.image_url}`;

          try {
            const response = await fetch(cogUrl);
            if (!response.ok) {
              console.error(
                `Error descargando ${filename}: HTTP ${response.status}`,
              );
              continue;
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = filename;
            link.style.display = "none";
            document.body.appendChild(link);
            downloadLinks.push({ link, url });
          } catch (err) {
            console.error(`Error preparando descarga de ${filename}:`, err);
          }
        }

        // Hacer click en todos los links con pequeños delays
        for (let i = 0; i < downloadLinks.length; i++) {
          downloadLinks[i].link.click();
          if (i < downloadLinks.length - 1) {
            await new Promise((resolve) => setTimeout(resolve, 300));
          }
        }

        // Limpiar después de un delay final
        setTimeout(() => {
          downloadLinks.forEach(({ link, url }) => {
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
          });
        }, 1000);

        enqueueSnackbar(
          `${downloadLinks.length} archivo(s) COG descargado(s)`,
          {
            variant: "success",
          },
        );
      } catch (error) {
        console.error("Error descargando COGs:", error);
        enqueueSnackbar("Error al descargar archivos COG", {
          variant: "error",
        });
      }
    },
    [currentOverlay, generateFilename, enqueueSnackbar],
  );

  // Configurar descargas disponibles para el toolbar
  const availableDownloads = useMemo(() => {
    const downloads = {};

    // Captura del mapa (siempre disponible si hay mapa)
    if (mapInstance) {
      downloads.mapScreenshot = {
        handler: () => handleScreenshot(mapInstance, "map-container-main"),
        label: "Captura del mapa",
        disabled: false,
      };
    }

    // Descargar COGs de capas actuales
    if (currentOverlay && currentOverlay.length > 0) {
      // Con múltiples radares solo se descarga el de mayor prioridad, de lo contrario todas las capas
      const hasMultiRadar =
        new Set(
          (currentOverlay || []).map((l) => l.source_file).filter(Boolean),
        ).size > 1;
      downloads.cogLayers = {
        handler: () => setDownloadLayersDialogOpen(true),
        label: hasMultiRadar
          ? `Descargar capas GeoTIFF`
          : `Descargar ${currentOverlay.length} capa(s) Geotiff`,
        disabled: false,
      };
    }

    return downloads;
  }, [mapInstance, currentOverlay, handleScreenshot, handleDownloadCOGs]);

  const handleFileUpload = () => {
    document.getElementById("upload-file").click();
  };

  const handleFilesSelected = async (files) => {
    try {
      setLoading(true);
      const uploadResp = await uploadFile(files, sessionId);
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
      setFilesInfo((prev) => {
        // Merge con archivos anteriores, evitando duplicados por filepath
        const existingPaths = new Set(prev.map((f) => f.filepath));
        const newFiles = filesInfo.filter(
          (f) => !existingPaths.has(f.filepath),
        );
        return [...prev, ...newFiles];
      });
      setVolumes((prev) => {
        const merged = [...prev, ...uploadResp.data.volumes];
        return Array.from(new Set(merged));
      });
      setAvailableRadars((prev) => {
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
      // LEGACY: array posicional, usado como fallback en MapView cuando opacityByField no tiene el field
      const opacities = enabledLayerObjs.map((l) => l.opacity);

      // Build field-based opacity map so all radars for the same field share opacity
      const opacityMap = Object.fromEntries(
        enabledLayerObjs.map((l) => [
          String(l.label || l.field).toUpperCase(),
          Number(l.opacity ?? 1),
        ]),
      );

      setOpacity(opacities);
      setOpacityByField(opacityMap);
      setOpacityByLayer({}); // Reset per-layer overrides al reprocesar
      setFieldsUsed(enabledLayers);
      setSavedLayers(data.layers);
      setFiltersUsed(filters);
      setSelectedVolumesUsed(selectedVolumes);
      setSelectedRadarsUsed(selectedRadars);
      // Reset per-field filters when re-processing from ProductSelector (new configuration)
      setFiltersPerField({});
      if (elevation !== undefined) setActiveElevation(elevation);

      // Reset height to null when product is not CAPPI
      if (height !== undefined) {
        setActiveHeight(height);
      } else {
        setActiveHeight(null);
      }

      const nextKey = buildComputeKey({
        files: uploadedFiles,
        product,
        fields: enabledLayers,
        elevation,
        height,
        filters,
        selectedVolumes,
        selectedRadars,
        colormap_overrides: selectedColormaps,
        weightFunc: interpSettings.weightFunc,
        maxNeighbors: interpSettings.maxNeighbors,
        smoothing: interpSettings.smoothing,
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
        colormap_overrides: selectedColormaps,
        session_id: sessionId,
        weight_func: interpSettings.weightFunc,
        max_neighbors: interpSettings.maxNeighbors,
        smoothing: interpSettings.smoothing,
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
      setWarnings(processResp.data.warnings || []);
      setCurrentIndex(0);
      setComputeKey(nextKey);
      setHiddenLayers(new Set()); // Limpiar capas ocultas al reprocesar
      // Guardar las paletas usadas como "iniciales" para futuras comparaciones
      setInitialColormaps({ ...selectedColormaps });

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

  const handleGenerateRHI = async ({
    filepath,
    field,
    end_lat,
    end_lon,
    start_lat,
    start_lon,
    filters,
    max_length_km,
    max_height_km,
    min_length_km,
    min_height_km,
    smoothing,
  }) => {
    const resp = await generatePseudoRHI({
      filepath,
      field,
      end_lat,
      end_lon,
      start_lat,
      start_lon,
      filters,
      max_length_km,
      max_height_km,
      min_length_km,
      min_height_km,
      colormap_overrides: selectedColormaps,
      weight_func: interpSettings.weightFunc,
      max_neighbors: interpSettings.maxNeighbors,
      smoothing: smoothing ?? interpSettings.smoothing,
      session_id: sessionId,
    });
    // devolvemos lo que el dialog espera
    return resp.data;
  };

  const handleAreaStatsRequest = async (payload) => {
    // backend espera: filepath, field, product, elevation?, height?, filters?, polygon
    const r = await generateAreaStats({
      ...payload,
      session_id: sessionId,
      weight_func: interpSettings.weightFunc,
      max_neighbors: interpSettings.maxNeighbors,
    });
    return r.data;
  };

  // Handlers para modo de marcadores
  const handleToggleMarkerMode = () => {
    setMarkerMode((prev) => !prev);
  };

  const handleAddMarker = (marker) => {
    setMarkers((prev) => [...prev, marker]);
  };

  const handleRemoveMarker = (markerId) => {
    setMarkers((prev) => prev.filter((m) => m.id !== markerId));
  };

  const handleRenameMarker = (markerId, name) => {
    setMarkers((prev) =>
      prev.map((m) => (m.id === markerId ? { ...m, name } : m)),
    );
  };

  /**
   * Alterna la visibilidad de una capa en el mapa.
   * La capa permanece en la lista de capas pero se oculta/muestra en el mapa.
   */
  const handleToggleLayerVisibility = useCallback(
    (field, sourceFile) => {
      const key = `${field}::${sourceFile}`;

      setHiddenLayers((prev) => {
        const next = new Set(prev);
        if (next.has(key)) {
          next.delete(key);
        } else {
          next.add(key);
        }

        // Actualizar savedLayers (enabled flag) con el estado real resultante.
        setSavedLayers((prevLayers) =>
          prevLayers.map((layer) =>
            layer.field === field || layer.label === field
              ? { ...layer, enabled: !next.has(key) }
              : layer,
          ),
        );

        // Actualizar fieldsUsed (solo campos visibles) con el hidden set actualizado.
        const allFields = mergedOutputs.flatMap((frame) =>
          Array.isArray(frame) ? frame.map((l) => l.field) : [],
        );
        const uniqueFields = [...new Set(allFields)];
        const visibleFields = uniqueFields.filter((f) => {
          return mergedOutputs.some(
            (frame) =>
              Array.isArray(frame) &&
              frame.some(
                (l) =>
                  l.field === f && !next.has(`${l.field}::${l.source_file}`),
              ),
          );
        });
        setFieldsUsed(visibleFields.length > 0 ? visibleFields : uniqueFields);

        return next;
      });
    },
    [mergedOutputs],
  );

  /**
   * Actualiza la opacidad de una capa individual identificada por field + sourceFile.
   * Usa clave compuesta "FIELD::source_file" para soportar múltiples radares con el mismo field.
   */
  const handleLayerOpacityChange = useCallback((field, sourceFile, value) => {
    const key = `${String(field || "").toUpperCase()}::${sourceFile || ""}`;
    setOpacityByLayer((prev) => ({ ...prev, [key]: value }));
  }, []);

  /**
   * Aplica filtros por campo desde el LayerManagerDialog y reprocesa.
   * newFiltersPerField: { FIELD: [{field, min, max}, ...] }
   */
  const handleApplyLayerFilters = useCallback(
    async (newFiltersPerField) => {
      if (uploadedFiles.length === 0) return;
      setFiltersPerField(newFiltersPerField);
      try {
        setLoading(true);
        const processResp = await processFile({
          files: uploadedFiles,
          layers: fieldsUsed,
          product: overlayData?.product || "ppi",
          height: activeHeight,
          elevation: activeElevation,
          filters: filtersUsed,
          selectedVolumes: selectedVolumesUsed,
          selectedRadars: selectedRadarsUsed,
          filters_per_field: newFiltersPerField,
          colormap_overrides: selectedColormaps,
          session_id: sessionId,
          weight_func: interpSettings.weightFunc,
          max_neighbors: interpSettings.maxNeighbors,
          smoothing: interpSettings.smoothing,
        });
        if (processResp.data?.results?.length > 0) {
          setOverlayData(processResp.data);
          setWarnings(processResp.data.warnings || []);
          setCurrentIndex(0);
        }
      } catch (err) {
        console.error(err);
        setAlert({
          open: true,
          message: "Error al aplicar filtros",
          severity: "error",
        });
      } finally {
        setLoading(false);
      }
    },
    [
      uploadedFiles,
      fieldsUsed,
      overlayData,
      activeHeight,
      activeElevation,
      filtersUsed,
      selectedVolumesUsed,
      selectedRadarsUsed,
      selectedColormaps,
      sessionId,
      processFile,
      interpSettings,
    ],
  );

  const reprocessCurrentView = useCallback(
    async (settings = interpSettings) => {
      if (uploadedFiles.length === 0) return;

      const enabledLayers =
        savedLayers
          .filter((layer) => layer.enabled)
          .map((layer) => layer.label) || [];
      const layersToProcess =
        enabledLayers.length > 0 ? enabledLayers : fieldsUsed;
      const currentProduct = overlayData?.product || product || "PPI";

      if (!currentProduct || layersToProcess.length === 0) return;

      try {
        setLoading(true);
        const processResp = await processFile({
          files: uploadedFiles,
          layers: layersToProcess,
          product: currentProduct,
          height: activeHeight,
          elevation: activeElevation,
          filters: filtersUsed,
          selectedVolumes: selectedVolumesUsed,
          selectedRadars: selectedRadarsUsed,
          filters_per_field: filtersPerField,
          colormap_overrides: selectedColormaps,
          session_id: sessionId,
          weight_func: settings.weightFunc,
          max_neighbors: settings.maxNeighbors,
          smoothing: settings.smoothing,
        });

        if (processResp.data) {
          setOverlayData(processResp.data);
          setWarnings(processResp.data.warnings || []);
          setCurrentIndex(0);
          setComputeKey(
            buildComputeKey({
              files: uploadedFiles,
              product: currentProduct,
              fields: layersToProcess,
              elevation: activeElevation,
              height: activeHeight,
              filters: filtersUsed,
              selectedVolumes: selectedVolumesUsed,
              selectedRadars: selectedRadarsUsed,
              colormap_overrides: selectedColormaps,
              weightFunc: settings.weightFunc,
              maxNeighbors: settings.maxNeighbors,
              smoothing: settings.smoothing,
            }),
          );
          setHiddenLayers(new Set());
        }
      } catch (err) {
        console.error(err);
        setAlert({
          open: true,
          message: "Error al reprocesar con la nueva configuración",
          severity: "error",
        });
      } finally {
        setLoading(false);
      }
    },
    [
      uploadedFiles,
      savedLayers,
      fieldsUsed,
      overlayData,
      product,
      activeHeight,
      activeElevation,
      filtersUsed,
      selectedVolumesUsed,
      selectedRadarsUsed,
      filtersPerField,
      selectedColormaps,
      sessionId,
      interpSettings,
    ],
  );

  useEffect(() => {
    if (settingsApplyVersion === 0) return;
    if (lastProcessedSettingsVersionRef.current === settingsApplyVersion)
      return;
    lastProcessedSettingsVersionRef.current = settingsApplyVersion;
    reprocessCurrentView(interpSettings);
  }, [settingsApplyVersion, interpSettings, reprocessCurrentView]);

  /**
   * Elimina uno o varios archivos subidos del servidor y actualiza todo el estado.
   * Borra el NetCDF, COGs y cache en backend. En frontend filtra el archivo
   * de todos los estados derivados.
   */
  const handleRemoveFile = useCallback(
    async (targetFilepaths) => {
      const filepaths = (
        Array.isArray(targetFilepaths) ? targetFilepaths : [targetFilepaths]
      ).filter(Boolean);

      if (filepaths.length === 0) return;

      const normalizedFilepaths = filepaths.map((p) =>
        String(p).replace(/\\/g, "/"),
      );
      const removedPathSet = new Set(normalizedFilepaths);
      const removedFileNames = new Set(
        normalizedFilepaths.map((p) => p.split("/").pop()),
      );

      try {
        setLoading(true);

        // 1. Llamar al backend para borrar archivo(s), COGs y cache
        await removeFiles(filepaths, sessionId);

        // 2. Actualizar uploadedFiles
        const newUploadedFiles = uploadedFiles.filter(
          (f) => !removedPathSet.has(String(f).replace(/\\/g, "/")),
        );
        setUploadedFiles(newUploadedFiles);

        // 3. Actualizar filesInfo
        const newFilesInfo = filesInfo.filter(
          (f) => !removedPathSet.has(String(f.filepath).replace(/\\/g, "/")),
        );
        setFilesInfo(newFilesInfo);

        // 4. Recalcular volumes y availableRadars
        const newVolumes = [
          ...new Set(
            newFilesInfo
              .map((f) => {
                const parts = f.filepath.split("_");
                return parts.length >= 3 ? parts[2] : null;
              })
              .filter(Boolean),
          ),
        ];
        setVolumes(newVolumes);

        const newRadars = [
          ...new Set(
            newFilesInfo
              .map((f) => {
                const parts = f.filepath.split("_");
                return parts.length >= 1 ? parts[0] : null;
              })
              .filter(Boolean),
          ),
        ];
        setAvailableRadars(newRadars);

        // 5. Filtrar capas del archivo eliminado de mergedOutputs
        if (mergedOutputs.length > 0) {
          const updatedOutputs = mergedOutputs
            .map((frame) => {
              if (!Array.isArray(frame)) return frame;
              return frame.filter((layer) => {
                // Comparar por nombre de archivo (el source_file puede tener ruta completa)
                const layerFile = String(layer.source_file || "")
                  .replace(/\\/g, "/")
                  .split("/")
                  .pop();
                return !removedFileNames.has(layerFile);
              });
            })
            .filter((frame) => Array.isArray(frame) && frame.length > 0);

          if (updatedOutputs.length === 0) {
            setOverlayData({ outputs: [], animation: false, metadata: {} });
            setCurrentIndex(0);
          } else {
            setCurrentIndex((prev) =>
              prev >= updatedOutputs.length ? updatedOutputs.length - 1 : prev,
            );
            setOverlayData({
              ...overlayData,
              outputs: updatedOutputs,
              results: null,
            });
          }

          // Calcular campos que quedan en los outputs restantes
          const remainingFields = new Set(
            updatedOutputs.flatMap((frame) =>
              Array.isArray(frame) ? frame.map((l) => l.field) : [],
            ),
          );

          // Actualizar savedLayers: quitar campos que ya no existen en ningún output
          setSavedLayers((prev) =>
            prev.filter(
              (layer) =>
                remainingFields.has(layer.field) ||
                remainingFields.has(layer.label),
            ),
          );

          // Actualizar fieldsUsed: solo los campos que siguen presentes
          setFieldsUsed((prev) => {
            const filtered = prev.filter((f) => remainingFields.has(f));
            return filtered.length > 0 ? filtered : ["DBZH"];
          });

          // Limpiar opacityByField de campos eliminados
          setOpacityByField((prev) => {
            const next = { ...prev };
            for (const key of Object.keys(next)) {
              if (
                !remainingFields.has(key) &&
                !remainingFields.has(key.toUpperCase())
              ) {
                delete next[key];
              }
            }
            return next;
          });

          // Limpiar opacityByLayer de capas del archivo eliminado
          setOpacityByLayer((prev) => {
            const next = { ...prev };
            for (const key of Object.keys(next)) {
              if (
                normalizedFilepaths.some((path) => key.includes(path)) ||
                Array.from(removedFileNames).some((name) => key.includes(name))
              ) {
                delete next[key];
              }
            }
            return next;
          });
        }

        // 6. Si no quedan archivos, resetear a estado inicial
        if (newUploadedFiles.length === 0) {
          setOverlayData({ outputs: [], animation: false, metadata: {} });
          setCurrentIndex(0);
          setSavedLayers([]);
          setFieldsUsed(["DBZH"]);
          setFiltersUsed([]);
          setComputeKey("");
          setWarnings([]);
          setHiddenLayers(new Set());
          setOpacityByField({});
          setOpacityByLayer({});
        } else {
          // Limpiar hiddenLayers de entries del archivo eliminado
          setHiddenLayers((prev) => {
            const next = new Set(prev);
            for (const key of prev) {
              if (
                normalizedFilepaths.some((path) => key.includes(path)) ||
                Array.from(removedFileNames).some((name) => key.includes(name))
              ) {
                next.delete(key);
              }
            }
            return next;
          });
        }

        enqueueSnackbar(
          filepaths.length === 1
            ? "Archivo eliminado correctamente"
            : `Se eliminaron ${filepaths.length} archivos correctamente`,
          {
            variant: "success",
          },
        );
      } catch (err) {
        console.error("Error eliminando archivo:", err);
        enqueueSnackbar(
          err?.response?.data?.detail ||
            (filepaths.length === 1
              ? "Error al eliminar archivo"
              : "Error al eliminar archivos"),
          { variant: "error" },
        );
      } finally {
        setLoading(false);
      }
    },
    [
      uploadedFiles,
      filesInfo,
      mergedOutputs,
      overlayData,
      sessionId,
      enqueueSnackbar,
    ],
  );

  const handleLayerReorder = (reorderedLayers) => {
    // Actualizar el orden de las capas en currentOverlay
    // Asignar nuevo 'order' basado en el índice
    const updatedLayers = reorderedLayers.map((layer, idx) => ({
      ...layer,
      order: idx,
    }));

    // Extraer el orden de los campos
    const fieldOrder = updatedLayers.map((l) => l.field);
    const uniqueFields = [...new Set(fieldOrder)];

    // Actualizar savedLayers para sincronizar con ProductSelectorDialog
    // Mantener TODOS los campos (habilitados y deshabilitados) pero reordenar los habilitados
    const reorderedSavedLayers = [];

    // Primero agregar los campos activos en el nuevo orden
    uniqueFields.forEach((field) => {
      const existingLayer = savedLayers.find(
        (l) => l.field === field || l.label === field,
      );
      if (existingLayer) {
        reorderedSavedLayers.push(existingLayer);
      }
    });

    // Luego agregar los campos deshabilitados que no están en el nuevo orden
    savedLayers.forEach((layer) => {
      const isInNewOrder =
        uniqueFields.includes(layer.field) ||
        uniqueFields.includes(layer.label);
      if (!isInNewOrder) {
        reorderedSavedLayers.push(layer);
      }
    });

    if (reorderedSavedLayers.length > 0) {
      setSavedLayers(reorderedSavedLayers);
    }

    // Sincronizar fieldsUsed con el nuevo orden para que pixel/area stats
    // usen el campo de la primera capa visible
    if (updatedLayers.length > 0) {
      setFieldsUsed(uniqueFields);
    }

    // Actualizar TODOS los frames con el nuevo orden
    // Aplicar el orden a todas las capas de todos los frames
    const updatedOutputs = mergedOutputs.map((frame) => {
      if (!Array.isArray(frame)) return frame;

      // Reordenar las capas de este frame según el nuevo orden
      const reordered = [...frame].sort((a, b) => {
        const aIdx = updatedLayers.findIndex(
          (l) => l.field === a.field && l.source_file === a.source_file,
        );
        const bIdx = updatedLayers.findIndex(
          (l) => l.field === b.field && l.source_file === b.source_file,
        );

        // Si no se encuentra, mantener al final
        if (aIdx === -1 && bIdx === -1) return 0;
        if (aIdx === -1) return 1;
        if (bIdx === -1) return -1;

        return aIdx - bIdx;
      });

      // Asignar los índices de orden
      return reordered.map((layer, idx) => ({
        ...layer,
        order: idx,
      }));
    });

    setOverlayData({
      ...overlayData,
      outputs: updatedOutputs,
      results: null, // Eliminar results para que use outputs
    });
  };

  const handleGenerateElevationProfile = async (coordinates) => {
    return await generateElevationProfile({ coordinates });
  };

  const handleMapClickPixelStat = async (latlng) => {
    try {
      const topVisibleLayer =
        Array.isArray(visibleOverlay) && visibleOverlay.length > 0
          ? visibleOverlay[0]
          : null;
      const payload = {
        // Usar la capa visible de mayor prioridad.
        filepath: topVisibleLayer?.source_file || uploadedFiles[currentIndex],
        field: topVisibleLayer?.field || fieldsUsed?.[0] || "DBZH",
        product: overlayData?.product || "PPI",
        elevation: activeElevation,
        height: activeHeight,
        filters: filtersUsed,
        lat: latlng.lat,
        lon: latlng.lng,
        weight_func: interpSettings.weightFunc,
        max_neighbors: interpSettings.maxNeighbors,
        session_id: sessionId,
      };
      const resp = await generatePixelStat(payload);
      const v = resp.data?.value.toFixed(2);
      if (resp.data.masked || v == null) {
        enqueueSnackbar("Sin dato (masked / fuera de cobertura)", {
          variant: "warning",
        });
      } else {
        enqueueSnackbar(`${payload.field || "DBZH"}: ${v}`, {
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

  return (
    <div
      id="app-container"
      style={{ height: "100vh", width: "100%", position: "relative" }}
    >
      {/* Header común para ambas vistas */}
      <HeaderCard onUploadClick={handleFileUpload} />

      {/* Contenedor de split screen que maneja uno o dos mapas */}
      <SplitScreenContainer
        splitScreenActive={splitScreenActive}
        setSplitScreenActive={setSplitScreenActive}
        map1Props={{
          currentOverlay: visibleOverlay,
          allLayersOverlay: currentOverlay,
          hiddenLayers,
          mergedOutputs,
          opacity,
          opacityByField,
          opacityByLayer,
          currentIndex,
          setCurrentIndex,
          animation,
          pixelStatMode,
          setPixelStatMode,
          pixelStatMarker,
          setPixelStatMarker,
          pickPointMode,
          setPickPointMode,
          pickedPoint,
          setPickedPoint,
          areaDrawMode,
          setAreaDrawMode,
          areaPolygon,
          setAreaPolygon,
          lineDrawMode,
          setLineDrawMode,
          drawnLineCoords,
          setDrawnLineCoords,
          lineDrawingFinished,
          setLineDrawingFinished,
          highlightedPoint,
          setHighlightedPoint,
          markerMode,
          setMarkerMode,
          markers,
          setMarkers,
          rhiLinePreview,
          setRhiLinePreview,
          selectorOpen,
          setSelectorOpen,
          rhiOpen,
          setRhiOpen,
          areaStatsOpen,
          setAreaStatsOpen,
          elevationProfileOpen,
          setElevationProfileOpen,
          mapSelectorOpen,
          setMapSelectorOpen,
          paletteSelectorOpen,
          setPaletteSelectorOpen,
          layerManagerOpen,
          setLayerManagerOpen,
          selectedBaseMap,
          setSelectedBaseMap,
          selectedColormaps,
          setSelectedColormaps,
          initialColormaps,
          setInitialColormaps,
          onProductChosen: handleProductChosen,
          onGenerateRHI: handleGenerateRHI,
          onAreaStatsRequest: handleAreaStatsRequest,
          onPixelStatClick: handleMapClickPixelStat,
          onGenerateElevationProfile: handleGenerateElevationProfile,
          onToggleMarkerMode: handleToggleMarkerMode,
          onAddMarker: handleAddMarker,
          onRemoveMarker: handleRemoveMarker,
          onRenameMarker: handleRenameMarker,
          onLayerReorder: handleLayerReorder,
          onToggleLayerVisibility: handleToggleLayerVisibility,
          onLayerOpacityChange: handleLayerOpacityChange,
          filtersPerField,
          onApplyFilters: handleApplyLayerFilters,
          fileManagerOpen,
          setFileManagerOpen,
          onRemoveFile: handleRemoveFile,
          mapInstance,
          setMapInstance,
          onScreenshot: handleScreenshot,
          onPrint: handlePrint,
          onFullscreen: handleFullscreen,
          isFullscreen,
          onSettingsOpen: () => setSettingsOpen(true),
          savedLayers,
          fieldsUsed,
          filtersUsed,
          activeElevation,
          activeHeight,
          radarSite,
          warnings,
          availableDownloads,
          drawnLayerRef,
          product,
          loading,
        }}
        sharedProps={{
          uploadedFiles,
          filesInfo,
          volumes,
          availableRadars,
          sessionId,
          enqueueSnackbar,
          processFile,
          generatePseudoRHI,
          generatePixelStat,
          generateAreaStats,
          mergeRadarFrames,
        }}
      />

      {/* UploadButton oculto - funcionalidad movida a HeaderCard */}
      <div style={{ display: "none" }}>
        <UploadButton onFilesSelected={handleFilesSelected} />
      </div>

      <Alerts
        open={alert.open}
        message={alert.message}
        severity={alert.severity}
        onClose={() => setAlert({ ...alert, open: false })}
      />

      <DownloadLayersDialog
        open={downloadLayersDialogOpen}
        onClose={() => setDownloadLayersDialogOpen(false)}
        layers={currentOverlay || []}
        onDownload={handleDownloadCOGs}
      />

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onApply={({
          deltaT: newDeltaT,
          weightFunc,
          maxNeighbors,
          smoothing,
        }) => {
          setDeltaT(newDeltaT);
          setInterpSettings({ weightFunc, maxNeighbors, smoothing });
          setSettingsApplyVersion((prev) => prev + 1);
        }}
        initialSettings={{
          deltaT,
          weightFunc: interpSettings.weightFunc,
          maxNeighbors: interpSettings.maxNeighbors,
          smoothing: interpSettings.smoothing,
        }}
      />

      <Loader open={loading} />
    </div>
  );
}
