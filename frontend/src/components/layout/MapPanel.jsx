import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import MapView from "../map/MapView";
import VerticalToolbar from "../controls/VerticalToolbar";
import MapToolbar from "../controls/MapToolbar";
import DrawingToolbar from "../controls/DrawingToolbar";
import ZoomControls from "../controls/ZoomControls";
import ColorLegend from "../map/ColorLegend";
import BaseMapSelector from "../map/BaseMapSelector";
import ColorPaletteSelector from "../controls/ColorPaletteSelector";
import LayerManagerDialog from "../dialogs/LayerManagerDialog";
import FileManagerDialog from "../dialogs/FileManagerDialog";
import AnimationControls from "../controls/AnimationControls";
import ProductSelectorDialog from "../dialogs/ProductSelectorDialog";
import PseudoRHIDialog from "../dialogs/PseudoRHIDialog";
import AreaStatsDialog from "../dialogs/AreaStatsDialog";
import ElevationProfileDialog from "../dialogs/ElevationProfileDialog";
import WarningPanel from "../ui/WarningPanel";
import Loader from "../ui/Loader";
import { analyzeFieldsAcrossFiles } from "../../utils/fieldAnalysis";

function getRangeCircleKey(layer) {
  return `${layer?.field || ""}::${layer?.source_file || ""}`;
}

function areProfilePointsEqual(prevPoints = [], nextPoints = []) {
  if (prevPoints === nextPoints) return true;
  if (prevPoints.length !== nextPoints.length) return false;

  for (let i = 0; i < prevPoints.length; i += 1) {
    const prev = prevPoints[i];
    const next = nextPoints[i];

    if (
      prev?.lat !== next?.lat ||
      prev?.lon !== next?.lon ||
      prev?.distance !== next?.distance ||
      prev?.elevation !== next?.elevation
    ) {
      return false;
    }
  }

  return true;
}

/**
 * MapPanel - Encapsula un mapa completo con todas sus herramientas
 * Puede funcionar independiente o sincronizado
 */
export default function MapPanel({
  // Identificador del panel (para claves únicas)
  panelId = "main",

  // Datos del mapa
  overlayData, // El frame actual (array de capas visibles para el mapa)
  allLayersOverlay, // Todas las capas del frame (incluye ocultas) para LayerManager
  hiddenLayers, // Set de "field::source_file" keys ocultas
  mergedOutputs, // Todos los frames para AnimationControls
  opacity,
  opacityByField,
  currentIndex,
  setCurrentIndex,
  animation,

  // Estados y handlers de herramientas
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

  // Diálogos
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

  // Mapa base
  selectedBaseMap,
  setSelectedBaseMap,

  // Paletas de colores
  selectedColormaps,
  setSelectedColormaps,
  initialColormaps,

  // Handlers de acciones
  onProductChosen,
  onGenerateRHI,
  onAreaStatsRequest,
  onPixelStatClick,
  onGenerateElevationProfile,
  onToggleMarkerMode,
  onAddMarker,
  onRemoveMarker,
  onRenameMarker,
  onLayerReorder,
  onToggleLayerVisibility, // (field, source_file) => void - toggle visibilidad de capa
  opacityByLayer, // { "FIELD::source_file": number } opacidades por capa individual
  onLayerOpacityChange, // (field, source_file, opacity) => void
  onMapReady,
  onScreenshot,
  onPrint,
  onFullscreen,
  isFullscreen,

  // Datos compartidos
  uploadedFiles,
  filesInfo,
  volumes,
  availableRadars,
  savedLayers,
  fieldsUsed,
  filtersUsed,
  filtersPerField = {}, // { FIELD: [{field, min, max}] } filtros por campo
  onApplyFilters, // (filtersPerField) => void
  activeElevation,
  activeHeight,
  radarSite,
  warnings,
  availableDownloads,
  product, // El producto actual (PPI, CAPPI, etc.)

  // File manager
  fileManagerOpen,
  setFileManagerOpen,
  onRemoveFile,

  // Refs
  drawnLayerRef,

  // Split screen props
  isSplitScreen = false,
  showSplitButton = true,
  showLockButton = false,
  locked = false,
  onToggleSplit,
  onToggleLock,
  loading = false,
  onSettingsOpen,
  hideFilesButton = false,
}) {
  // Estado local para la instancia del mapa
  const [localMapInstance, setLocalMapInstance] = useState(null);

  const hasValidRhiLine =
    rhiLinePreview?.start &&
    rhiLinePreview?.end &&
    Number.isFinite(rhiLinePreview.start.lat) &&
    Number.isFinite(rhiLinePreview.start.lon) &&
    Number.isFinite(rhiLinePreview.end.lat) &&
    Number.isFinite(rhiLinePreview.end.lon);

  // ── Estado local para anotaciones visuales ──────────────────────────────────
  const [annotationMode, setAnnotationModeState] = useState(null);
  const [textAnnotations, setTextAnnotations] = useState([]);
  const [shapeAnnotations, setShapeAnnotations] = useState([]);
  const [rangeCircleLayers, setRangeCircleLayers] = useState(new Set());
  const [elevationProfilePoints, setElevationProfilePoints] = useState([]);
  const [rhiElevationProfilePoints, setRhiElevationProfilePoints] = useState(
    [],
  );
  const elevationProfilePointsRef = useRef([]);
  const rhiElevationProfilePointsRef = useRef([]);

  const availableRangeCircleKeys = useMemo(() => {
    if (!Array.isArray(mergedOutputs)) return new Set();

    return new Set(
      mergedOutputs
        .flatMap((frame) => (Array.isArray(frame) ? frame : []))
        .map((layer) => getRangeCircleKey(layer))
        .filter((key) => key !== "::"),
    );
  }, [mergedOutputs]);

  useEffect(() => {
    setRangeCircleLayers((prev) => {
      const next = new Set(
        [...prev].filter((key) => availableRangeCircleKeys.has(key)),
      );
      return next.size === prev.size ? prev : next;
    });
  }, [availableRangeCircleKeys]);

  useEffect(() => {
    elevationProfilePointsRef.current = elevationProfilePoints;
  }, [elevationProfilePoints]);

  useEffect(() => {
    rhiElevationProfilePointsRef.current = rhiElevationProfilePoints;
  }, [rhiElevationProfilePoints]);

  const filesMetadataByPath = useMemo(
    () =>
      new Map(
        (filesInfo || []).map((fileInfo) => [
          fileInfo.filepath,
          fileInfo.metadata || {},
        ]),
      ),
    [filesInfo],
  );

  const rangeCircleShapes = useMemo(() => {
    if (!Array.isArray(allLayersOverlay) || allLayersOverlay.length === 0) {
      return [];
    }

    return allLayersOverlay
      .filter((layer) => {
        const key = getRangeCircleKey(layer);
        return rangeCircleLayers.has(key) && !hiddenLayers?.has(key);
      })
      .map((layer) => {
        const layerMetadata = layer?.metadata || {};
        const fallbackMetadata =
          filesMetadataByPath.get(layer?.source_file) || {};
        const radarSite =
          layerMetadata.radar_site ||
          layerMetadata.site ||
          fallbackMetadata.radar_site ||
          fallbackMetadata.site;
        const lastGateRangeM = Number(
          layerMetadata.last_gate_range_m ??
            layerMetadata.range_max_m ??
            fallbackMetadata.last_gate_range_m ??
            fallbackMetadata.range_max_m,
        );

        if (
          !radarSite ||
          !Number.isFinite(Number(radarSite.lat)) ||
          !Number.isFinite(Number(radarSite.lon)) ||
          !Number.isFinite(lastGateRangeM) ||
          lastGateRangeM <= 0
        ) {
          return null;
        }

        return {
          id: `range-circle:${getRangeCircleKey(layer)}`,
          type: "circle",
          center: {
            lat: Number(radarSite.lat),
            lon: Number(radarSite.lon),
          },
          radius: lastGateRangeM,
          interactive: false,
          style: {
            color: "#0f766e",
            fillColor: "#0f766e",
            fillOpacity: 0,
            weight: 2,
            opacity: 0.9,
            dashArray: "10,6",
          },
        };
      })
      .filter(Boolean);
  }, [allLayersOverlay, filesMetadataByPath, hiddenLayers, rangeCircleLayers]);

  const handleSetAnnotationMode = (mode) => {
    setAnnotationModeState((prev) => {
      const next = prev === mode ? null : mode;
      return next;
    });
    // Desactivar marcadores al activar una anotación
    if (mode && markerMode) setMarkerMode(false);
  };

  const handleToggleMarkerModeLocal = () => {
    // Limpiar modo anotación al activar marcadores
    setAnnotationModeState(null);
    onToggleMarkerMode?.();
  };

  const handleTextAdd = (ann) => setTextAnnotations((prev) => [...prev, ann]);
  const handleTextUpdate = (id, patch) =>
    setTextAnnotations((prev) =>
      prev.map((a) => (a.id === id ? { ...a, ...patch } : a)),
    );
  const handleTextRemove = (id) =>
    setTextAnnotations((prev) => prev.filter((a) => a.id !== id));

  const handleDeactivateAnnotationMode = () => setAnnotationModeState(null);

  const handleDeactivateMarkerMode = () => setMarkerMode(false);

  const handleUpdateMarker = (id, patch) =>
    setMarkers((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    );

  const handleShapeAdd = (shape) =>
    setShapeAnnotations((prev) => [...prev, shape]);
  const handleShapeUpdate = (id, patch) =>
    setShapeAnnotations((prev) =>
      prev.map((s) => (s.id === id ? { ...s, ...patch } : s)),
    );
  const handleShapeRemove = (id) =>
    setShapeAnnotations((prev) => prev.filter((s) => s.id !== id));
  const handleToggleRangeCircle = (layer) => {
    const key = getRangeCircleKey(layer);

    setRangeCircleLayers((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  // ────────────────────────────────────────────────────────────────────────────

  // Wrapper para onMapReady que actualiza tanto el estado local como el externo
  const handleMapReady = (map) => {
    setLocalMapInstance(map);
    onMapReady?.(map);
  };

  // Handlers locales para diálogos
  const handleOpenRHI = () => setRhiOpen(true);

  const handleRequestPickPoint = () => {
    setPickedPoint(null);
    setPickPointMode(true);
  };

  const handlePickPoint = (pt) => {
    setPickedPoint(pt);
  };

  const handleClearPickedPoint = () => {
    setPickedPoint(null);
    setPickPointMode(false);
    setRhiLinePreview({ start: null, end: null });
    setRhiElevationProfilePoints([]);
    setHighlightedPoint(null);
  };

  const handleClearLineOverlay = () => {
    setRhiLinePreview({ start: null, end: null });
    setRhiElevationProfilePoints([]);
    setHighlightedPoint(null);
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
    try {
      drawnLayerRef.current?.remove();
    } catch {
      console.log("Error");
    }
    drawnLayerRef.current = null;
    setAreaStatsOpen(false);
  };

  const handleTogglePixelStat = () => {
    setPixelStatMode((v) => {
      const next = !v;
      if (!next) setPixelStatMarker(null);
      return next;
    });
  };

  const handleToggleMapSelector = () => {
    setMapSelectorOpen((prev) => !prev);
  };

  const handleSelectBaseMap = (map) => {
    setSelectedBaseMap(map);
  };

  const handleTogglePaletteSelector = () => {
    setPaletteSelectorOpen((prev) => !prev);
  };

  const handleSelectColormap = (field, colormap) => {
    setSelectedColormaps((prev) => ({
      ...prev,
      [field]: colormap,
    }));
  };

  const handleApplyColormaps = () => {
    setPaletteSelectorOpen(false);
    setSelectorOpen(true);
  };

  const handleToggleLayerManager = () => {
    setLayerManagerOpen((prev) => !prev);
  };

  const handleToggleFileManager = () => {
    setFileManagerOpen?.((prev) => !prev);
  };

  const handleRequestLineDrawing = () => {
    setDrawnLineCoords([]);
    setLineDrawingFinished(false);
    setElevationProfilePoints([]);
    setRhiElevationProfilePoints([]);
    setHighlightedPoint(null);
    setLineDrawMode(true);
  };

  const handleLineComplete = () => {
    setLineDrawMode(false);
    setLineDrawingFinished(true);
  };

  const handleClearLineDrawing = () => {
    setDrawnLineCoords([]);
    setLineDrawMode(false);
    setLineDrawingFinished(false);
    setElevationProfilePoints([]);
    setHighlightedPoint(null);
  };

  const handleHighlightPoint = useCallback(
    (lat, lon) => {
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        setHighlightedPoint(null);
        return;
      }

      // Leemos desde refs para mantener este callback estable y evitar que los
      // efectos de los diálogos entren en loops al cambiar la identidad del handler.
      const matchingPoint = [
        ...elevationProfilePointsRef.current,
        ...rhiElevationProfilePointsRef.current,
      ].find((point) => point.lat === lat && point.lon === lon);

      setHighlightedPoint(matchingPoint || { lat, lon });
    },
    [setHighlightedPoint],
  );

  const handleProfileGenerated = useCallback(
    (profilePoints = []) => {
      setLineDrawingFinished(false);
      const nextProfilePoints = Array.isArray(profilePoints)
        ? profilePoints
        : [];
      setElevationProfilePoints((prevPoints) =>
        areProfilePointsEqual(prevPoints, nextProfilePoints)
          ? prevPoints
          : nextProfilePoints,
      );
      if (nextProfilePoints.length === 0) {
        setHighlightedPoint(null);
      }
    },
    [setLineDrawingFinished, setHighlightedPoint],
  );

  const handleRhiProfileChange = useCallback(
    (profilePoints = []) => {
      const nextProfilePoints = Array.isArray(profilePoints)
        ? profilePoints
        : [];
      setRhiElevationProfilePoints((prevPoints) =>
        areProfilePointsEqual(prevPoints, nextProfilePoints)
          ? prevPoints
          : nextProfilePoints,
      );
      if (nextProfilePoints.length === 0) {
        setHighlightedPoint(null);
      }
    },
    [setHighlightedPoint],
  );

  const handleOpenElevationProfile = () => {
    setElevationProfileOpen(true);
  };

  // Analizar campos en todos los archivos para identificar campos comunes vs específicos
  // Los campos comunes son los que están presentes en todos los archivos
  // Los campos específicos son los que solo están en algunos archivos
  const fieldAnalysis = useMemo(() => {
    return analyzeFieldsAcrossFiles(filesInfo);
  }, [filesInfo]);

  return (
    <div
      id={`map-container-${panelId}`}
      style={{
        height: "100vh",
        width: "100%",
        position: "relative",
      }}
    >
      <MapView
        overlayData={overlayData}
        opacities={opacity}
        opacityByField={opacityByField}
        opacityByLayer={opacityByLayer}
        pickPointMode={pickPointMode}
        radarSite={radarSite}
        pickedPoint={pickedPoint}
        onPickPoint={handlePickPoint}
        drawAreaMode={areaDrawMode}
        onAreaComplete={handleAreaComplete}
        pixelStatMode={pixelStatMode}
        onPixelStatClick={onPixelStatClick}
        pixelStatMarker={pixelStatMarker}
        lineOverlay={
          hasValidRhiLine
            ? [
                [rhiLinePreview.start.lat, rhiLinePreview.start.lon],
                [rhiLinePreview.end.lat, rhiLinePreview.end.lon],
              ]
            : null
        }
        onClearLineOverlay={handleClearLineOverlay}
        rhiEndpoints={{ start: rhiLinePreview.start, end: rhiLinePreview.end }}
        onMapReady={handleMapReady}
        baseMapUrl={selectedBaseMap.url}
        baseMapAttribution={selectedBaseMap.attribution}
        lineDrawMode={lineDrawMode}
        drawnLineCoords={drawnLineCoords}
        onLineComplete={handleLineComplete}
        onLinePointsChange={setDrawnLineCoords}
        elevationProfilePoints={elevationProfilePoints}
        onLineHoverPoint={setHighlightedPoint}
        rhiProfilePoints={rhiElevationProfilePoints}
        onRhiLineHoverPoint={setHighlightedPoint}
        highlightedPoint={highlightedPoint}
        markerMode={markerMode}
        markers={markers}
        onAddMarker={onAddMarker}
        onRemoveMarker={onRemoveMarker}
        onRenameMarker={onRenameMarker}
        onUpdateMarker={handleUpdateMarker}
        onMarkerModeDeactivate={handleDeactivateMarkerMode}
        annotationMode={annotationMode}
        textAnnotations={textAnnotations}
        onTextAdd={handleTextAdd}
        onTextUpdate={handleTextUpdate}
        onTextRemove={handleTextRemove}
        onTextModeDeactivate={handleDeactivateAnnotationMode}
        shapeAnnotations={shapeAnnotations}
        rangeCircleShapes={rangeCircleShapes}
        onShapeAdd={handleShapeAdd}
        onShapeUpdate={handleShapeUpdate}
        onShapeRemove={handleShapeRemove}
        onShapeModeDeactivate={handleDeactivateAnnotationMode}
      />

      <VerticalToolbar
        onChangeProductClick={() => setSelectorOpen(true)}
        onPseudoRhiClick={handleOpenRHI}
        onAreaStatsClick={handleOpenAreaStatsMode}
        onPixelStatToggle={handleTogglePixelStat}
        onMapSelectorToggle={handleToggleMapSelector}
        onPaletteSelectorToggle={handleTogglePaletteSelector}
        onElevationProfileClick={handleOpenElevationProfile}
        onLayerManagerToggle={handleToggleLayerManager}
        onFileManagerToggle={handleToggleFileManager}
        areaStatsActive={areaDrawMode || areaStatsOpen}
        pixelStatActive={pixelStatMode}
        mapSelectorActive={mapSelectorOpen}
        paletteSelectorActive={paletteSelectorOpen}
        layerManagerActive={layerManagerOpen}
        fileManagerActive={fileManagerOpen}
        hideFilesButton={hideFilesButton}
      />

      <MapToolbar
        onScreenshot={onScreenshot}
        onPrint={onPrint}
        onFullscreen={onFullscreen}
        isFullscreen={isFullscreen}
        availableDownloads={availableDownloads}
        isSplitScreen={isSplitScreen}
        showSplitButton={showSplitButton}
        showLockButton={showLockButton}
        locked={locked}
        onToggleSplit={onToggleSplit}
        onToggleLock={onToggleLock}
        onSettingsOpen={onSettingsOpen}
      />

      <DrawingToolbar
        markerMode={markerMode}
        onToggleMarkerMode={handleToggleMarkerModeLocal}
        annotationMode={annotationMode}
        onSetAnnotationMode={handleSetAnnotationMode}
      />

      <BaseMapSelector
        open={mapSelectorOpen}
        onClose={() => setMapSelectorOpen(false)}
        selectedMap={selectedBaseMap}
        onSelectMap={handleSelectBaseMap}
      />

      <ColorPaletteSelector
        open={paletteSelectorOpen}
        onClose={() => setPaletteSelectorOpen(false)}
        selectedColormaps={selectedColormaps}
        onSelectColormap={handleSelectColormap}
        availableFields={fieldsUsed}
        onApply={handleApplyColormaps}
        hasLoadedImages={Array.isArray(overlayData) && overlayData.length > 0}
        initialColormaps={initialColormaps}
      />

      <LayerManagerDialog
        open={layerManagerOpen}
        onClose={() => setLayerManagerOpen(false)}
        layers={
          Array.isArray(allLayersOverlay)
            ? allLayersOverlay
            : Array.isArray(overlayData)
              ? overlayData
              : []
        }
        onReorder={onLayerReorder}
        onToggleLayerVisibility={onToggleLayerVisibility}
        hiddenLayers={hiddenLayers}
        opacityByField={opacityByField}
        opacityByLayer={opacityByLayer}
        onLayerOpacityChange={onLayerOpacityChange}
        rangeCirclesVisible={rangeCircleLayers}
        onToggleLayerRangeCircle={handleToggleRangeCircle}
        filtersPerField={filtersPerField}
        onApplyFilters={onApplyFilters}
      />

      <FileManagerDialog
        open={fileManagerOpen}
        onClose={() => setFileManagerOpen?.(false)}
        filesInfo={filesInfo}
        onRemoveFile={onRemoveFile}
      />

      <ZoomControls map={localMapInstance} bottomOffset={46} />

      <ColorLegend overlayData={overlayData} />

      {Array.isArray(mergedOutputs) && mergedOutputs.length > 0 && (
        <AnimationControls
          overlayData={{ outputs: mergedOutputs, animation, product }}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
          showPlayButton={animation}
          isSplitScreen={isSplitScreen}
        />
      )}

      <ProductSelectorDialog
        open={selectorOpen}
        fieldAnalysis={fieldAnalysis}
        elevations={
          filesInfo.length > 0
            ? filesInfo.reduce(
                (longest, f) =>
                  f.metadata.elevations.length > longest.length
                    ? f.metadata.elevations
                    : longest,
                filesInfo[0].metadata.elevations || [],
              )
            : []
        }
        volumes={volumes}
        radars={availableRadars}
        initialLayers={savedLayers}
        onClose={() => setSelectorOpen(false)}
        onConfirm={onProductChosen}
      />

      <PseudoRHIDialog
        open={rhiOpen}
        onClose={() => setRhiOpen(false)}
        // El archivo del radar con mayor prioridad (primera capa del overlay) se usa como fuente para herramientas
        filepath={overlayData?.[0]?.source_file || uploadedFiles[currentIndex]}
        radarSite={radarSite}
        fields_present={(() => {
          const fields = Array.from(
            new Set(filesInfo.map((f) => f.metadata.fields_present).flat()),
          );
          return fields.length > 0 ? fields : ["DBZH", "KDP", "RHOHV", "ZDR"];
        })()}
        onRequestPickPoint={handleRequestPickPoint}
        pickedPoint={pickedPoint}
        onClearPickedPoint={handleClearPickedPoint}
        onGenerate={onGenerateRHI}
        onLinePreviewChange={setRhiLinePreview}
        onElevationProfileChange={handleRhiProfileChange}
        onHighlightPoint={handleHighlightPoint}
        highlightedPoint={highlightedPoint}
        onAutoClose={() => setRhiOpen(false)}
        onAutoReopen={() => setRhiOpen(true)}
      />

      <AreaStatsDialog
        open={areaStatsOpen}
        onClose={handleCloseAreaStats}
        requestFn={onAreaStatsRequest}
        payload={{
          // Archivo del radar con mayor prioridad (primera capa del overlay)
          filepath:
            overlayData?.[0]?.source_file || uploadedFiles[currentIndex],
          field: fieldsUsed?.[0] || "DBZH",
          product: product || "PPI",
          elevation: activeElevation,
          height: activeHeight,
          filters: filtersUsed,
          polygon: areaPolygon,
        }}
        fields_present={(() => {
          const fields = Array.from(
            new Set(filesInfo.map((f) => f.metadata.fields_present).flat()),
          );
          return fields.length > 0 ? fields : ["DBZH", "KDP", "RHOHV", "ZDR"];
        })()}
      />

      <ElevationProfileDialog
        open={elevationProfileOpen}
        onClose={() => {
          setElevationProfileOpen(false);
          handleClearLineDrawing();
        }}
        onRequestDraw={handleRequestLineDrawing}
        drawnCoordinates={drawnLineCoords}
        drawingFinished={lineDrawingFinished}
        onGenerate={onGenerateElevationProfile}
        onClearDrawing={handleClearLineDrawing}
        onHighlightPoint={handleHighlightPoint}
        onProfileGenerated={handleProfileGenerated}
        highlightedPoint={highlightedPoint}
      />

      <WarningPanel warnings={warnings} />
      <Loader open={loading} />
    </div>
  );
}
