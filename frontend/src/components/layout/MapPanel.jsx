import { useState, useMemo } from "react";
import MapView from "../map/MapView";
import VerticalToolbar from "../controls/VerticalToolbar";
import MapToolbar from "../controls/MapToolbar";
import ZoomControls from "../controls/ZoomControls";
import ColorLegend from "../map/ColorLegend";
import BaseMapSelector from "../map/BaseMapSelector";
import ColorPaletteSelector from "../controls/ColorPaletteSelector";
import LayerManagerDialog from "../dialogs/LayerManagerDialog";
import FileManagerDialog from "../dialogs/FileManagerDialog";
import AnimationControls from "../controls/AnimationControls";
import ActiveLayerPicker from "../controls/ActiveLayerPicker";
import ProductSelectorDialog from "../dialogs/ProductSelectorDialog";
import PseudoRHIDialog from "../dialogs/PseudoRHIDialog";
import AreaStatsDialog from "../dialogs/AreaStatsDialog";
import ElevationProfileDialog from "../dialogs/ElevationProfileDialog";
import WarningPanel from "../ui/WarningPanel";
import Loader from "../ui/Loader";
import { analyzeFieldsAcrossFiles } from "../../utils/fieldAnalysis";

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
  setInitialColormaps,

  // Handlers de acciones
  onProductChosen,
  onGenerateRHI,
  onAreaStatsRequest,
  onPixelStatClick,
  onGenerateElevationProfile,
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
  activeElevation,
  activeHeight,
  activeToolFile,
  setActiveToolFile,
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
}) {
  // Estado local para la instancia del mapa
  const [localMapInstance, setLocalMapInstance] = useState(null);

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
  };

  const handleClearLineOverlay = () => {
    setRhiLinePreview({ start: null, end: null });
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
    setHighlightedPoint(null);
  };

  const handleHighlightPoint = (point) => {
    setHighlightedPoint(point);
  };

  const handleProfileGenerated = () => {
    setLineDrawingFinished(false);
  };

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
          rhiLinePreview?.start && rhiLinePreview?.end
            ? [
                [rhiLinePreview.start.lat, rhiLinePreview.start.lon],
                [rhiLinePreview.end.lat, rhiLinePreview.end.lon],
              ]
            : null
        }
        onClearLineOverlay={handleClearLineOverlay}
        rhiEndpoints={{ start: rhiLinePreview.start, end: rhiLinePreview.end }}
        activeToolFile={activeToolFile}
        onMapReady={handleMapReady}
        baseMapUrl={selectedBaseMap.url}
        baseMapAttribution={selectedBaseMap.attribution}
        lineDrawMode={lineDrawMode}
        drawnLineCoords={drawnLineCoords}
        onLineComplete={handleLineComplete}
        onLinePointsChange={setDrawnLineCoords}
        highlightedPoint={highlightedPoint}
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
        pixelStatActive={pixelStatMode}
        mapSelectorActive={mapSelectorOpen}
        paletteSelectorActive={paletteSelectorOpen}
        layerManagerActive={layerManagerOpen}
        fileManagerActive={fileManagerOpen}
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
        opacityByLayer={opacityByLayer}
        onLayerOpacityChange={onLayerOpacityChange}
      />

      <FileManagerDialog
        open={fileManagerOpen}
        onClose={() => setFileManagerOpen?.(false)}
        filesInfo={filesInfo}
        onRemoveFile={onRemoveFile}
      />

      <ZoomControls map={localMapInstance} />

      {activeToolFile && (
        <ActiveLayerPicker
          layers={Array.isArray(overlayData) ? overlayData : []}
          value={activeToolFile}
          onChange={setActiveToolFile}
        />
      )}

      <ColorLegend overlayData={overlayData} />

      {Array.isArray(mergedOutputs) && mergedOutputs.length > 0 && (
        <AnimationControls
          overlayData={{ outputs: mergedOutputs, animation, product }}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
          showPlayButton={animation}
        />
      )}

      <ProductSelectorDialog
        open={selectorOpen}
        fieldAnalysis={fieldAnalysis}
        elevations={Array.from(
          new Set(filesInfo.map((f) => f.metadata.elevations).flat()),
        )}
        volumes={volumes}
        radars={availableRadars}
        initialLayers={savedLayers}
        onClose={() => setSelectorOpen(false)}
        onConfirm={onProductChosen}
      />

      <PseudoRHIDialog
        open={rhiOpen}
        onClose={() => setRhiOpen(false)}
        filepath={activeToolFile || uploadedFiles[currentIndex]}
        radarSite={radarSite}
        fields_present={
          Array.from(
            new Set(filesInfo.map((f) => f.metadata.fields_present).flat()),
          ) || ["DBZH", "KDP", "RHOHV", "ZDR"]
        }
        onRequestPickPoint={handleRequestPickPoint}
        pickedPoint={pickedPoint}
        onClearPickedPoint={handleClearPickedPoint}
        onGenerate={onGenerateRHI}
        onLinePreviewChange={setRhiLinePreview}
        onAutoClose={() => setRhiOpen(false)}
        onAutoReopen={() => setRhiOpen(true)}
      />

      <AreaStatsDialog
        open={areaStatsOpen}
        onClose={handleCloseAreaStats}
        requestFn={onAreaStatsRequest}
        payload={{
          filepath: activeToolFile || uploadedFiles[currentIndex],
          field: fieldsUsed?.[0] || "DBZH",
          product: product || "PPI",
          elevation: activeElevation,
          height: activeHeight,
          filters: filtersUsed,
          polygon: areaPolygon,
        }}
        fields_present={
          Array.from(
            new Set(filesInfo.map((f) => f.metadata.fields_present).flat()),
          ) || ["DBZH", "KDP", "RHOHV", "ZDR"]
        }
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
      />

      <WarningPanel warnings={warnings} />
      <Loader open={loading} />
    </div>
  );
}
