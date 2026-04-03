import axios from "axios";

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

function isLoopbackHost(hostname) {
  return LOOPBACK_HOSTS.has(String(hostname || "").toLowerCase());
}

export function getApiBaseUrl() {
  const configured = String(import.meta.env.VITE_API_URL || "").trim();

  if (typeof window === "undefined") {
    return configured || "http://localhost:8000";
  }

  const { location } = window;
  const pageIsLocal = isLoopbackHost(location.hostname);

  if (!configured) {
    return pageIsLocal ? "http://localhost:8000" : location.origin;
  }

  try {
    const parsed = new URL(configured, location.origin);

    if (!pageIsLocal && isLoopbackHost(parsed.hostname)) {
      return location.origin;
    }

    return parsed.toString().replace(/\/$/, "");
  } catch {
    return configured.replace(/\/$/, "");
  }
}

export function resolveApiUrl(pathOrUrl = "") {
  const value = String(pathOrUrl || "").trim();
  if (!value) return getApiBaseUrl();

  if (/^https?:\/\//i.test(value)) {
    if (typeof window === "undefined") return value;

    try {
      const parsed = new URL(value);
      if (
        !isLoopbackHost(window.location.hostname) &&
        isLoopbackHost(parsed.hostname)
      ) {
        return `${window.location.origin}${parsed.pathname}${parsed.search}${parsed.hash}`;
      }
    } catch {
      return value;
    }

    return value;
  }

  const baseUrl = getApiBaseUrl().replace(/\/+$/, "");
  return `${baseUrl}/${value.replace(/^\/+/, "")}`;
}

const api = axios.create({
  baseURL: getApiBaseUrl(),
});

export const uploadFile = async (files, session_id = null) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  if (session_id) {
    formData.append("session_id", session_id);
  }
  return api.post("/upload", formData);
};

export const processFile = async ({
  files,
  layers,
  product,
  height,
  elevation,
  filters,
  filters_per_field,
  selectedVolumes,
  selectedRadars,
  colormap_overrides,
  session_id,
  weight_func,
  max_neighbors,
  smoothing,
}) => {
  const payload = {
    filepaths: files,
    product: product,
    fields: layers,
    ...(height !== null &&
      height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined &&
      elevation !== null && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
    ...(filters_per_field &&
      Object.keys(filters_per_field).length > 0 && { filters_per_field }),
    ...(selectedVolumes && { selectedVolumes }),
    ...(selectedRadars && { selectedRadars }),
    ...(colormap_overrides && { colormap_overrides }),
    ...(session_id && { session_id }),
    ...(weight_func && { weight_func }),
    ...(max_neighbors != null && { max_neighbors: parseInt(max_neighbors) }),
    ...(smoothing && { smoothing }),
  };

  return api.post("/process", payload);
};

export async function generateAnimationGif({
  frames,
  fps = 1,
  session_id,
  basemap_id,
}) {
  return api.post("/process/animation/gif", {
    frames,
    fps,
    ...(session_id && { session_id }),
    ...(basemap_id && { basemap_id }),
  });
}

export function cleanupClose(payload) {
  // payload: { uploads: string[], cogs: string[], delete_cache: boolean, session_id?: string }
  return api.post("/cleanup/close", payload, {
    headers: { "Content-Type": "application/json" },
  });
}

export async function generatePseudoRHI({
  filepath,
  field,
  end_lon,
  end_lat,
  start_lon,
  start_lat,
  max_length_km,
  max_height_km,
  min_length_km = 0,
  min_height_km = 0,
  elevation = 0,
  filters = [],
  png_width_px = 900,
  png_height_px = 500,
  colormap_overrides,
  weight_func,
  max_neighbors,
  smoothing,
  session_id,
}) {
  return api.post("/process/pseudo_rhi", {
    filepaths: [filepath],
    field,
    end_lon,
    end_lat,
    ...(start_lon != null &&
      start_lat != null && {
        start_lon,
        start_lat,
      }),
    max_length_km: max_length_km,
    max_height_km: max_height_km,
    min_length_km: min_length_km,
    min_height_km: min_height_km,
    elevation,
    filters,
    png_width_px,
    png_height_px,
    ...(colormap_overrides && { colormap_overrides }),
    ...(weight_func && { weight_func }),
    ...(max_neighbors != null && { max_neighbors: parseInt(max_neighbors) }),
    ...(smoothing && { smoothing }),
    ...(session_id && { session_id }),
  });
}

export async function generateAreaStats(payload) {
  const {
    polygon,
    filepath,
    product,
    field,
    height,
    elevation,
    filters,
    session_id,
    weight_func,
    max_neighbors,
  } = payload;

  return api.post("/stats/area", {
    polygon_geojson: polygon,
    filepath,
    product,
    field,
    ...(height !== null &&
      height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined &&
      elevation !== null && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
    ...(session_id && { session_id }),
    ...(weight_func && { weight_func }),
    ...(max_neighbors != null && { max_neighbors }),
  });
}

export async function generatePixelStat(payload) {
  const {
    filepath,
    product,
    field,
    height,
    elevation,
    filters,
    lat,
    lon,
    session_id,
    weight_func,
    max_neighbors,
    render_zoom,
    render_native_zoom,
  } = payload;

  return api.post("/stats/pixel", {
    filepath,
    product,
    field,
    ...(height !== null &&
      height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined &&
      elevation !== null && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
    lat,
    lon,
    ...(session_id && { session_id }),
    ...(weight_func && { weight_func }),
    ...(max_neighbors != null && { max_neighbors }),
    ...(render_zoom != null && { render_zoom: parseInt(render_zoom) }),
    ...(render_native_zoom != null && {
      render_native_zoom: parseInt(render_native_zoom),
    }),
  });
}

export async function generateElevationProfile({
  coordinates,
  interpolate = true,
  points_per_km = 10,
}) {
  return api.post("/stats/elevation_profile", {
    coordinates,
    interpolate,
    points_per_km,
  });
}

export async function getColormapOptions() {
  const response = await api.get("/colormap/options");
  return response.data;
}

export async function getColormapDefaults() {
  const response = await api.get("/colormap/defaults");
  return response.data;
}

export async function getColormapColors(cmapName, steps = 256) {
  const response = await api.get(`/colormap/colors/${cmapName}?steps=${steps}`);
  return response.data;
}

export async function getColormapLegend(field, cmapName = null) {
  const query = cmapName ? `?cmap_name=${encodeURIComponent(cmapName)}` : "";
  const response = await api.get(
    `/colormap/legend/${encodeURIComponent(field)}${query}`,
  );
  return response.data;
}

export async function getCacheStats() {
  const response = await api.get("/admin/cache-stats");
  return response.data;
}

export async function clearCache(cacheType = "all") {
  const response = await api.post(`/admin/clear-cache?cache_type=${cacheType}`);
  return response.data;
}

/**
 * Elimina archivos subidos específicos del servidor.
 * Borra el NetCDF, COGs asociados y entradas de cache.
 * @param {string[]} filepaths - Rutas de archivos a eliminar
 * @param {string|null} sessionId - ID de sesión
 * @returns {{ deleted: object, removed_files: string[] }}
 */
export async function removeFiles(filepaths, sessionId = null) {
  return api.post("/cleanup/files", {
    filepaths,
    ...(sessionId && { session_id: sessionId }),
  });
}
