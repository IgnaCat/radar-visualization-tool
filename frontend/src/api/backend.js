import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
});

export const uploadFile = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return api.post("/upload", formData);
};

export const processFile = async ({
  files,
  layers,
  product,
  height,
  elevation,
  filters,
  selectedVolumes,
}) => {
  const payload = {
    filepaths: files,
    product: product,
    fields: layers,
    ...(height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
    ...(selectedVolumes && { selectedVolumes }),
  };

  return api.post("/process", payload);
};

export function cleanupClose(payload) {
  // payload: { uploads: string[], cogs: string[], delete_cache: boolean }
  return api.post("/cleanup/close", payload, {
    headers: { "Content-Type": "application/json" },
  });
}

export async function generatePseudoRHI({
  filepath,
  field,
  end_lon,
  end_lat,
  max_length_km = 240,
  elevation = 0,
  filters = [],
  png_width_px = 900,
  png_height_px = 500,
}) {
  return api.post("/process/pseudo_rhi", {
    filepaths: [filepath],
    field,
    end_lon,
    end_lat,
    max_length_km,
    elevation,
    filters,
    png_width_px,
    png_height_px,
  });
}

export async function generateAreaStats(payload) {
  const { polygon, filepath, product, field, height, elevation, filters } =
    payload;

  console.log("area", payload)
  return api.post("/stats/area", {
    polygon_geojson: polygon,
    filepath,
    product,
    field,
    ...(height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
  });
}

export async function generatePixelStat(payload) {
  const { filepath, product, field, height, elevation, filters, lat, lon } = payload;

  console.log(payload)
  return api.post("/stats/pixel", {
    filepath,
    product,
    field,
    ...(height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
    lat,
    lon
  });
}
