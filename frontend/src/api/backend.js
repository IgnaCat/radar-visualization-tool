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
}) => {
  const payload = {
    filepaths: files,
    product: product,
    field: layers.find((l) => l.enabled)?.label || "DBZH",
    ...(height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
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
