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
  //Busca la primera layer enabled para usar su field
  let field = "DBZH";
  for (let layer of layers) {
    if (layer.enabled) {
      field = layer.label;
      break;
    }
  }

  const payload = {
    filepaths: files,
    product: product,
    field: field,
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
