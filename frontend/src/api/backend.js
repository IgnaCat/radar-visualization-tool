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
  product,
  height,
  elevation,
  filters,
}) => {
  const payload = {
    filepaths: files,
    product: product,
    ...(height !== undefined && { height: parseInt(height) }),
    ...(elevation !== undefined && { elevation: parseInt(elevation) }),
    ...(filters && { filters }),
  };

  console.log("processFile payload:", payload);

  return api.post("/process", payload);
};

export function cleanupClose(payload) {
  // payload: { uploads: string[], cogs: string[], delete_cache: boolean }
  return api.post("/cleanup/close", payload, {
    headers: { "Content-Type": "application/json" },
  });
}
