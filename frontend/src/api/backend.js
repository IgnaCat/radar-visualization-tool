import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000",
});

export const uploadFile = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return api.post("/upload", formData);
};

export const processFile = async (filepaths) => {
  return api.post("/process", {
    filepaths: filepaths,
  });
};
