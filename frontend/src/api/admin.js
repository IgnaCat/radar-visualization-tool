import axios from "axios";
import { getApiBaseUrl } from "./backend";

function makeAdminApi(token) {
  return axios.create({
    baseURL: getApiBaseUrl(),
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function fetchUsers(token) {
  const resp = await makeAdminApi(token).get("/admin/users");
  return resp.data;
}

export async function createUser(token, { username, password, role }) {
  const resp = await makeAdminApi(token).post("/admin/users", { username, password, role });
  return resp.data;
}

export async function updateUser(token, userId, data) {
  const resp = await makeAdminApi(token).patch(`/admin/users/${userId}`, data);
  return resp.data;
}

export async function fetchAccessLogs(token, { userId, limit } = {}) {
  const params = new URLSearchParams();
  if (userId) params.set("user_id", userId);
  if (limit) params.set("limit", limit);
  const resp = await makeAdminApi(token).get(`/admin/access-logs?${params}`);
  return resp.data;
}

export async function fetchActiveSessions(token) {
  const resp = await makeAdminApi(token).get("/admin/sessions/active");
  return resp.data;
}

export async function forceCleanupUser(token, userId) {
  const resp = await makeAdminApi(token).post(`/admin/cleanup/${userId}`);
  return resp.data;
}

export async function fetchLogs(token, { lines = 200, level, search } = {}) {
  const params = new URLSearchParams();
  params.set("lines", lines);
  if (level) params.set("level", level);
  if (search) params.set("search", search);
  const resp = await makeAdminApi(token).get(`/admin/logs?${params}`);
  return resp.data;
}
