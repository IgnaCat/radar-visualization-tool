import axios from "axios";
import { getApiBaseUrl } from "./backend";

const api = axios.create({ baseURL: getApiBaseUrl() });

export async function loginApi({ username, password, session_id }) {
  const resp = await api.post("/auth/login", { username, password, session_id });
  return resp.data; // { access_token, token_type, user }
}

export async function logoutApi({ session_id, token }) {
  return api.post(
    "/auth/logout",
    { session_id },
    { headers: { Authorization: `Bearer ${token}` } },
  );
}
