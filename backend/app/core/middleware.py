import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from .security import decode_access_token

access_logger = logging.getLogger("app.access")

class CustomAccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Determine User IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        # Attempt to get username from Authorization header
        username = "anon"
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = decode_access_token(token)
                username = payload.get("username", "anon")
            except Exception:
                pass  # Validation is handled by auth dependency

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            # Let FastAPI handle the actual 500 response, but we log the error status
            raise e
        finally:
            process_time = time.time() - start_time
            
            # Log format: INFO [app.access] [User: admin] [IP: 1.2.3.4] POST /process - Status: 200 - Tiempo: 0.12s
            if request.url.path != "/health" and request.method != "OPTIONS":
                access_logger.info(
                    f"[User: {username}] [IP: {client_ip}] {request.method} {request.url.path} - Status: {status_code} - Tiempo: {process_time:.2f}s"
                )
        return response
