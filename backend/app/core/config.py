from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "Radar Visualization"
    BASE_URL: str = "http://localhost:8000"
    FRONTEND_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Storage directories (set via env vars in Docker, defaults for local)
    IMAGES_DIR: str = os.path.join(os.getcwd(), "app/storage/tmp")
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "app/storage/uploads")
    DATA_DIR: str = os.path.join(os.getcwd(), "app/storage/data")
    CACHE_DIR: str = os.path.join(os.getcwd(), "app/storage/cache")
    ALLOWED_PRODUCTS: List[str] = ["PPI", "RHI", "CAPPI", "COLMAX"]

    # Reglas de upload
    ALLOWED_EXTENSIONS: List[str] = [".nc", ".BUFR", ".bufr"]
    MAX_UPLOAD_MB: int = 500

    # Archivo NetCDF de demo (modo admin): se copia a la sesión sin depender de una
    # subida HTTP. Default: fixture ya horneado en la imagen. Override por env var DEMO_NC_PATH.
    DEMO_NC_PATH: str = os.path.join(os.getcwd(), "tests/data/RMA1_0315_01_20250819T001715Z.nc")

    # Auth & database
    JWT_SECRET: str = "CHANGE-ME-IN-PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    # Token expiration time (hours)
    JWT_EXPIRE_HOURS: int = 2
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin"
    DB_DIR: str = os.path.join(os.getcwd(), "app/storage/db")
    LOG_DIR: str = os.path.join(os.getcwd(), "app/storage/logs")

    class Config:
        env_file = ".env"

settings = Settings()
