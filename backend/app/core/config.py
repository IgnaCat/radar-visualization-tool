from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "Radar Visualization"
    BASE_URL: str = "http://localhost:8000"
    FRONTEND_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
<<<<<<< HEAD
    IMAGES_DIR: str = "app/storage/tmp"
=======

    # Storage directories (set via env vars in Docker, defaults for local)
    IMAGES_DIR: str = "/app/storage/tmp"
>>>>>>> 14ecc66fede379e1713e30b02a21c905ba0baad7
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "app/storage/uploads")
    DATA_DIR: str = os.path.join(os.getcwd(), "app/storage/data")
    ALLOWED_PRODUCTS: List[str] = ["PPI", "RHI", "CAPPI", "COLMAX"]

    # Reglas de upload
    ALLOWED_EXTENSIONS: List[str] = [".nc"]
    MAX_UPLOAD_MB: int = 500

    class Config:
        env_file = ".env"

settings = Settings()
