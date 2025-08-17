from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "Radar Visualization"
    FRONTEND_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    COG_DIR: str = os.path.join(os.getcwd(), "app/storage/cogs")
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "app/storage/uploads")

    # Reglas de upload
    ALLOWED_EXTENSIONS: List[str] = [".nc"]
    MAX_UPLOAD_MB: int = 500

    class Config:
        env_file = ".env"

settings = Settings()
