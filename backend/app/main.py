from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# from titiler.application import TiTilerFastAPI

from .core.config import settings
from .routers import process, upload

app = FastAPI(title=settings.APP_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = Path("app/storage/tmp")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Todo lo que guardes en app/storage/tmp será accesible en /static/tmp/...
app.mount("/static/tmp", StaticFiles(directory=IMAGES_DIR), name="tmp")

# COGs estáticos (para construir la URL pública del .tif)
# cog_dir = Path(settings.COG_DIR); cog_dir.mkdir(parents=True, exist_ok=True)
# app.mount("/static/cogs", StaticFiles(directory=cog_dir), name="cogs")

# app.mount("/cog", TiTilerFastAPI())

app.include_router(upload.router)
app.include_router(process.router)

@app.get("/health")
def health():
    return {"status": "ok"}
