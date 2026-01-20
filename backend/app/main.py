import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from titiler.core.factory import TilerFactory
from fastapi.responses import HTMLResponse

from .core.config import settings
from .routers import process, upload, cleanup, pseudo_rhi, radar_stats, radar_pixel, elevation_profile, colormap, admin

app = FastAPI(title=settings.APP_NAME)

# GDAL/Rasterio optimizations for COG tile serving
os.environ.setdefault("GDAL_CACHEMAX", "512")  # 512 MB cache for better tile performance
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "262144000")  # 250 MB
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_MAX_DATASET_POOL_SIZE", "450")
os.environ.setdefault("GDAL_FORCE_CACHING", "YES")
os.environ.setdefault("PROJ_NETWORK", "OFF")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for COG and processed images
images_dir = Path(settings.IMAGES_DIR)
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/tmp", StaticFiles(directory=images_dir), name="tmp")

# TiTiler factory for COG tile serving
cog = TilerFactory(
    router_prefix="/cog",
    add_preview=True,
    add_part=True,
    add_viewer=False,
)

# Include routers
app.include_router(cog.router, prefix="/cog", tags=["cog"])
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(cleanup.router)
app.include_router(pseudo_rhi.router)
app.include_router(radar_stats.router)
app.include_router(radar_pixel.router)
app.include_router(elevation_profile.router)
app.include_router(colormap.router)
app.include_router(admin.router)

@app.get("/health")
def health():
    return {"status": "ok"}
