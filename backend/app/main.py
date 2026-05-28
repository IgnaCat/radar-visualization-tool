import asyncio
import faulthandler
import os
import sys
import logging
import importlib.util
from pathlib import Path

# Dump C-level stack trace to stderr on SIGSEGV/SIGFPE/SIGABRT.
# This survives crashes in native extensions (GDAL, HDF5, libdecbufr, etc.)
# and prints *before* the process dies, so Docker logs capture it.
faulthandler.enable()

# Fix PROJ database version conflict: osgeo ships an older proj.db (minor=3)
# but rasterio/pyproj expect minor>=4. Must be set before osgeo/rasterio import
# so that PROJ finds the correct data dir when it first initialises.
for _pkg, _rel in [("pyproj", "proj_dir/share/proj"), ("rasterio", "proj_data")]:
    _spec = importlib.util.find_spec(_pkg)
    if _spec and _spec.submodule_search_locations:
        _proj_data = Path(next(iter(_spec.submodule_search_locations))) / _rel
        if (_proj_data / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_proj_data)
            os.environ["PROJ_LIB"] = str(_proj_data)
            break

from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from titiler.core.factory import TilerFactory
from fastapi.responses import HTMLResponse

from .core.config import settings
from .core.middleware import CustomAccessLogMiddleware
from .routers import process, upload, cleanup, pseudo_rhi, radar_stats, radar_pixel, elevation_profile, colormap, admin, auth, admin_users, location

# Forzar zona horaria Argentina (UTC-3) para todos los logs
logging.Formatter.converter = lambda *args: datetime.now(timezone(timedelta(hours=-3))).timetuple()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,       # stdout is captured by Docker logs
    force=True,              # override any prior basicConfig
)

# Reduce noise from chatty libraries
logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)

# Disable Uvicorn default access log to use our custom middleware instead
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.disabled = True

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

# GDAL/Rasterio optimizations for COG tile serving
# NOTE: these are fallback defaults for running without Docker.
# In Docker, environment variables from docker-compose.yml take precedence
# (setdefault won't overwrite existing env vars).
# Keep values in sync with docker-compose.yml.
os.environ.setdefault("GDAL_CACHEMAX", "512")           # 512 MB raster block cache
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")   # parallel TIFF decompression
os.environ.setdefault("VSI_CACHE", "TRUE")               # VSI file block caching
os.environ.setdefault("VSI_CACHE_SIZE", "262144000")     # 250 MB VSI cache
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")  # skip dir listing on open
os.environ.setdefault("GDAL_MAX_DATASET_POOL_SIZE", "450")  # max open datasets
os.environ.setdefault("GDAL_FORCE_CACHING", "NO")       # NO for local files (YES only for S3/HTTP)
os.environ.setdefault("PROJ_NETWORK", "OFF")             # no network CRS lookups

app.add_middleware(CustomAccessLogMiddleware)
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
app.include_router(auth.router)
app.include_router(admin_users.router)
app.include_router(location.router)


@app.on_event("startup")
async def on_startup():
    """Initialize database, seed admin user, clean stale files, and start background tasks."""
    from .core.database import init_db
    from .core.migrations import run_migrations
    from .services.seed import seed_admin
    from .services.stale_cleanup import cleanup_stale_files
    from .services.inactivity_cleanup import inactivity_cleanup_loop
    init_db()
    run_migrations()
    seed_admin()
    cleanup_stale_files()
    # Background task: libera RAM de sesiones abandonadas (tab cerrada sin logout)
    asyncio.create_task(inactivity_cleanup_loop())


@app.get("/health")
def health():
    return {"status": "ok"}
