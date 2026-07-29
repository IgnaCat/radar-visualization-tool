"""
stale_cleanup.py
────────────────
Removes uploaded raw files (NetCDF/BUFR) and generated COG files left over
from previous server runs.  Called once at application startup.

W-operator matrices (Barnes cache) stored in CACHE_DIR are intentionally
preserved — they are expensive to recompute and are not user-specific.
"""

import logging
import shutil
from pathlib import Path

from ..core.config import settings

logger = logging.getLogger(__name__)


def cleanup_stale_files() -> None:
    """Delete stale uploads and generated images from previous server sessions."""
    upload_dir = Path(settings.UPLOAD_DIR)
    images_dir = Path(settings.IMAGES_DIR)

    _purge_directory(upload_dir, label="uploads")
    _purge_directory(images_dir, label="generated images")


def _purge_directory(base: Path, label: str) -> None:
    """Remove all direct children (files and subdirectories) under *base*.

    The *base* directory itself is kept so the server can still write to it.
    If *base* does not exist yet it is created and nothing is deleted.
    """
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return

    removed_count = 0
    errors = 0
    for child in base.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed_count += 1
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not remove %s: %s", child, exc)
            errors += 1

    if removed_count or errors:
        logger.info(
            "Stale cleanup [%s]: removed %d item(s)%s.",
            label,
            removed_count,
            f", {errors} error(s)" if errors else "",
        )
    else:
        logger.info("Stale cleanup [%s]: nothing to remove.", label)
