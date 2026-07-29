"""
GeoIP lookup using MaxMind GeoLite2 database.
Returns city/country for an IP address. Fails silently if DB is missing.

Setup: Download GeoLite2-City.mmdb from MaxMind (free account required)
and place it in backend/app/storage/geolite2/GeoLite2-City.mmdb
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_reader = None
_initialized = False


def _get_reader():
    global _reader, _initialized
    if _initialized:
        return _reader
    _initialized = True

    try:
        import geoip2.database

        candidates = [
            Path(__file__).parent.parent / "storage" / "geolite2" / "GeoLite2-City.mmdb",
            Path("/app/app/storage/geolite2/GeoLite2-City.mmdb"),  # Docker path
        ]
        for path in candidates:
            if path.exists():
                _reader = geoip2.database.Reader(str(path))
                logger.info("GeoIP database loaded from %s", path)
                return _reader

        logger.warning("GeoLite2-City.mmdb not found. GeoIP lookup disabled.")
    except ImportError:
        logger.warning("geoip2 package not installed. GeoIP lookup disabled.")
    except Exception as e:
        logger.warning("Failed to load GeoIP database: %s", e)

    return None


def lookup_ip(ip_address: str) -> dict[str, Optional[str]]:
    """
    Resolve IP to city/country. Returns {"city": ..., "country": ...}.
    Returns nulls if lookup fails or DB is unavailable.
    """
    result: dict[str, Optional[str]] = {"city": None, "country": None}

    # Skip private/local IPs
    if not ip_address or ip_address in ("127.0.0.1", "::1", "localhost"):
        return result
    if ip_address.startswith(("192.168.", "10.", "172.16.", "172.17.")):
        return result

    reader = _get_reader()
    if reader is None:
        return result

    try:
        response = reader.city(ip_address)
        result["city"] = response.city.name
        result["country"] = response.country.name
    except Exception:
        pass  # Unknown IP or lookup error — return nulls

    return result
