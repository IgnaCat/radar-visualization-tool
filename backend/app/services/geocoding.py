"""
Reverse geocoding via Nominatim (OpenStreetMap).
Resolves lat/lon to a human-readable address.
Free, no API key, 1 req/sec rate limit.
"""
import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "RadarApp/1.0 (thesis project, FAMAF UNC)"
TIMEOUT_SECONDS = 5.0
# DNS cold-start in Docker: retry ConnectError up to this many times
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0


async def reverse_geocode(lat: float, lon: float) -> Optional[dict]:
    """
    Call Nominatim reverse API to resolve coordinates to an address.

    Retries on ConnectError (e.g. DNS cold-start in Docker) up to MAX_RETRIES times.
    Returns {"address": str, "city": str | None, "country": str | None}
    or None on any failure.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 2):  # attempts: 1, 2, 3
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                response = await client.get(
                    NOMINATIM_URL, params=params, headers=headers
                )

            if response.status_code != 200:
                logger.warning(
                    "Nominatim returned %d for (%s, %s)", response.status_code, lat, lon
                )
                return None

            data = response.json()
            addr = data.get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village")

            return {
                "address": data.get("display_name"),
                "city": city,
                "country": addr.get("country"),
            }

        except httpx.ConnectError as e:
            # DNS or TCP connection failure — worth retrying (Docker DNS cold-start)
            if attempt <= MAX_RETRIES:
                logger.warning(
                    "Nominatim connect error for (%s, %s), retrying (%d/%d): %s",
                    lat, lon, attempt, MAX_RETRIES, e,
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.warning("Nominatim connect error for (%s, %s) after %d attempts: %s", lat, lon, attempt, e)
            return None

        except httpx.TimeoutException:
            logger.warning("Nominatim timeout for (%s, %s)", lat, lon)
            return None

        except Exception as e:
            logger.warning("Nominatim error for (%s, %s): %s", lat, lon, e)
            return None

    return None  # unreachable, but satisfies type checker
