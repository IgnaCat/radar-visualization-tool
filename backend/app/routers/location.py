"""
Router for receiving browser geolocation data.
Updates the user's AccessLog with precise coordinates and reverse-geocoded address.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..models.location import LocationRequest
from ..models.db import AccessLog, UserSession
from ..services.geocoding import reverse_geocode
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["location"])


@router.post("/location")
async def receive_location(
    body: LocationRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Receive browser geolocation, reverse-geocode it, and update the AccessLog.
    Called once after the user grants location permission in the browser.
    """
    # Find the active session
    session = (
        db.query(UserSession)
        .filter(
            UserSession.session_id == body.session_id,
            UserSession.user_id == current_user.id,
            UserSession.is_active == True,
        )
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the most recent AccessLog for this user
    access_log = (
        db.query(AccessLog)
        .filter(AccessLog.user_id == current_user.id)
        .order_by(AccessLog.logged_in_at.desc())
        .first()
    )
    if not access_log:
        raise HTTPException(status_code=404, detail="No access log found")

    # Reverse geocode (async, best-effort)
    geo = await reverse_geocode(body.latitude, body.longitude)

    # Update the access log
    access_log.latitude = body.latitude
    access_log.longitude = body.longitude
    access_log.location_source = "browser"

    if geo:
        access_log.address = geo["address"]
        access_log.city = geo["city"]
        access_log.country = geo["country"]

    try:
        db.flush()
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Failed to persist location for user %d: %s", current_user.id, exc)
        raise HTTPException(status_code=500, detail="Could not save location")

    logger.info(
        "Location saved for user %d: lat=%.4f lon=%.4f source=browser address=%s",
        current_user.id,
        body.latitude,
        body.longitude,
        geo["address"] if geo else "(geocoding failed)",
    )

    return {"status": "ok"}
