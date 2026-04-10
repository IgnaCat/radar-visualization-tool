"""Authentication endpoints: login and logout."""
import logging
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import verify_password, create_access_token
from ..dependencies.auth import get_current_user
from ..models.db.user import User
from ..models.db.access_log import AccessLog
from ..models.db.session import UserSession
from ..models.auth import LoginRequest, LoginResponse, UserOut
from ..services.geo_ip import lookup_ip

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For from Traefik."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    # Find user
    user = db.query(User).filter(User.username == body.username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
        )

    # Check password
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
        )

    # Check active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado",
        )

    # Create JWT
    token = create_access_token({
        "user_id": user.id,
        "username": user.username,
        "role": user.role.value,
    })

    # Register session (if session_id provided)
    if body.session_id:
        existing_session = db.query(UserSession).filter(
            UserSession.session_id == body.session_id
        ).first()

        if existing_session:
            existing_session.user_id = user.id
            existing_session.is_active = True
        else:
            db.add(UserSession(user_id=user.id, session_id=body.session_id, is_active=True))

    # Log access
    client_ip = _get_client_ip(request)
    geo = lookup_ip(client_ip)
    db.add(AccessLog(
        user_id=user.id,
        ip_address=client_ip,
        city=geo["city"],
        country=geo["country"],
        user_agent=request.headers.get("User-Agent", "")[:500],
    ))
    db.commit()

    logger.info(
        "Login exitoso: %s desde %s (%s, %s)",
        user.username, client_ip, geo["city"], geo["country"],
    )

    return LoginResponse(
        access_token=token,
        user=UserOut(
            id=user.id,
            username=user.username,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        ),
    )


class LogoutRequest(BaseModel):
    session_id: str | None = None


@router.post("/logout")
def logout(
    body: LogoutRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Deactivate the session. Frontend should also call /cleanup/close."""
    if body.session_id:
        db.query(UserSession).filter(
            UserSession.session_id == body.session_id,
            UserSession.user_id == current_user.id,
        ).update({"is_active": False})
        db.commit()

    logger.info("Logout: %s", current_user.username)
    return {"detail": "Sesión cerrada"}
