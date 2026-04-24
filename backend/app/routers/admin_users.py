"""Admin endpoints: user CRUD, access logs, active sessions."""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import hash_password
from ..dependencies.auth import require_admin
from ..models.db.user import User, UserRole
from ..models.db.access_log import AccessLog
from ..models.db.session import UserSession
from ..models.auth import UserCreate, UserUpdate, UserOut, AccessLogOut, ActiveSessionOut

router = APIRouter(prefix="/admin", tags=["admin-users"])
logger = logging.getLogger(__name__)


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        UserOut(
            id=u.id,
            username=u.username,
            role=u.role.value,
            is_active=u.is_active,
            created_at=u.created_at,
            updated_at=u.updated_at,
        )
        for u in users
    ]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    existing = db.query(User).filter(User.username == body.username).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username ya existe")

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        role=UserRole(body.role),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Usuario creado: %s (role=%s)", user.username, user.role.value)
    return UserOut(
        id=user.id,
        username=user.username,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    body: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Cannot deactivate yourself
    if body.is_active is False and user.id == admin.id:
        raise HTTPException(status_code=400, detail="No podés desactivarte a vos mismo")

    if body.is_active is not None:
        user.is_active = body.is_active
    if body.role is not None:
        user.role = UserRole(body.role)
    if body.password is not None:
        user.hashed_password = hash_password(body.password)

    db.commit()
    db.refresh(user)
    logger.info("Usuario actualizado: %s", user.username)
    return UserOut(
        id=user.id,
        username=user.username,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.get("/access-logs", response_model=list[AccessLogOut])
def get_access_logs(
    user_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    query = (
        db.query(AccessLog, User.username)
        .join(User, AccessLog.user_id == User.id)
    )
    if user_id is not None:
        query = query.filter(AccessLog.user_id == user_id)
    rows = query.order_by(AccessLog.logged_in_at.desc()).limit(limit).all()
    return [
        AccessLogOut(
            id=log.id,
            user_id=log.user_id,
            username=username,
            ip_address=log.ip_address,
            city=log.city,
            country=log.country,
            user_agent=log.user_agent,
            logged_in_at=log.logged_in_at,
            address=log.address,
            location_source=log.location_source,
        )
        for log, username in rows
    ]


@router.get("/sessions/active", response_model=list[ActiveSessionOut])
def get_active_sessions(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    rows = (
        db.query(UserSession, User.username)
        .join(User, UserSession.user_id == User.id)
        .filter(UserSession.is_active == True)  # noqa: E712
        .order_by(UserSession.created_at.desc())
        .all()
    )
    return [
        ActiveSessionOut(
            id=s.id,
            user_id=s.user_id,
            username=username,
            session_id=s.session_id,
            created_at=s.created_at,
        )
        for s, username in rows
    ]


@router.post("/cleanup/{target_user_id}")
def force_cleanup_user(
    target_user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Limpieza forzada de todos los recursos de un usuario (acción de admin).
    Borra TODAS sus sesiones activas y sus archivos.

    Estructura de directorios:
      uploads → UPLOAD_DIR/{user_id}/{session_id}/   (un dir por sesión)
      COGs    → IMAGES_DIR/{session_id}/             (keyed por session_id del browser)
    """
    from pathlib import Path
    import shutil
    from ..core.config import settings

    user = db.query(User).filter(User.id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    deleted = {"uploads": 0, "tmp": 0, "sessions": 0}

    # 1. Borrar uploads: UPLOAD_DIR/{user_id}/ contiene todos los {session_id}/ del usuario
    upload_user_dir = Path(settings.UPLOAD_DIR) / str(target_user_id)
    if upload_user_dir.exists():
        shutil.rmtree(upload_user_dir, ignore_errors=True)
        deleted["uploads"] = 1

    # 2. Borrar COGs por session_id (viven en IMAGES_DIR/{session_id}/, no en user_id/)
    #    Obtenemos los session_ids de este usuario desde la DB.
    user_sessions = (
        db.query(UserSession)
        .filter(UserSession.user_id == target_user_id)
        .all()
    )
    images_base = Path(settings.IMAGES_DIR)
    for us in user_sessions:
        session_dir = images_base / us.session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            deleted["tmp"] += 1

    # 3. Marcar todas las sesiones del usuario como inactivas
    count = (
        db.query(UserSession)
        .filter(UserSession.user_id == target_user_id, UserSession.is_active == True)  # noqa: E712
        .update({"is_active": False})
    )
    deleted["sessions"] = count
    db.commit()

    return {"detail": f"Recursos limpiados para {user.username}", "deleted": deleted}
