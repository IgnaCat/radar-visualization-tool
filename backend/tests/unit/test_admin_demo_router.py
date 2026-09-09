"""Tests for POST /admin/demo/load."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.security import hash_password, create_access_token
from app.core.config import settings
from app.models.db.user import User, UserRole

_engine = create_engine(
    "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
_Session = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def _override_get_db():
    db = _Session()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture
def db():
    s = _Session()
    yield s
    s.close()


@pytest.fixture
def admin(db):
    u = User(username="admin", hashed_password=hash_password("pass"), role=UserRole.admin)
    db.add(u); db.commit(); db.refresh(u)
    return u


@pytest.fixture
def admin_token(admin):
    return create_access_token({"user_id": admin.id, "username": "admin", "role": "admin"})


@pytest.fixture
def user_token(db):
    u = User(username="regular", hashed_password=hash_password("pass"), role=UserRole.user)
    db.add(u); db.commit(); db.refresh(u)
    return create_access_token({"user_id": u.id, "username": "regular", "role": "user"})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    from app.main import app
    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def test_load_demo_as_admin(client, admin_token, admin):
    resp = client.post(
        "/admin/demo/load", json={"session_id": "sess1"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert len(data["files"]) == 1
    entry = data["files"][0]
    assert entry["filepath"] == "RMA1_0315_01_20250819T001715Z.nc"
    assert entry["size_bytes"] > 0
    assert "fields_present" in entry["metadata"]
    assert "RMA1" in data["radars"]
    assert "01" in data["volumes"]
    dest = Path(settings.UPLOAD_DIR) / str(admin.id) / "sess1" / "RMA1_0315_01_20250819T001715Z.nc"
    assert dest.exists()


def test_load_demo_forbidden_for_regular_user(client, user_token):
    resp = client.post(
        "/admin/demo/load", json={"session_id": "s"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403


def test_load_demo_missing_file_returns_404(client, admin_token, monkeypatch):
    monkeypatch.setattr(settings, "DEMO_NC_PATH", "/nonexistent/demo.nc")
    resp = client.post(
        "/admin/demo/load", json={"session_id": "s"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 404


def test_load_demo_idempotent(client, admin_token):
    h = {"Authorization": f"Bearer {admin_token}"}
    first = client.post("/admin/demo/load", json={"session_id": "sess2"}, headers=h)
    assert first.status_code == 201
    assert first.json()["warnings"] == []
    second = client.post("/admin/demo/load", json={"session_id": "sess2"}, headers=h)
    assert second.status_code == 201
    assert any("ya existe" in w for w in second.json()["warnings"])
