"""Tests for /auth endpoints using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.security import hash_password
from app.models.db.user import User, UserRole

# Isolated in-memory DB — never touches the real radar.db
# StaticPool ensures all connections share the same SQLite in-memory database,
# so tables created by setup_db are visible to sessions opened during requests.
_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_test_engine, autocommit=False, autoflush=False)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    """Create fresh tables for each test, then tear them down."""
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def db():
    session = _TestSession()
    yield session
    session.close()


@pytest.fixture
def admin_user(db):
    user = User(
        username="admin",
        hashed_password=hash_password("adminpass"),
        role=UserRole.admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def client():
    from app.main import app
    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def test_login_success(client, admin_user):
    resp = client.post("/auth/login", json={
        "username": "admin",
        "password": "adminpass",
        "session_id": "session-123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_login_wrong_password(client, admin_user):
    resp = client.post("/auth/login", json={
        "username": "admin",
        "password": "wrongpass",
    })
    assert resp.status_code == 401


def test_login_nonexistent_user(client):
    resp = client.post("/auth/login", json={
        "username": "ghost",
        "password": "anything",
    })
    assert resp.status_code == 401


def test_login_inactive_user(client, db):
    user = User(
        username="disabled",
        hashed_password=hash_password("pass"),
        role=UserRole.user,
        is_active=False,
    )
    db.add(user)
    db.commit()
    resp = client.post("/auth/login", json={
        "username": "disabled",
        "password": "pass",
    })
    assert resp.status_code == 403


def test_logout_clears_session(client, admin_user):
    # Login first
    login_resp = client.post("/auth/login", json={
        "username": "admin",
        "password": "adminpass",
        "session_id": "session-abc",
    })
    token = login_resp.json()["access_token"]

    # Logout
    resp = client.post(
        "/auth/logout",
        json={"session_id": "session-abc"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
