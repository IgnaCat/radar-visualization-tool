"""Tests for /admin/* user management endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.core.database import Base, engine, SessionLocal, init_db
from app.core.security import hash_password, create_access_token
from app.models.db.user import User, UserRole


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def admin_user(db):
    # seed_admin() ya crea un 'admin' durante el startup del app,
    # así que intentamos recuperarlo primero.
    existing = db.query(User).filter(User.username == "admin").first()
    if existing:
        return existing
    user = User(
        username="admin",
        hashed_password=hash_password("pass"),
        role=UserRole.admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    return create_access_token({"user_id": admin_user.id, "username": "admin", "role": "admin"})


@pytest.fixture
def user_token(db):
    user = User(
        username="regular",
        hashed_password=hash_password("pass"),
        role=UserRole.user,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return create_access_token({"user_id": user.id, "username": "regular", "role": "user"})


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_list_users_as_admin(client, admin_token, admin_user):
    resp = client.get("/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    users = resp.json()
    assert len(users) >= 1
    assert users[0]["username"] == "admin"


def test_list_users_as_regular_user_forbidden(client, user_token):
    resp = client.get("/admin/users", headers={"Authorization": f"Bearer {user_token}"})
    assert resp.status_code == 403


def test_create_user(client, admin_token):
    resp = client.post(
        "/admin/users",
        json={"username": "newuser", "password": "secretpass", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 201
    assert resp.json()["username"] == "newuser"


def test_create_user_duplicate_username(client, admin_token, admin_user):
    resp = client.post(
        "/admin/users",
        json={"username": "admin", "password": "pass", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 409


def test_update_user_deactivate(client, admin_token, db):
    user = User(
        username="todeactivate",
        hashed_password=hash_password("p"),
        role=UserRole.user,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    resp = client.patch(
        f"/admin/users/{user.id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False


def test_admin_cannot_deactivate_self(client, admin_token, admin_user):
    resp = client.patch(
        f"/admin/users/{admin_user.id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 400


def test_unauthenticated_request_returns_401(client):
    resp = client.get("/admin/users")
    assert resp.status_code == 401
