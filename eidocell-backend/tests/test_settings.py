import json
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    """Redirect settings file to tmp_path for test isolation."""
    settings_path = tmp_path / "settings.json"
    with patch("services.settings_service.SETTINGS_PATH", settings_path):
        yield settings_path


# ── Get defaults ───────────────────────────────────────────────────────


def test_get_default_settings(client):
    resp = client.get("/settings/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["theme"] == "dark"
    assert data["thumbnail_quality"] == 75
    assert data["images_per_collage"] == 25
    assert data["default_segmentation_method"] == "otsu_intensity"
    assert data["default_feature_method"] == "morphological"
    assert data["default_dim_reduction_method"] == "pca"


# ── Update settings ───────────────────────────────────────────────────


def test_update_theme(client):
    resp = client.patch("/settings/", json={"theme": "light"})
    assert resp.status_code == 200
    assert resp.json()["theme"] == "light"

    # Verify persistence
    resp = client.get("/settings/")
    assert resp.json()["theme"] == "light"


def test_update_thumbnail_quality(client):
    resp = client.patch("/settings/", json={"thumbnail_quality": 90})
    assert resp.status_code == 200
    assert resp.json()["thumbnail_quality"] == 90


def test_update_multiple_fields(client):
    resp = client.patch("/settings/", json={
        "theme": "light",
        "images_per_collage": 16,
        "default_segmentation_method": "adaptive",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["theme"] == "light"
    assert data["images_per_collage"] == 16
    assert data["default_segmentation_method"] == "adaptive"
    # Unchanged fields should keep defaults
    assert data["thumbnail_quality"] == 75


def test_partial_update_preserves_other_fields(client):
    client.patch("/settings/", json={"theme": "light"})
    client.patch("/settings/", json={"thumbnail_quality": 50})

    resp = client.get("/settings/")
    data = resp.json()
    assert data["theme"] == "light"
    assert data["thumbnail_quality"] == 50


# ── Reset settings ────────────────────────────────────────────────────


def test_reset_settings(client):
    client.patch("/settings/", json={"theme": "light", "thumbnail_quality": 10})
    resp = client.post("/settings/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["theme"] == "dark"
    assert data["thumbnail_quality"] == 75


# ── Settings persistence ─────────────────────────────────────────────


def test_settings_persisted_to_file(client, isolated_settings):
    client.patch("/settings/", json={"theme": "light"})
    assert isolated_settings.exists()
    saved = json.loads(isolated_settings.read_text())
    assert saved["theme"] == "light"


# ── Validation ────────────────────────────────────────────────────────


def test_invalid_thumbnail_quality(client):
    resp = client.patch("/settings/", json={"thumbnail_quality": 0})
    assert resp.status_code == 422

    resp = client.patch("/settings/", json={"thumbnail_quality": 101})
    assert resp.status_code == 422


def test_invalid_images_per_collage(client):
    resp = client.patch("/settings/", json={"images_per_collage": 0})
    assert resp.status_code == 422

    resp = client.patch("/settings/", json={"images_per_collage": 51})
    assert resp.status_code == 422
