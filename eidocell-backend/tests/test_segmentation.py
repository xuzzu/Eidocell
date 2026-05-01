import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _create_test_image(path: Path, size=(100, 100)):
    """Create a test image with a bright circle on dark background for segmentation."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    # Draw a white circle in the center
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [200, 200, 200]
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_for_segmentation(client, tmp_path):
    """Create a session with synthetic images that have clear objects to segment."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Segmentation Test",
        "images_directory": str(img_dir),
    }).json()
    return session


# ── Methods listing ─────────────────────────────────────────────────────


def test_list_methods(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.get(f"/sessions/{sid}/segmentation/methods")
    assert resp.status_code == 200
    methods = resp.json()
    assert len(methods) == 4
    method_ids = [m["id"] for m in methods]
    assert "otsu_intensity" in method_ids
    assert "otsu_edges" in method_ids
    assert "adaptive" in method_ids
    assert "watershed" in method_ids
    # Each method should have params
    for m in methods:
        assert len(m["params"]) >= 2


# ── Run segmentation ───────────────────────────────────────────────────


def test_run_segmentation_otsu_intensity(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 50, "min_component_size": 10},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert data["processed"] == 5
    assert data["failed"] == 0


def test_run_segmentation_otsu_edges(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_edges",
        "params": {"distance_from_center": 50, "min_component_size": 10, "close_radius": 5},
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] == 5


def test_run_segmentation_adaptive(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "adaptive",
        "params": {"block_size": 35, "c_value": 2},
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] == 5


def test_run_segmentation_watershed(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "watershed",
        "params": {"foreground_threshold": 70, "morph_kernel_size": 3},
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] == 5


def test_run_segmentation_default_params(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] == 5


def test_run_segmentation_unknown_method(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "nonexistent",
    })
    assert resp.status_code == 400


# ── Mask attributes ────────────────────────────────────────────────────


def test_get_mask_attributes(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    # Run segmentation first
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    # Get samples
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_id = samples[0]["id"]

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/mask/attributes")
    assert resp.status_code == 200
    attrs = resp.json()
    # Should have key morphological attributes
    assert "area" in attrs
    assert "perimeter" in attrs
    assert "solidity" in attrs
    assert "mean_intensity" in attrs
    assert "form_factor" in attrs
    assert "hu1" in attrs
    # Values should be reasonable for a circle
    assert attrs["area"] > 0
    assert attrs["perimeter"] > 0
    assert 0 < attrs["solidity"] <= 1
    assert 0 < attrs["form_factor"] <= 1


def test_get_mask_attributes_no_mask(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    resp = client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes")
    assert resp.status_code == 404


# ── Mask overlay ───────────────────────────────────────────────────────


def test_get_mask_overlay(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_id = samples[0]["id"]

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/mask/overlay")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 0


def test_get_mask_overlay_no_mask(client, session_for_segmentation):
    sid = session_for_segmentation["id"]
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    resp = client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/overlay")
    assert resp.status_code == 404


# ── Re-segmentation overwrites ─────────────────────────────────────────


def test_resegmentation_overwrites_masks(client, session_for_segmentation):
    sid = session_for_segmentation["id"]

    # First run
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    attrs1 = client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes").json()

    # Second run with different method
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "adaptive",
        "params": {"block_size": 35, "c_value": 2},
    })
    attrs2 = client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes").json()

    # Attributes should differ (different segmentation methods)
    assert attrs1["area"] != attrs2["area"]


# ── Gallery reflects mask status ────────────────────────────────────────


def test_gallery_has_mask_after_segmentation(client, session_for_segmentation):
    sid = session_for_segmentation["id"]

    # Before segmentation
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    assert all(not s["has_mask"] for s in samples)

    # After segmentation
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    assert all(s["has_mask"] for s in samples)


# ── Real images test ───────────────────────────────────────────────────


def test_real_images_segmentation(client, real_images_dir):
    """Test segmentation with real blood test images if available."""
    session = client.post("/sessions/", json={
        "name": "Blood Segmentation", "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 50, "min_component_size": 15},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] > 0

    # Check attributes on first sample
    samples = client.post(f"/sessions/{sid}/samples/list", json={"limit": 1}).json()["items"]
    attrs = client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes").json()
    assert "area" in attrs
    assert attrs["area"] > 0
