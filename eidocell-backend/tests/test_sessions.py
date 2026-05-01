import numpy as np
import pytest
from PIL import Image


def _create_test_image(path, size=(100, 100), brightness=200):
    """Create image with a circle for segmentation."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [brightness, brightness, brightness]
    Image.fromarray(img).save(path)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_sessions_empty(client):
    resp = client.get("/sessions/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_session(client, sample_images_dir):
    resp = client.post("/sessions/", json={
        "name": "Blood Test",
        "images_directory": str(sample_images_dir),
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Blood Test"
    assert data["sample_count"] == 5
    assert data["scale_factor"] == 1.0
    assert data["scale_units"] == "px"
    assert data["id"]
    assert data["session_folder"]


def test_create_session_no_images(client, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    resp = client.post("/sessions/", json={
        "name": "Empty",
        "images_directory": str(empty_dir),
    })
    assert resp.status_code == 400


def test_create_session_bad_directory(client):
    resp = client.post("/sessions/", json={
        "name": "Bad",
        "images_directory": "/nonexistent/path",
    })
    assert resp.status_code == 400


def test_get_session(client, sample_images_dir):
    create_resp = client.post("/sessions/", json={
        "name": "Test Session",
        "images_directory": str(sample_images_dir),
    })
    session_id = create_resp.json()["id"]

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Session"
    assert data["sample_count"] == 5


def test_get_session_not_found(client):
    resp = client.get("/sessions/nonexistent")
    assert resp.status_code == 404


def test_update_session(client, sample_images_dir):
    create_resp = client.post("/sessions/", json={
        "name": "Original Name",
        "images_directory": str(sample_images_dir),
    })
    session_id = create_resp.json()["id"]

    resp = client.patch(f"/sessions/{session_id}", json={
        "name": "Renamed Session",
        "scale_factor": 0.5,
        "scale_units": "um",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Renamed Session"
    assert data["scale_factor"] == 0.5
    assert data["scale_units"] == "um"


def test_delete_session(client, sample_images_dir):
    create_resp = client.post("/sessions/", json={
        "name": "To Delete",
        "images_directory": str(sample_images_dir),
    })
    session_id = create_resp.json()["id"]

    resp = client.delete(f"/sessions/{session_id}")
    assert resp.status_code == 204

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 404


def test_list_sessions_after_create(client, sample_images_dir):
    client.post("/sessions/", json={
        "name": "Session A",
        "images_directory": str(sample_images_dir),
    })
    client.post("/sessions/", json={
        "name": "Session B",
        "images_directory": str(sample_images_dir),
    })

    resp = client.get("/sessions/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Most recently opened first
    names = [s["name"] for s in data]
    assert "Session A" in names
    assert "Session B" in names


def test_create_session_with_real_images(client, real_images_dir):
    """Test with the actual blood test images if available."""
    resp = client.post("/sessions/", json={
        "name": "Blood Elongated",
        "images_directory": real_images_dir,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["sample_count"] > 0
    assert data["name"] == "Blood Elongated"


# ── Cascading delete ──────────────────────────────────────────────────


def test_delete_session_cascades_samples_and_masks(client, tmp_path):
    """Deleting a session removes all its samples, masks, and classes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Cascade Test", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Run segmentation to create masks + attributes
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    # Create a class and assign samples
    cls = client.post(f"/sessions/{sid}/classes", json={
        "name": "TypeA", "color": "#FF0000",
    }).json()
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples[:2]],
        "class_id": cls["id"],
    })

    # Verify data exists
    assert len(client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]) == 5
    assert client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes").status_code == 200

    # Delete the session
    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 204

    # Session is gone
    assert client.get(f"/sessions/{sid}").status_code == 404

    # Samples/masks/classes are all unreachable
    assert client.post(f"/sessions/{sid}/samples/list", json={}).status_code == 404
    assert client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes").status_code == 404


def test_delete_session_cascades_clusters_and_plots(client, tmp_path):
    """Deleting a session removes clusters, plots, and gates."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        _create_test_image(img_dir / f"cell_{i:03d}.png", brightness=50 + i * 20)

    session = client.post("/sessions/", json={
        "name": "Cascade Full", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Build up data: segment → extract → cluster → plot → gate
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })
    client.post(f"/sessions/{sid}/features/extract", json={"method": "morphological"})
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()
    client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 99999},
        "parameters": ["area"],
    })

    # Verify everything exists
    assert len(client.get(f"/sessions/{sid}/clusters/").json()) > 0
    assert len(client.get(f"/sessions/{sid}/analysis/plots").json()) == 1
    assert len(client.get(f"/sessions/{sid}/analysis/gates").json()) == 1

    # Delete
    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


# ── Error handling ────────────────────────────────────────────────────


def test_create_session_corrupted_images(client, tmp_path):
    """Session creation with corrupt image files should still succeed
    (corrupt files are skipped during ingestion)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # One valid image
    _create_test_image(img_dir / "good.png")
    # Corrupt files (invalid image data)
    (img_dir / "bad1.png").write_bytes(b"not a real png")
    (img_dir / "bad2.jpg").write_bytes(b"\xff\xd8\xff garbage")

    resp = client.post("/sessions/", json={
        "name": "Mixed", "images_directory": str(img_dir),
    })
    # Should succeed — at least the valid image is ingested
    assert resp.status_code == 201
    assert resp.json()["sample_count"] >= 1
