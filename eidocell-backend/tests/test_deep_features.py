"""Tests for deep feature extraction (MobileNetV3)."""

import time

import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _wait_for_task(client, task_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed", "cancelled"):
            return data
        time.sleep(0.1)
    raise AssertionError(f"task {task_id} did not finish in {timeout_s}s")


def _run_extract(client, sid: str, body: dict) -> dict:
    resp = client.post(f"/sessions/{sid}/features/extract", json=body)
    assert resp.status_code == 200, resp.text
    task = _wait_for_task(client, resp.json()["task_id"])
    assert task["status"] == "completed", task
    return task["result"]


def _run_dim_reduction(client, sid: str, body: dict) -> dict:
    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json=body)
    assert resp.status_code == 200, resp.text
    task = _wait_for_task(client, resp.json()["task_id"])
    assert task["status"] == "completed", task
    return task["result"]


def _create_test_image(path: Path, size=(100, 100)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [200, 200, 200]
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_for_deep(client, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        _create_test_image(img_dir / f"cell_{i:03d}.png", size=(80 + i * 10, 80 + i * 10))
    session = client.post("/sessions/", json={
        "name": "Deep Feature Test",
        "images_directory": str(img_dir),
    }).json()
    return session


# ── MobileNetV3 processor unit tests ──────────────────────────────────


def test_mobilenetv3_processor_extract():
    from core.processors.inference.deep_feature_extraction import MobileNetV3Extraction
    proc = MobileNetV3Extraction()
    assert proc.feature_dim() == 576

    # Create a dummy image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    features = proc.extract(img)
    assert features.shape == (576,)
    assert features.dtype == np.float32


def test_mobilenetv3_processor_grayscale():
    from core.processors.inference.deep_feature_extraction import MobileNetV3Extraction
    proc = MobileNetV3Extraction()
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    features = proc.extract(img)
    assert features.shape == (576,)


def test_mobilenetv3_processor_rgba():
    from core.processors.inference.deep_feature_extraction import MobileNetV3Extraction
    proc = MobileNetV3Extraction()
    img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    features = proc.extract(img)
    assert features.shape == (576,)


def test_mobilenetv3_deterministic():
    from core.processors.inference.deep_feature_extraction import MobileNetV3Extraction
    proc = MobileNetV3Extraction()
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    f1 = proc.extract(img)
    f2 = proc.extract(img)
    np.testing.assert_array_equal(f1, f2)


# ── Registry ──────────────────────────────────────────────────────────


def test_mobilenetv3_in_registry():
    from core.processors.inference.feature_extraction import FEATURE_EXTRACTION_REGISTRY
    assert "mobilenetv3" in FEATURE_EXTRACTION_REGISTRY


def test_list_methods_includes_mobilenetv3(client, session_for_deep):
    sid = session_for_deep["id"]
    resp = client.get(f"/sessions/{sid}/features/methods")
    assert resp.status_code == 200
    methods = resp.json()
    ids = [m["id"] for m in methods]
    assert "mobilenetv3" in ids
    mobilenet = next(m for m in methods if m["id"] == "mobilenetv3")
    assert mobilenet["feature_dim"] == 576


# ── End-to-end via API ────────────────────────────────────────────────


def test_deep_feature_extraction_sync(client, session_for_deep):
    from core.storage import features as lance_features
    sid = session_for_deep["id"]
    result = _run_extract(client, sid, {"method": "mobilenetv3"})
    assert result["processed"] == 5
    assert result["skipped"] == 0
    assert result["feature_dim"] == 576

    _, vectors = lance_features.load_all_vectors(sid, "mobilenetv3")
    assert vectors.shape == (5, 576)


def test_deep_features_different_from_zero(client, session_for_deep):
    from core.storage import features as lance_features
    sid = session_for_deep["id"]
    _run_extract(client, sid, {"method": "mobilenetv3"})

    _, vectors = lance_features.load_all_vectors(sid, "mobilenetv3")
    assert np.any(vectors != 0)


def test_deep_features_then_pca(client, session_for_deep):
    """Deep features → PCA should work end-to-end."""
    sid = session_for_deep["id"]

    result = _run_extract(client, sid, {"method": "mobilenetv3"})
    assert result["processed"] == 5

    data = _run_dim_reduction(client, sid, {"method": "pca", "n_components": 2})
    assert data["n_samples"] == 5
    assert len(data["embeddings"]) == 5

    xs = [e["x"] for e in data["embeddings"]]
    assert len(set(round(x, 6) for x in xs)) > 1


def test_deep_features_then_clustering(client, session_for_deep):
    """Deep features → clustering should work."""
    sid = session_for_deep["id"]

    _run_extract(client, sid, {"method": "mobilenetv3"})

    resp = client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})
    assert resp.status_code == 200
    assert resp.json()["total_samples_clustered"] == 5


# ── Real images ───────────────────────────────────────────────────────


def test_real_images_deep_features(client, real_images_dir):
    session = client.post("/sessions/", json={
        "name": "Blood Deep Features",
        "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    result = _run_extract(client, sid, {"method": "mobilenetv3"})
    assert result["processed"] > 0
    assert result["feature_dim"] == 576

    data = _run_dim_reduction(client, sid, {"method": "pca", "n_components": 2})
    assert len(data["embeddings"]) > 0
