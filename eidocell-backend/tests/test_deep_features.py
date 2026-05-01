"""Tests for deep feature extraction (MobileNetV3)."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image


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
    sid = session_for_deep["id"]
    resp = client.post(f"/sessions/{sid}/features/extract", json={
        "method": "mobilenetv3",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 5
    assert data["skipped"] == 0
    assert data["feature_dim"] == 576

    # Verify features file
    features_path = Path(session_for_deep["session_folder"]) / "features" / "session_features.npy"
    features = np.load(features_path)
    assert features.shape[1] == 576


def test_deep_features_different_from_zero(client, session_for_deep):
    sid = session_for_deep["id"]
    client.post(f"/sessions/{sid}/features/extract", json={"method": "mobilenetv3"})

    features_path = Path(session_for_deep["session_folder"]) / "features" / "session_features.npy"
    features = np.load(features_path)
    # Features should not be all zeros (model produces non-trivial output)
    assert np.any(features != 0)


def test_deep_features_then_pca(client, session_for_deep):
    """Deep features → PCA should work end-to-end."""
    sid = session_for_deep["id"]

    # Extract deep features
    resp = client.post(f"/sessions/{sid}/features/extract", json={"method": "mobilenetv3"})
    assert resp.json()["processed"] == 5

    # Run PCA
    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 5
    assert len(data["embeddings"]) == 5

    # With different-sized images, PCA should produce varied coordinates
    xs = [e["x"] for e in data["embeddings"]]
    assert len(set(round(x, 6) for x in xs)) > 1


def test_deep_features_then_clustering(client, session_for_deep):
    """Deep features → clustering should work."""
    sid = session_for_deep["id"]

    client.post(f"/sessions/{sid}/features/extract", json={"method": "mobilenetv3"})

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

    resp = client.post(f"/sessions/{sid}/features/extract", json={"method": "mobilenetv3"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] > 0
    assert data["feature_dim"] == 576

    # PCA on deep features
    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 2,
    })
    assert resp.status_code == 200
    embeddings = resp.json()["embeddings"]
    assert len(embeddings) > 0
