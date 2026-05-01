import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _create_test_image(path: Path, size=(100, 100)):
    """Create a test image with a bright circle on dark background."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [200, 200, 200]
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_with_masks(client, tmp_path):
    """Create a session, run segmentation so masks/attributes exist."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Feature Test",
        "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Run segmentation to generate mask attributes
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] == 10

    return session


# ── Feature extraction methods ─────────────────────────────────────────


def test_list_extraction_methods(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.get(f"/sessions/{sid}/features/methods")
    assert resp.status_code == 200
    methods = resp.json()
    assert len(methods) >= 1
    ids = [m["id"] for m in methods]
    assert "morphological" in ids
    for m in methods:
        assert m["feature_dim"] > 0


# ── Run feature extraction ─────────────────────────────────────────────


def test_run_feature_extraction(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.post(f"/sessions/{sid}/features/extract", json={
        "method": "morphological",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 10
    assert data["skipped"] == 0
    assert data["total"] == 10
    assert data["feature_dim"] == 15


def test_run_feature_extraction_default_method(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.post(f"/sessions/{sid}/features/extract", json={})
    assert resp.status_code == 200
    assert resp.json()["processed"] == 10


def test_run_feature_extraction_unknown_method(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.post(f"/sessions/{sid}/features/extract", json={
        "method": "nonexistent",
    })
    assert resp.status_code == 400


def test_run_feature_extraction_no_masks(client, tmp_path):
    """Without segmentation, samples have no masks → all skipped."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "No Masks",
        "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    resp = client.post(f"/sessions/{sid}/features/extract", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 0
    assert data["skipped"] == 3


def test_features_file_created(client, session_with_masks):
    sid = session_with_masks["id"]
    client.post(f"/sessions/{sid}/features/extract", json={})

    # Verify .npy file was created
    session_folder = session_with_masks["session_folder"]
    features_path = Path(session_folder) / "features" / "session_features.npy"
    assert features_path.exists()

    features = np.load(features_path)
    assert features.ndim == 2
    assert features.shape[1] == 15  # morphological feature dim


def test_clustering_works_after_extraction(client, session_with_masks):
    """End-to-end: segmentation → feature extraction → clustering."""
    sid = session_with_masks["id"]

    # Extract features
    resp = client.post(f"/sessions/{sid}/features/extract", json={})
    assert resp.status_code == 200

    # Now clustering should work
    resp = client.post(f"/sessions/{sid}/clusters/run", json={
        "n_clusters": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["clusters"]) == 2
    assert data["total_samples_clustered"] == 10


# ── Dimensionality reduction methods ──────────────────────────────────


def test_list_dim_reduction_methods(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.get(f"/sessions/{sid}/features/dim-reduction/methods")
    assert resp.status_code == 200
    methods = resp.json()
    ids = [m["id"] for m in methods]
    assert "pca" in ids


# ── Run dimensionality reduction ──────────────────────────────────────


def test_run_pca_2d(client, session_with_masks):
    sid = session_with_masks["id"]

    # Extract features first
    client.post(f"/sessions/{sid}/features/extract", json={})

    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "pca"
    assert data["n_components"] == 2
    assert data["n_samples"] == 10
    assert len(data["embeddings"]) == 10

    emb = data["embeddings"][0]
    assert "sample_id" in emb
    assert "filename" in emb
    assert "x" in emb
    assert "y" in emb
    assert "z" not in emb


def test_run_pca_3d(client, session_with_masks):
    sid = session_with_masks["id"]
    client.post(f"/sessions/{sid}/features/extract", json={})

    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_components"] == 3
    emb = data["embeddings"][0]
    assert "z" in emb


def test_run_dim_reduction_no_features(client, session_with_masks):
    """Dim reduction without feature extraction should fail."""
    sid = session_with_masks["id"]

    # Delete the features file if it exists
    session_folder = session_with_masks["session_folder"]
    features_path = Path(session_folder) / "features" / "session_features.npy"
    if features_path.exists():
        features_path.unlink()

    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 2,
    })
    assert resp.status_code == 400


def test_run_dim_reduction_unknown_method(client, session_with_masks):
    sid = session_with_masks["id"]
    client.post(f"/sessions/{sid}/features/extract", json={})

    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "nonexistent",
        "n_components": 2,
    })
    assert resp.status_code == 400


def test_embeddings_contain_class_info(client, session_with_masks):
    """Embeddings should include class_id for coloring in the UI."""
    sid = session_with_masks["id"]
    client.post(f"/sessions/{sid}/features/extract", json={})

    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 2,
    })
    assert resp.status_code == 200
    emb = resp.json()["embeddings"][0]
    assert "class_id" in emb


# ── Real images test ──────────────────────────────────────────────────


def test_real_images_feature_pipeline(client, real_images_dir):
    """End-to-end: segmentation → extraction → PCA on real images."""
    session = client.post("/sessions/", json={
        "name": "Blood Features",
        "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    # Segment
    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 50, "min_component_size": 15},
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] > 0

    # Extract features
    resp = client.post(f"/sessions/{sid}/features/extract", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] > 0
    assert data["feature_dim"] == 15

    # PCA 2D
    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "pca",
        "n_components": 2,
    })
    assert resp.status_code == 200
    embeddings = resp.json()["embeddings"]
    assert len(embeddings) > 0
    # Verify coordinates are not all identical
    xs = [e["x"] for e in embeddings]
    assert len(set(xs)) > 1
