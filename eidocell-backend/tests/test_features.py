import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.storage import features as lance_features


def _create_test_image(path: Path, size=(100, 100)):
    """Create a test image with a bright circle on dark background."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [200, 200, 200]
    Image.fromarray(img).save(path)


def _wait_for_task(client, task_id: str, timeout_s: float = 10.0) -> dict:
    """Poll /tasks/{id} until the task is no longer running."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish in {timeout_s}s")


def _run_extract(client, sid: str, body: dict | None = None) -> dict:
    """Submit feature extraction and wait for it to finish; return the task dict."""
    resp = client.post(f"/sessions/{sid}/features/extract", json=body or {})
    if resp.status_code != 200:
        return {"_http_status": resp.status_code, **resp.json()}
    return _wait_for_task(client, resp.json()["task_id"])


def _run_dim_reduction(client, sid: str, body: dict) -> dict:
    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json=body)
    if resp.status_code != 200:
        return {"_http_status": resp.status_code, **resp.json()}
    return _wait_for_task(client, resp.json()["task_id"])


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
    task = _run_extract(client, sid, {"method": "morphological"})
    assert task["status"] == "completed"
    result = task["result"]
    assert result["processed"] == 10
    assert result["skipped"] == 0
    assert result["total"] == 10
    assert result["feature_dim"] == 15


def test_run_feature_extraction_default_method(client, session_with_masks):
    sid = session_with_masks["id"]
    task = _run_extract(client, sid, {})
    assert task["status"] == "completed"
    assert task["result"]["processed"] == 10


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

    task = _run_extract(client, sid, {})
    assert task["status"] == "completed"
    assert task["result"]["processed"] == 0
    assert task["result"]["skipped"] == 3


def test_features_file_created(client, session_with_masks):
    sid = session_with_masks["id"]
    _run_extract(client, sid, {"method": "morphological"})

    # Verify Lance has 10 rows for morphological in this session
    sample_ids, vectors = lance_features.load_all_vectors(sid, "morphological")
    assert len(sample_ids) == 10
    assert vectors.ndim == 2
    assert vectors.shape[1] == 15


def test_clustering_works_after_extraction(client, session_with_masks):
    """End-to-end: segmentation → feature extraction → clustering."""
    sid = session_with_masks["id"]

    task = _run_extract(client, sid, {"method": "morphological"})
    assert task["status"] == "completed"

    resp = client.post(f"/sessions/{sid}/clusters/run", json={
        "n_clusters": 2, "feature_method": "morphological",
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
    _run_extract(client, sid, {"method": "morphological"})

    task = _run_dim_reduction(client, sid, {
        "method": "pca", "n_components": 2, "feature_method": "morphological",
    })
    assert task["status"] == "completed"
    result = task["result"]
    assert result["method"] == "pca"
    assert result["n_components"] == 2
    assert result["n_samples"] == 10
    assert len(result["embeddings"]) == 10

    emb = result["embeddings"][0]
    assert "sample_id" in emb
    assert "filename" in emb
    assert "x" in emb
    assert "y" in emb
    assert "z" not in emb


def test_run_pca_3d(client, session_with_masks):
    sid = session_with_masks["id"]
    _run_extract(client, sid, {"method": "morphological"})

    task = _run_dim_reduction(client, sid, {
        "method": "pca", "n_components": 3, "feature_method": "morphological",
    })
    assert task["status"] == "completed"
    result = task["result"]
    assert result["n_components"] == 3
    assert "z" in result["embeddings"][0]


def test_run_dim_reduction_no_features(client, session_with_masks):
    """Dim reduction without feature extraction should fail."""
    sid = session_with_masks["id"]
    # No extraction has run; Lance has no features for any method.
    task = _run_dim_reduction(client, sid, {"method": "pca", "n_components": 2})
    assert task["status"] == "failed"


def test_run_dim_reduction_unknown_method(client, session_with_masks):
    sid = session_with_masks["id"]
    _run_extract(client, sid, {"method": "morphological"})

    resp = client.post(f"/sessions/{sid}/features/dim-reduction/run", json={
        "method": "nonexistent",
        "n_components": 2,
    })
    assert resp.status_code == 400


def test_embeddings_contain_class_info(client, session_with_masks):
    """Embeddings should include class_id for coloring in the UI."""
    sid = session_with_masks["id"]
    _run_extract(client, sid, {"method": "morphological"})

    task = _run_dim_reduction(client, sid, {
        "method": "pca", "n_components": 2, "feature_method": "morphological",
    })
    assert task["status"] == "completed"
    assert "class_id" in task["result"]["embeddings"][0]


# ── Real images test ──────────────────────────────────────────────────


def test_real_images_feature_pipeline(client, real_images_dir):
    """End-to-end: segmentation → extraction → PCA on real images."""
    session = client.post("/sessions/", json={
        "name": "Blood Features",
        "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    resp = client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 50, "min_component_size": 15},
    })
    assert resp.status_code == 200
    assert resp.json()["processed"] > 0

    task = _run_extract(client, sid, {})
    assert task["status"] == "completed"
    assert task["result"]["processed"] > 0
    assert task["result"]["feature_dim"] == 15

    task = _run_dim_reduction(client, sid, {"method": "pca", "n_components": 2})
    assert task["status"] == "completed"
    embeddings = task["result"]["embeddings"]
    assert len(embeddings) > 0
    xs = [e["x"] for e in embeddings]
    assert len(set(xs)) > 1
