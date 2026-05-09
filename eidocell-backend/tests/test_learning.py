"""Tests for the learning module: training, inference, cross-session, model registry."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from tests._helpers import list_samples, seed_features


def _create_test_image(path: Path, size=(100, 100), brightness=128):
    """Create a test image with configurable brightness for class separation."""
    img = np.full((*size, 3), brightness, dtype=np.uint8)
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_with_labeled_data(client, tmp_path):
    """Session with 20 samples: 8 labeled (2 classes), 12 unlabeled.
    Features are well-separated for reliable classification.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(20):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Learning Test",
        "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    samples = list_samples(client, sid)
    rng = np.random.RandomState(42)
    features = np.vstack([
        rng.randn(10, 32).astype(np.float32) + 5,   # first 10 samples → class A
        rng.randn(10, 32).astype(np.float32) - 5,   # last 10 samples → class B
    ])
    seed_features(sid, [s["id"] for s in samples], features, method="raw")

    # Create two classes
    class_a = client.post(f"/sessions/{sid}/classes", json={
        "name": "Round", "color": "#FF0000",
    }).json()
    class_b = client.post(f"/sessions/{sid}/classes", json={
        "name": "Elongated", "color": "#0000FF",
    }).json()

    # First 4 samples → Round, samples 10-13 → Elongated
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples[:4]],
        "class_id": class_a["id"],
    })
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples[10:14]],
        "class_id": class_b["id"],
    })

    return {
        "session": session,
        "class_a": class_a,
        "class_b": class_b,
        "samples": samples,
    }


# ── Training ──────────────────────────────────────────────────────────


def test_train_knn(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]

    resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Test KNN",
        "classifier_type": "knn",
        "feature_source": "raw",
        "params": {"n_neighbors": 3},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"]
    assert data["name"] == "Test KNN"
    assert data["classifier_type"] == "knn"
    assert data["n_classes"] == 2
    assert data["n_training_samples"] == 8
    assert data["accuracy"] == 1.0  # well-separated data
    assert "0" in data["label_mapping"]
    assert "1" in data["label_mapping"]
    label_names = set(data["label_mapping"].values())
    assert label_names == {"Round", "Elongated"}


def test_train_linear_probe(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]

    resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Test LP",
        "classifier_type": "linear_probe",
        "feature_source": "raw",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["classifier_type"] == "linear_probe"
    assert data["accuracy"] == 1.0
    assert data["n_classes"] == 2


def test_train_unknown_classifier(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]
    resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Bad", "classifier_type": "unknown",
    })
    assert resp.status_code == 400


def test_train_no_features(client, tmp_path):
    """Training without features should fail."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "No Features", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Fail", "classifier_type": "knn",
    })
    assert resp.status_code == 400


def test_train_insufficient_classes(client, tmp_path):
    """Training with only 1 class should fail."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "One Class", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Create features
    samples = list_samples(client, sid)
    seed_features(
        sid, [s["id"] for s in samples],
        np.random.randn(5, 16).astype(np.float32),
        method="raw",
    )

    # Create one class and assign samples
    cls = client.post(f"/sessions/{sid}/classes", json={
        "name": "OnlyOne", "color": "#FF0000",
    }).json()
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples],
        "class_id": cls["id"],
    })

    resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Fail", "classifier_type": "knn",
    })
    assert resp.status_code == 400


# ── Inference ─────────────────────────────────────────────────────────


def test_infer_assigns_unlabeled(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]
    samples = session_with_labeled_data["samples"]

    # Train
    train_resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Infer Test", "classifier_type": "knn", "params": {"n_neighbors": 3},
    }).json()
    model_id = train_resp["model_id"]

    # Count unlabeled before
    samples_before = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    uncategorized_before = sum(1 for s in samples_before if s["class_name"] == "Uncategorized")
    assert uncategorized_before == 12  # 20 - 4 - 4

    # Infer
    resp = client.post(f"/sessions/{sid}/learning/infer", json={
        "model_id": model_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_predicted"] == 12
    assert data["classes_created"] == []  # classes already exist
    assert sum(data["assignments"].values()) == 12

    # Verify no more Uncategorized
    samples_after = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    uncategorized_after = sum(1 for s in samples_after if s["class_name"] == "Uncategorized")
    assert uncategorized_after == 0


def test_infer_correct_predictions(client, session_with_labeled_data):
    """With well-separated features, predictions should match the expected clusters."""
    sid = session_with_labeled_data["session"]["id"]

    train_resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Accuracy Test", "classifier_type": "knn", "params": {"n_neighbors": 3},
    }).json()

    resp = client.post(f"/sessions/{sid}/learning/infer", json={
        "model_id": train_resp["model_id"],
    }).json()

    # Samples 0-9 have features at +5, samples 10-19 at -5
    # Samples 0-3 labeled Round, 10-13 labeled Elongated
    # So samples 4-9 should be predicted Round, 14-19 Elongated
    assert resp["assignments"].get("Round", 0) == 6   # samples 4-9
    assert resp["assignments"].get("Elongated", 0) == 6  # samples 14-19


def test_infer_no_unlabeled(client, session_with_labeled_data):
    """If all samples are labeled, inference returns 0."""
    sid = session_with_labeled_data["session"]["id"]
    class_a = session_with_labeled_data["class_a"]
    class_b = session_with_labeled_data["class_b"]

    # Re-fetch samples to get current class assignments
    samples = list_samples(client, sid)

    # Assign remaining Uncategorized to one of the two classes
    uncategorized = [s["id"] for s in samples if s["class_name"] == "Uncategorized"]
    # Split: first half → Round, second half → Elongated
    mid = len(uncategorized) // 2
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": uncategorized[:mid],
        "class_id": class_a["id"],
    })
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": uncategorized[mid:],
        "class_id": class_b["id"],
    })

    train_resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "All Labeled", "classifier_type": "knn",
    })
    assert train_resp.status_code == 200

    resp = client.post(f"/sessions/{sid}/learning/infer", json={
        "model_id": train_resp.json()["model_id"],
    }).json()
    assert resp["total_predicted"] == 0


# ── Model registry ───────────────────────────────────────────────────


def test_list_models(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]

    client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Model 1", "classifier_type": "knn",
    })
    client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Model 2", "classifier_type": "linear_probe",
    })

    # Session-scoped list
    resp = client.get(f"/sessions/{sid}/learning/models")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    # Global list
    resp = client.get("/models/")
    assert resp.status_code == 200
    models = resp.json()
    assert any(m["name"] == "Model 1" for m in models)


def test_get_model(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]

    train_resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Get Test", "classifier_type": "knn",
    }).json()

    resp = client.get(f"/models/{train_resp['model_id']}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Get Test"
    assert resp.json()["feature_dim"] == 32


def test_delete_model(client, session_with_labeled_data):
    sid = session_with_labeled_data["session"]["id"]

    train_resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Delete Test", "classifier_type": "knn",
    }).json()
    model_id = train_resp["model_id"]

    resp = client.delete(f"/models/{model_id}")
    assert resp.status_code == 204

    resp = client.get(f"/models/{model_id}")
    assert resp.status_code == 404


def test_delete_model_not_found(client):
    resp = client.delete("/models/nonexistent")
    assert resp.status_code == 404


# ── Cross-session ────────────────────────────────────────────────────


def test_cross_session_apply(client, session_with_labeled_data, tmp_path):
    """Train in session A, apply to session B."""
    sid_a = session_with_labeled_data["session"]["id"]

    # Train model in session A
    train_resp = client.post(f"/sessions/{sid_a}/learning/train", json={
        "name": "Cross Session", "classifier_type": "knn", "params": {"n_neighbors": 3},
    }).json()
    model_id = train_resp["model_id"]

    # Create session B with same feature structure
    img_dir_b = tmp_path / "images_b"
    img_dir_b.mkdir()
    for i in range(10):
        _create_test_image(img_dir_b / f"cell_{i:03d}.png")

    session_b = client.post("/sessions/", json={
        "name": "Session B", "images_directory": str(img_dir_b),
    }).json()
    sid_b = session_b["id"]

    # Create matching features (32-dim, same as session A)
    samples_b = list_samples(client, sid_b)
    rng = np.random.RandomState(99)
    features_b = np.vstack([
        rng.randn(5, 32).astype(np.float32) + 5,   # should be Round
        rng.randn(5, 32).astype(np.float32) - 5,   # should be Elongated
    ])
    seed_features(sid_b, [s["id"] for s in samples_b], features_b, method="raw")

    # Apply model to session B
    resp = client.post(f"/models/{model_id}/apply", json={
        "target_session_id": sid_b,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_predicted"] == 10
    # Classes should be auto-created in session B
    assert set(data["classes_created"]) == {"Round", "Elongated"}
    assert data["assignments"]["Round"] == 5
    assert data["assignments"]["Elongated"] == 5

    # Verify classes exist in session B
    classes_b = client.get(f"/sessions/{sid_b}/classes").json()
    class_names = [c["name"] for c in classes_b]
    assert "Round" in class_names
    assert "Elongated" in class_names


def test_cross_session_feature_dim_mismatch(client, session_with_labeled_data, tmp_path):
    """Applying model with wrong feature dimensions should fail."""
    sid_a = session_with_labeled_data["session"]["id"]

    train_resp = client.post(f"/sessions/{sid_a}/learning/train", json={
        "name": "Dim Mismatch", "classifier_type": "knn",
    }).json()
    model_id = train_resp["model_id"]

    # Create session B with different feature dim (16 instead of 32)
    img_dir_b = tmp_path / "images_b"
    img_dir_b.mkdir()
    for i in range(5):
        _create_test_image(img_dir_b / f"cell_{i:03d}.png")

    session_b = client.post("/sessions/", json={
        "name": "Wrong Dim", "images_directory": str(img_dir_b),
    }).json()
    sid_b = session_b["id"]

    samples_b = list_samples(client, sid_b)
    seed_features(
        sid_b, [s["id"] for s in samples_b],
        np.random.randn(5, 16).astype(np.float32),  # 16-dim != 32-dim
        method="raw",
    )

    resp = client.post(f"/models/{model_id}/apply", json={
        "target_session_id": sid_b,
    })
    assert resp.status_code == 400
    assert "mismatch" in resp.json()["detail"].lower()


def test_cross_session_no_features(client, session_with_labeled_data, tmp_path):
    """Applying model to session without features should fail."""
    sid_a = session_with_labeled_data["session"]["id"]

    train_resp = client.post(f"/sessions/{sid_a}/learning/train", json={
        "name": "No Features Target", "classifier_type": "knn",
    }).json()

    img_dir_b = tmp_path / "images_b"
    img_dir_b.mkdir()
    for i in range(3):
        _create_test_image(img_dir_b / f"cell_{i:03d}.png")

    session_b = client.post("/sessions/", json={
        "name": "No Features", "images_directory": str(img_dir_b),
    }).json()

    resp = client.post(f"/models/{train_resp['model_id']}/apply", json={
        "target_session_id": session_b["id"],
    })
    assert resp.status_code == 400


# ── Real images ───────────────────────────────────────────────────────


def test_real_images_learning_pipeline(client, real_images_dir):
    """End-to-end: real images → segmentation → features → train → infer."""
    session = client.post("/sessions/", json={
        "name": "Blood Learning", "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    # Segment
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 50, "min_component_size": 15},
    })

    # Extract morphological features
    resp = client.post(f"/sessions/{sid}/features/extract", json={"method": "morphological"})
    assert resp.json()["processed"] > 0

    # Assign some samples to classes
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    if len(samples) < 6:
        pytest.skip("Not enough samples")

    cls_a = client.post(f"/sessions/{sid}/classes", json={
        "name": "TypeA", "color": "#FF0000",
    }).json()
    cls_b = client.post(f"/sessions/{sid}/classes", json={
        "name": "TypeB", "color": "#0000FF",
    }).json()

    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples[:3]],
        "class_id": cls_a["id"],
    })
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples[3:6]],
        "class_id": cls_b["id"],
    })

    # Train
    resp = client.post(f"/sessions/{sid}/learning/train", json={
        "name": "Blood KNN", "classifier_type": "knn", "params": {"n_neighbors": 3},
    })
    assert resp.status_code == 200
    model_id = resp.json()["model_id"]

    # Infer
    resp = client.post(f"/sessions/{sid}/learning/infer", json={"model_id": model_id})
    assert resp.status_code == 200
    assert resp.json()["total_predicted"] > 0
