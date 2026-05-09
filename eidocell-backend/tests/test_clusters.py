import numpy as np
import pytest
from PIL import Image

from tests._helpers import list_samples, seed_features


@pytest.fixture()
def session_with_features(client, tmp_path):
    """Create a session with images and fake features for clustering."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    n_samples = 20
    for i in range(n_samples):
        img = Image.new("RGB", (64, 64), color=(i * 12, 100, 50))
        img.save(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Cluster Test",
        "images_directory": str(img_dir),
    }).json()

    # Create fake features (two clear groups), aligned to filename order
    samples = list_samples(client, session["id"])
    rng = np.random.RandomState(42)
    features = np.vstack([
        rng.randn(10, 32) + np.array([5] * 32),    # group 1
        rng.randn(10, 32) + np.array([-5] * 32),   # group 2
    ]).astype(np.float32)
    seed_features(session["id"], [s["id"] for s in samples], features)

    return session


# ── Run clustering ──────────────────────────────────────────────────────


def test_run_clustering(client, session_with_features):
    sid = session_with_features["id"]
    resp = client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["clusters"]) == 2
    assert data["total_samples_clustered"] == 20
    # Each cluster should have ~10 samples given the clear separation
    counts = sorted(c["sample_count"] for c in data["clusters"])
    assert counts == [10, 10]


def test_run_clustering_no_features(client, tmp_path):
    """Should fail if no features exist."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 32)).save(img_dir / f"cell_{i}.png")

    session = client.post("/sessions/", json={
        "name": "No Features", "images_directory": str(img_dir),
    }).json()

    resp = client.post(f"/sessions/{session['id']}/clusters/run", json={"n_clusters": 2})
    assert resp.status_code == 400
    assert "features" in resp.json()["detail"].lower()


def test_run_clustering_too_many_clusters(client, session_with_features):
    sid = session_with_features["id"]
    resp = client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 50})
    assert resp.status_code == 400


# ── List & delete ───────────────────────────────────────────────────────


def test_list_clusters_empty(client, session_with_features):
    sid = session_with_features["id"]
    resp = client.get(f"/sessions/{sid}/clusters/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_clusters_after_run(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 3})

    resp = client.get(f"/sessions/{sid}/clusters/")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_delete_single_cluster(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 3})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    cid = clusters[0]["id"]

    resp = client.delete(f"/sessions/{sid}/clusters/{cid}")
    assert resp.status_code == 204

    remaining = client.get(f"/sessions/{sid}/clusters/").json()
    assert len(remaining) == 2


def test_clear_all_clusters(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 3})

    resp = client.delete(f"/sessions/{sid}/clusters/")
    assert resp.status_code == 204

    remaining = client.get(f"/sessions/{sid}/clusters/").json()
    assert len(remaining) == 0


# ── Cluster samples & preview ──────────────────────────────────────────


def test_get_cluster_samples(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    cid = clusters[0]["id"]
    expected_count = clusters[0]["sample_count"]

    resp = client.get(f"/sessions/{sid}/clusters/{cid}/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == expected_count
    assert len(data["items"]) == expected_count


def test_get_cluster_preview(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    cid = clusters[0]["id"]

    resp = client.get(f"/sessions/{sid}/clusters/{cid}/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


# ── Split ───────────────────────────────────────────────────────────────


def test_split_cluster(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    big_cluster = max(clusters, key=lambda c: c["sample_count"])

    resp = client.post(
        f"/sessions/{sid}/clusters/{big_cluster['id']}/split",
        json={"n_sub_clusters": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_cluster_id"] == big_cluster["id"]
    new_clusters = body["new_clusters"]
    assert len(new_clusters) == 3
    assert sum(c["sample_count"] for c in new_clusters) == big_cluster["sample_count"]

    # Original cluster should be gone, total = 1 remaining + 3 new = 4
    all_clusters = client.get(f"/sessions/{sid}/clusters/").json()
    assert len(all_clusters) == 4
    assert big_cluster["id"] not in {c["id"] for c in all_clusters}


# ── Merge ───────────────────────────────────────────────────────────────


def test_merge_clusters(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 3})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    ids_to_merge = [clusters[0]["id"], clusters[1]["id"]]
    expected_count = clusters[0]["sample_count"] + clusters[1]["sample_count"]

    resp = client.post(f"/sessions/{sid}/clusters/merge", json={
        "cluster_ids": ids_to_merge,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["deleted_cluster_ids"]) == set(ids_to_merge)
    merged = body["new_cluster"]
    assert merged["sample_count"] == expected_count

    # Should have 2 clusters now: 1 untouched + 1 merged
    all_clusters = client.get(f"/sessions/{sid}/clusters/").json()
    assert len(all_clusters) == 2
    assert set(all_clusters[i]["id"] for i in range(len(all_clusters))).isdisjoint(set(ids_to_merge))


def test_merge_clusters_insufficient(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    resp = client.post(f"/sessions/{sid}/clusters/merge", json={
        "cluster_ids": [clusters[0]["id"]],
    })
    # Pydantic validation: min_length=2
    assert resp.status_code == 422


# ── Assign to class ────────────────────────────────────────────────────


def test_assign_clusters_to_class(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    # Create a class
    new_class = client.post(f"/sessions/{sid}/classes", json={
        "name": "Elongated", "color": "#FF0000",
    }).json()

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    target_cluster = clusters[0]

    resp = client.post(f"/sessions/{sid}/clusters/assign-class", json={
        "cluster_ids": [target_cluster["id"]],
        "class_id": new_class["id"],
    })
    assert resp.status_code == 200
    assert resp.json()["updated"] == target_cluster["sample_count"]

    # Verify class has those samples
    class_info = client.get(f"/sessions/{sid}/classes/{new_class['id']}/summary").json()
    assert class_info["sample_count"] == target_cluster["sample_count"]


# ── Re-clustering clears old clusters ──────────────────────────────────


def test_rerun_clustering_clears_old(client, session_with_features):
    sid = session_with_features["id"]

    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 3})
    assert len(client.get(f"/sessions/{sid}/clusters/").json()) == 3

    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 5})
    assert len(client.get(f"/sessions/{sid}/clusters/").json()) == 5


# ── New metric fields ──────────────────────────────────────────────────


def test_list_clusters_includes_quality_and_labeled(client, session_with_features):
    """list_clusters should expose quality_score and labeled_count after a run."""
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    for c in clusters:
        assert "quality_score" in c
        assert "labeled_count" in c
        # quality_score is computed during run_clustering on the feature space
        assert c["quality_score"] is None or c["quality_score"] >= 0
        # New samples are auto-assigned to "Uncategorized", which doesn't count
        # toward labeled_count.
        assert c["labeled_count"] == 0


def test_labeled_count_after_assign(client, session_with_features):
    sid = session_with_features["id"]
    client.post(f"/sessions/{sid}/clusters/run", json={"n_clusters": 2})

    new_class = client.post(f"/sessions/{sid}/classes", json={
        "name": "Group A", "color": "#00FF00",
    }).json()

    clusters = client.get(f"/sessions/{sid}/clusters/").json()
    target = clusters[0]
    client.post(f"/sessions/{sid}/clusters/assign-class", json={
        "cluster_ids": [target["id"]],
        "class_id": new_class["id"],
    })

    refreshed = client.get(f"/sessions/{sid}/clusters/").json()
    by_id = {c["id"]: c for c in refreshed}
    assert by_id[target["id"]]["labeled_count"] == target["sample_count"]
