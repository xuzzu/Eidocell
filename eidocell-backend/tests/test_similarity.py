from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_session(client, tmp_path, name="Sim Test", n_samples=20):
    img_dir = tmp_path / name.replace(" ", "_")
    img_dir.mkdir()
    for i in range(n_samples):
        Image.new("RGB", (32, 32), color=(i * 10, 100, 50)).save(
            img_dir / f"cell_{i:03d}.png"
        )
    session = client.post("/sessions/", json={
        "name": name,
        "images_directory": str(img_dir),
    }).json()
    return session


@pytest.fixture()
def session_with_features(client, tmp_path):
    """Session with two clear feature groups (10 + 10)."""
    session = _make_session(client, tmp_path, n_samples=20)
    session_folder = Path(session["session_folder"])
    features_dir = session_folder / "features"
    features_dir.mkdir(exist_ok=True)

    rng = np.random.RandomState(42)
    # Group A near +5, group B near -5; positive vectors are guaranteed
    # by adding a large offset so cosine stays in [0, 1].
    features = np.vstack([
        rng.randn(10, 32) * 0.1 + np.array([10.0] * 32),  # group A
        np.abs(rng.randn(10, 32)) * 0.1 + np.array([0.5] * 16 + [10.0] * 16),  # group B
    ]).astype(np.float32)
    np.save(features_dir / "session_features.npy", features)
    return session


def _list_samples(client, sid):
    return client.post(f"/sessions/{sid}/samples/list", json={
        "offset": 0, "limit": 100, "sort_by": "storage_index", "sort_order": "asc",
    }).json()["items"]


# ── Basic search ────────────────────────────────────────────────────────


def test_search_excludes_reference(client, session_with_features):
    sid = session_with_features["id"]
    samples = _list_samples(client, sid)
    ref = samples[0]

    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [ref["id"]],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert ref["id"] not in [h["sample"]["id"] for h in body["hits"]]
    # All other 19 samples should be candidates
    assert body["total_candidates"] == 19


def test_search_ranks_intra_group_higher(client, session_with_features):
    """A sample from group A should rank other group-A samples above group-B samples."""
    sid = session_with_features["id"]
    samples = _list_samples(client, sid)
    # Group A is first 10 (storage_index 0..9), group B is 10..19
    ref = samples[0]  # group A

    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [ref["id"]],
    })
    body = resp.json()
    hits_by_id = {h["sample"]["id"]: h for h in body["hits"]}
    # First 9 hits (excluding ref) should all be group A
    top9_ids = [h["sample"]["id"] for h in body["hits"][:9]]
    group_a_ids = {s["id"] for s in samples[:10]}
    assert set(top9_ids).issubset(group_a_ids), (
        f"Top 9 should be group A, got: {top9_ids} vs A={group_a_ids}"
    )
    # All similarities should be in [0, 100]
    for h in body["hits"]:
        assert 0.0 <= h["similarity_pct"] <= 100.0
        # bucket is floor(pct/10)*10, capped at 90
        assert h["bucket"] in {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def test_search_multi_reference_centroid(client, session_with_features):
    sid = session_with_features["id"]
    samples = _list_samples(client, sid)
    refs = [samples[0]["id"], samples[1]["id"]]  # two from group A

    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": refs,
    })
    body = resp.json()
    # Refs are excluded
    hit_ids = [h["sample"]["id"] for h in body["hits"]]
    assert refs[0] not in hit_ids and refs[1] not in hit_ids
    # The remaining 8 group-A samples should outrank group-B
    top8_ids = hit_ids[:8]
    group_a_ids = {s["id"] for s in samples[:10]}
    assert set(top8_ids).issubset(group_a_ids)


# ── Filter mode ─────────────────────────────────────────────────────────


def test_search_unlabeled_filter(client, session_with_features):
    sid = session_with_features["id"]
    samples = _list_samples(client, sid)
    ref = samples[0]

    # Create a class and assign half of group A to it (samples 1..5)
    cls = client.post(f"/sessions/{sid}/classes", json={
        "name": "Labeled", "color": "#FF0000",
    }).json()
    to_assign = [s["id"] for s in samples[1:6]]
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": to_assign, "class_id": cls["id"],
    })

    # Unlabeled-only search should not return the labeled samples
    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [ref["id"]],
        "filter_mode": "unlabeled",
    })
    body = resp.json()
    hit_ids = {h["sample"]["id"] for h in body["hits"]}
    assert hit_ids.isdisjoint(set(to_assign))

    # All-mode search includes them
    resp_all = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [ref["id"]],
        "filter_mode": "all",
    })
    all_ids = {h["sample"]["id"] for h in resp_all.json()["hits"]}
    assert set(to_assign).issubset(all_ids)


# ── Threshold + top_k ────────────────────────────────────────────────────


def test_search_min_similarity_threshold(client, session_with_features):
    sid = session_with_features["id"]
    samples = _list_samples(client, sid)
    ref = samples[0]

    # Threshold 99% → only very similar samples (group A) survive
    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [ref["id"]],
        "min_similarity_pct": 99.0,
    })
    body = resp.json()
    for h in body["hits"]:
        assert h["similarity_pct"] >= 99.0


def test_search_top_k_caps_results(client, session_with_features):
    sid = session_with_features["id"]
    samples = _list_samples(client, sid)
    ref = samples[0]

    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [ref["id"]],
        "top_k": 3,
    })
    body = resp.json()
    assert body["returned"] == 3
    assert len(body["hits"]) == 3


# ── Error cases ─────────────────────────────────────────────────────────


def test_search_no_features(client, tmp_path):
    session = _make_session(client, tmp_path, name="No Features", n_samples=5)
    sid = session["id"]
    samples = _list_samples(client, sid)
    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [samples[0]["id"]],
    })
    assert resp.status_code == 400
    assert "features" in resp.json()["detail"].lower()


def test_search_unknown_reference(client, session_with_features):
    sid = session_with_features["id"]
    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": ["does-not-exist"],
    })
    assert resp.status_code == 400


def test_search_empty_references_rejected(client, session_with_features):
    sid = session_with_features["id"]
    resp = client.post(f"/sessions/{sid}/similarity/search", json={
        "reference_sample_ids": [],
    })
    # Pydantic min_length=1
    assert resp.status_code == 422
