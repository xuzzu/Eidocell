"""End-to-end import-service tests via the FastAPI test client.

Walks the full path: POST /sessions/{sid}/imports → background task →
poll task → assert Sample rows + Lance images table populated.
"""
from __future__ import annotations

import time
from pathlib import Path
from textwrap import dedent

import numpy as np
from PIL import Image


def _write_png(path: Path, size: tuple[int, int] = (16, 24), value: int = 50):
    arr = np.full((*size, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _wait_for_task(client, task_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = client.get(f"/tasks/{task_id}").json()
        if info["status"] in ("completed", "failed", "cancelled"):
            return info
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish within {timeout_s}s")


def _create_session(client, name="Imp Test") -> str:
    return client.post("/sessions/", json={"name": name}).json()["id"]


def test_import_folder_skip_preprocessing(client, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(4):
        _write_png(img_dir / f"cell_{i:03d}.png", value=20 + i * 30)

    sid = _create_session(client)
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "channel_grouping": False,
        "preprocessing": None,
    })
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    info = _wait_for_task(client, task_id)
    assert info["status"] == "completed", info

    detail = client.get(f"/sessions/{sid}/imports/{resp.json()['import_id']}").json()
    assert detail["status"] == "completed"
    assert detail["sample_count"] == 4
    assert detail["skipped_count"] == 0

    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()
    assert samples["total"] == 4

    # Verify Lance images table holds the bytes.
    from core.storage import images as image_store
    arr = image_store.read_array(sid, samples["items"][0]["id"])
    assert arr is not None
    # PNG round-trips as RGB → HWC.
    assert arr.shape[:2] == (16, 24)


def test_import_folder_with_channel_grouping_and_preprocessing(client, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    # Two groups, two channels each, varying size to exercise pad-to-max.
    for i, hw in enumerate([(10, 12), (8, 14)]):
        for ch in (1, 2):
            arr = np.full((*hw, 3), 10 + i * 50 + ch * 5, dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / f"cell_{i:03d}_ch{ch}.png")

    sid = _create_session(client)
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "channel_grouping": True,
        "preprocessing": {
            "target_shape_strategy": "max",
            "padding_method": "constant",
            "normalize": "percentile",
        },
    })
    assert resp.status_code == 202, resp.text
    info = _wait_for_task(client, resp.json()["task_id"])
    assert info["status"] == "completed", info

    detail = client.get(f"/sessions/{sid}/imports/{resp.json()['import_id']}").json()
    assert detail["sample_count"] == 2  # one per group
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()
    assert samples["total"] == 2

    from core.storage import images as image_store
    arr = image_store.read_array(sid, samples["items"][0]["id"])
    assert arr is not None
    # max-strategy now yields a square: max(max_h, max_w) = max(10, 14) = 14.
    h, w = arr.shape[:2]
    assert (h, w) == (14, 14)

    # Metadata file should exist.
    sess = client.get(f"/sessions/{sid}").json()
    meta_path = Path(sess["session_folder"]) / "imports" / f"{resp.json()['import_id']}.json"
    assert meta_path.is_file()
    text = meta_path.read_text()
    assert "pipeline_id" in text
    assert "pad_to" in text


def test_import_csv_attrs_attached(client, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        _write_png(img_dir / f"r{i}.png")
    csv = tmp_path / "meta.csv"
    csv.write_text(dedent("""
        image,area,intensity
        r0.png,12.5,0.1
        r1.png,33.0,0.5
        r2.png,21.0,0.9
    """).strip())

    sid = _create_session(client)
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "csv_path": str(csv),
        "csv_filename_col": "image",
        "channel_grouping": False,
    })
    assert resp.status_code == 202
    info = _wait_for_task(client, resp.json()["task_id"])
    assert info["status"] == "completed", info

    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sids = [s["id"] for s in samples]
    from core.storage import sample_attrs
    bulk = sample_attrs.get_attrs_bulk(sid, sids)
    assert len(bulk) == 3
    for v in bulk.values():
        assert "area" in v and "intensity" in v


def test_import_bad_folder_400(client):
    sid = _create_session(client)
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": "/no/such/dir",
    })
    assert resp.status_code == 400


# ── _transform_mask_to_match ────────────────────────────────────────────


def test_transform_mask_exact_shape_returns_input():
    from services.import_service import _transform_mask_to_match
    mask = np.ones((10, 12), dtype=np.uint8)
    out = _transform_mask_to_match(mask, (10, 12), resize=False)
    assert out is mask  # no copy needed


def test_transform_mask_none_target_returns_input():
    from services.import_service import _transform_mask_to_match
    mask = np.ones((10, 12), dtype=np.uint8)
    out = _transform_mask_to_match(mask, None, resize=False)
    assert out is mask


def test_transform_mask_pads_smaller_source():
    """Source smaller than target in both dims → centre-pad with zeros."""
    from services.import_service import _transform_mask_to_match
    mask = np.ones((4, 6), dtype=np.uint8)
    out = _transform_mask_to_match(mask, (8, 10), resize=False)
    assert out.shape == (8, 10)
    # Centred: 2-row top/bottom border, 2-col left/right border, ones in middle
    assert out[:2, :].sum() == 0 and out[-2:, :].sum() == 0
    assert out[:, :2].sum() == 0 and out[:, -2:].sum() == 0
    assert out[2:6, 2:8].sum() == 4 * 6


def test_transform_mask_crops_larger_source():
    """Source larger than target in both dims → centre-crop."""
    from services.import_service import _transform_mask_to_match
    mask = np.arange(60, dtype=np.uint8).reshape(6, 10)
    out = _transform_mask_to_match(mask, (4, 6), resize=False)
    assert out.shape == (4, 6)
    # Centre crop: rows 1..5, cols 2..8 of the original.
    np.testing.assert_array_equal(out, mask[1:5, 2:8])


def test_transform_mask_mixed_pads_one_axis_crops_other():
    """Regression test for the import crash:

    Source shape (138, 93), target (138, 89): equal height, source wider →
    crop width, no padding needed. Previously raised ValueError because the
    pad-only branch couldn't shrink an axis.
    """
    from services.import_service import _transform_mask_to_match
    mask = np.ones((138, 93), dtype=np.uint8)
    out = _transform_mask_to_match(mask, (138, 89), resize=False)
    assert out.shape == (138, 89)
    assert out.sum() == 138 * 89  # nothing zero-padded since we cropped


def test_transform_mask_mixed_taller_and_narrower():
    """Source taller but narrower than target → pad width, crop height."""
    from services.import_service import _transform_mask_to_match
    mask = np.ones((20, 5), dtype=np.uint8)
    out = _transform_mask_to_match(mask, (10, 8), resize=False)
    assert out.shape == (10, 8)
    # Width pad: 1 col left, 2 cols right (or vice-versa with floor div)
    pad_left = (8 - 5) // 2
    assert out[:, :pad_left].sum() == 0
    assert out[:, pad_left + 5 :].sum() == 0
    # Height crop: rows 5..15 of source (all ones) → all ones across kept cols
    assert out[:, pad_left : pad_left + 5].sum() == 10 * 5


def test_transform_mask_resize_uses_nearest_neighbour():
    """resize=True scales the mask via NN, preserving binary values."""
    from services.import_service import _transform_mask_to_match
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    out = _transform_mask_to_match(mask, (4, 4), resize=True)
    assert out.shape == (4, 4)
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_transform_mask_preserves_dtype():
    from services.import_service import _transform_mask_to_match
    mask = np.ones((5, 5), dtype=np.uint16)
    out = _transform_mask_to_match(mask, (7, 7), resize=False)
    assert out.dtype == np.uint16


def test_import_folder_per_image_square_with_post_resize(client, tmp_path):
    """per_image_square + post_resize_strategy=max_longest produces uniformly sized N×N outputs."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    # Three images with different shapes and longest sides 12, 30, 40.
    shapes = [(10, 12), (30, 8), (40, 15)]
    for i, hw in enumerate(shapes):
        arr = np.full((*hw, 3), 20 + i * 30, dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"cell_{i:03d}.png")

    sid = _create_session(client)
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "channel_grouping": False,
        "preprocessing": {
            "target_shape_strategy": "per_image_square",
            "padding_method": "constant",
            "post_resize_strategy": "max_longest",
            "normalize": "none",
        },
    })
    assert resp.status_code == 202, resp.text
    info = _wait_for_task(client, resp.json()["task_id"])
    assert info["status"] == "completed", info

    detail = client.get(f"/sessions/{sid}/imports/{resp.json()['import_id']}").json()
    assert detail["sample_count"] == 3

    # Every output is square at max_longest = 40.
    from core.storage import images as image_store
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    for s in samples:
        arr = image_store.read_array(sid, s["id"])
        assert arr is not None
        h, w = arr.shape[:2]
        assert (h, w) == (40, 40), f"sample {s['id']} has shape {(h, w)}"

    # Pipeline metadata records both steps.
    sess = client.get(f"/sessions/{sid}").json()
    meta_path = Path(sess["session_folder"]) / "imports" / f"{resp.json()['import_id']}.json"
    text = meta_path.read_text()
    assert "pad_to_square" in text
    assert "resize_to" in text
