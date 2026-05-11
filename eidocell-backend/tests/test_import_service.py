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
            "normalize": "zscore",
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
    # max H, W = 10, 14; 6 channels (3 RGB * 2 source files stacked).
    h, w = arr.shape[:2]
    assert (h, w) == (10, 14)

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
