"""End-to-end check: import → gallery thumbnail/full image route serves from
the Lance images table (not from any on-disk source path)."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image


def _wait(client, task_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.get(f"/tasks/{task_id}").json()
        if info["status"] in ("completed", "failed", "cancelled"):
            return info
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_thumbnail_served_from_lance(client, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.full((20, 30, 3), 80, dtype=np.uint8)
    arr[5:15, 10:20] = 200
    Image.fromarray(arr).save(img_dir / "p.png")

    sid = client.post("/sessions/", json={"name": "Lance thumb"}).json()["id"]
    sub = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "channel_grouping": False,
    })
    info = _wait(client, sub.json()["task_id"])
    assert info["status"] == "completed", info

    sample_id = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"][0]["id"]

    # Now delete the source PNG to prove serving doesn't depend on it.
    (img_dir / "p.png").unlink()

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    # JPEG magic bytes.
    assert resp.content[:3] == b"\xff\xd8\xff"

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/image")
    assert resp.status_code == 200
    # PNG magic bytes (rendered from Lance ndarray).
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
