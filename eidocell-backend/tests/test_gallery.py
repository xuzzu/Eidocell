import pytest
from PIL import Image


@pytest.fixture()
def session_with_images(client, tmp_path):
    """Create a session with real PNG images and return (session_data, images_dir)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(i * 25, 50, 100))
        img.save(img_dir / f"cell_{i:03d}.png")

    resp = client.post("/sessions/", json={
        "name": "Gallery Test",
        "images_directory": str(img_dir),
    })
    assert resp.status_code == 201
    return resp.json()


# ── Sample listing ───────────────────────────────────────────────────────


def test_list_samples_default(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.post(f"/sessions/{sid}/samples/list", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 10
    assert len(data["items"]) == 10
    assert data["items"][0]["class_name"] == "Uncategorized"


def test_list_samples_pagination(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "offset": 3, "limit": 4,
    })
    data = resp.json()
    assert data["total"] == 10
    assert len(data["items"]) == 4
    assert data["offset"] == 3
    assert data["limit"] == 4


def test_list_samples_sort_desc(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "sort_by": "filename", "sort_order": "desc",
    })
    items = resp.json()["items"]
    filenames = [s["filename"] for s in items]
    assert filenames == sorted(filenames, reverse=True)


def test_list_samples_filter_contains(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "filters": [{"field": "filename", "operator": "contains", "value": "003"}],
    })
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["filename"] == "cell_003.png"


def test_list_samples_filter_not_contains(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "filters": [{"field": "filename", "operator": "not_contains", "value": "00"}],
    })
    # All filenames contain "00" (cell_000 through cell_009), so nothing matches
    assert resp.json()["total"] == 0


# ── Sample detail & images ──────────────────────────────────────────────


def test_get_sample(client, session_with_images):
    sid = session_with_images["id"]
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_id = samples[0]["id"]

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}")
    assert resp.status_code == 200
    assert resp.json()["filename"] == samples[0]["filename"]


def test_get_sample_image(client, session_with_images):
    sid = session_with_images["id"]
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_id = samples[0]["id"]

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_get_sample_thumbnail(client, session_with_images):
    sid = session_with_images["id"]
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_id = samples[0]["id"]

    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"

    # Second request should use cached thumbnail
    resp2 = client.get(f"/sessions/{sid}/samples/{sample_id}/thumbnail")
    assert resp2.status_code == 200


# ── Classes ──────────────────────────────────────────────────────────────


def test_list_classes_default(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.get(f"/sessions/{sid}/classes")
    assert resp.status_code == 200
    classes = resp.json()
    assert len(classes) == 1
    assert classes[0]["name"] == "Uncategorized"
    assert classes[0]["sample_count"] == 10


def test_create_class(client, session_with_images):
    sid = session_with_images["id"]
    resp = client.post(f"/sessions/{sid}/classes", json={
        "name": "Elongated", "color": "#FF0000",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Elongated"
    assert data["color"] == "#FF0000"
    assert data["sample_count"] == 0


def test_create_class_duplicate(client, session_with_images):
    sid = session_with_images["id"]
    client.post(f"/sessions/{sid}/classes", json={"name": "Type A"})
    resp = client.post(f"/sessions/{sid}/classes", json={"name": "Type A"})
    assert resp.status_code == 409


def test_update_class(client, session_with_images):
    sid = session_with_images["id"]
    create_resp = client.post(f"/sessions/{sid}/classes", json={"name": "Old Name"})
    class_id = create_resp.json()["id"]

    resp = client.patch(f"/sessions/{sid}/classes/{class_id}", json={
        "name": "New Name", "color": "#00FF00",
    })
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"
    assert resp.json()["color"] == "#00FF00"


def test_delete_class_reassigns_to_uncategorized(client, session_with_images):
    sid = session_with_images["id"]

    # Create a new class and assign some samples
    new_class = client.post(f"/sessions/{sid}/classes", json={"name": "ToDelete"}).json()
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_ids = [s["id"] for s in samples[:3]]
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": sample_ids, "class_id": new_class["id"],
    })

    # Delete the class
    resp = client.delete(f"/sessions/{sid}/classes/{new_class['id']}")
    assert resp.status_code == 204

    # Samples should be back in Uncategorized
    classes = client.get(f"/sessions/{sid}/classes").json()
    assert len(classes) == 1
    assert classes[0]["name"] == "Uncategorized"
    assert classes[0]["sample_count"] == 10


def test_cannot_delete_uncategorized(client, session_with_images):
    sid = session_with_images["id"]
    classes = client.get(f"/sessions/{sid}/classes").json()
    uncat_id = classes[0]["id"]

    resp = client.delete(f"/sessions/{sid}/classes/{uncat_id}")
    assert resp.status_code == 400


# ── Bulk class assignment ────────────────────────────────────────────────


def test_assign_samples_to_class(client, session_with_images):
    sid = session_with_images["id"]

    # Create a new class
    new_class = client.post(f"/sessions/{sid}/classes", json={
        "name": "Elongated", "color": "#FF0000",
    }).json()

    # Get sample IDs
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    sample_ids = [s["id"] for s in samples[:5]]

    # Assign
    resp = client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": sample_ids, "class_id": new_class["id"],
    })
    assert resp.status_code == 200
    assert resp.json()["updated"] == 5

    # Verify
    classes = client.get(f"/sessions/{sid}/classes").json()
    class_map = {c["name"]: c["sample_count"] for c in classes}
    assert class_map["Elongated"] == 5
    assert class_map["Uncategorized"] == 5


def test_filter_by_class_name(client, session_with_images):
    sid = session_with_images["id"]

    # Create class and assign
    new_class = client.post(f"/sessions/{sid}/classes", json={"name": "Special"}).json()
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [samples[0]["id"]], "class_id": new_class["id"],
    })

    # Filter by class
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "filters": [{"field": "class_name", "operator": "==", "value": "Special"}],
    })
    assert resp.json()["total"] == 1


def test_real_images_gallery(client, real_images_dir):
    """Test gallery with real blood test images if available."""
    session = client.post("/sessions/", json={
        "name": "Blood Gallery", "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    # List samples
    resp = client.post(f"/sessions/{sid}/samples/list", json={"limit": 5})
    data = resp.json()
    assert data["total"] > 0
    assert len(data["items"]) <= 5

    # Get thumbnail for first sample
    sample_id = data["items"][0]["id"]
    resp = client.get(f"/sessions/{sid}/samples/{sample_id}/thumbnail")
    assert resp.status_code == 200


# ── session_has_any_mask aggregate ───────────────────────────────────────


def _wait_for_task(client, task_id: str, timeout_s: float = 10.0) -> dict:
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = client.get(f"/tasks/{task_id}").json()
        if info["status"] in ("completed", "failed", "cancelled"):
            return info
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish within {timeout_s}s")


def _import_circle_images(client, tmp_path) -> str:
    """Create a session, import 5 circle PNGs, return session id once import completes."""
    import numpy as np
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    size = (100, 100)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    yy, xx = np.ogrid[:size[0], :size[1]]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    for i in range(5):
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[circle] = [200, 200, 200]
        Image.fromarray(img).save(img_dir / f"cell_{i:03d}.png")

    sid = client.post("/sessions/", json={"name": "mask-flag"}).json()["id"]
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "channel_grouping": False,
        "preprocessing": None,
    })
    assert resp.status_code == 202, resp.text
    info = _wait_for_task(client, resp.json()["task_id"])
    assert info["status"] == "completed", info
    return sid


def test_samples_page_session_has_any_mask_false_when_no_masks(client, tmp_path):
    """SamplePage.session_has_any_mask is False on a fresh session with no segmentation."""
    sid = _import_circle_images(client, tmp_path)
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "offset": 0, "limit": 100, "sort_by": "filename", "sort_order": "asc",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert body["session_has_any_mask"] is False


def test_samples_page_session_has_any_mask_true_after_segmentation(client, tmp_path):
    """After segmentation creates at least one Mask row, the flag becomes True."""
    sid = _import_circle_images(client, tmp_path)
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })
    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "offset": 0, "limit": 100, "sort_by": "filename", "sort_order": "asc",
    })
    assert resp.status_code == 200
    assert resp.json()["session_has_any_mask"] is True
