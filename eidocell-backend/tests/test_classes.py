import pytest
from PIL import Image


@pytest.fixture()
def session_with_classes(client, tmp_path):
    """Create a session with images and a couple of custom classes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(12):
        img = Image.new("RGB", (80, 80), color=(i * 20, 100, 50))
        img.save(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Classes Test",
        "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Create two custom classes
    cls_a = client.post(f"/sessions/{sid}/classes", json={
        "name": "Type A", "color": "#FF0000",
    }).json()
    cls_b = client.post(f"/sessions/{sid}/classes", json={
        "name": "Type B", "color": "#0000FF",
    }).json()

    # Assign samples: first 4 → A, next 4 → B, rest stay Uncategorized
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    ids_a = [s["id"] for s in samples[:4]]
    ids_b = [s["id"] for s in samples[4:8]]

    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": ids_a, "class_id": cls_a["id"],
    })
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": ids_b, "class_id": cls_b["id"],
    })

    return {
        "session": session,
        "class_a": cls_a,
        "class_b": cls_b,
        "samples": samples,
    }


# ── Class summary ────────────────────────────────────────────────────────


def test_class_summary(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    cls_a = session_with_classes["class_a"]

    resp = client.get(f"/sessions/{sid}/classes/{cls_a['id']}/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Type A"
    assert data["color"] == "#FF0000"
    assert data["sample_count"] == 4


def test_class_summary_uncategorized(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    classes = client.get(f"/sessions/{sid}/classes").json()
    uncat = next(c for c in classes if c["name"] == "Uncategorized")

    resp = client.get(f"/sessions/{sid}/classes/{uncat['id']}/summary")
    assert resp.status_code == 200
    assert resp.json()["sample_count"] == 4  # 12 total - 4 A - 4 B


def test_class_summary_not_found(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    resp = client.get(f"/sessions/{sid}/classes/nonexistent/summary")
    assert resp.status_code == 404


# ── Class samples ────────────────────────────────────────────────────────


def test_list_class_samples(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    cls_b = session_with_classes["class_b"]

    resp = client.get(f"/sessions/{sid}/classes/{cls_b['id']}/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 4
    assert len(data["items"]) == 4
    for item in data["items"]:
        assert item["class_name"] == "Type B"
        assert item["class_color"] == "#0000FF"


def test_list_class_samples_pagination(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    cls_a = session_with_classes["class_a"]

    resp = client.get(
        f"/sessions/{sid}/classes/{cls_a['id']}/samples",
        params={"offset": 2, "limit": 2},
    )
    data = resp.json()
    assert data["total"] == 4
    assert len(data["items"]) == 2
    assert data["offset"] == 2


def test_list_class_samples_empty_class(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    empty = client.post(f"/sessions/{sid}/classes", json={"name": "Empty"}).json()

    resp = client.get(f"/sessions/{sid}/classes/{empty['id']}/samples")
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


# ── Class preview collage ────────────────────────────────────────────────


def test_class_preview(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    cls_a = session_with_classes["class_a"]

    resp = client.get(f"/sessions/{sid}/classes/{cls_a['id']}/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_class_preview_empty_class(client, session_with_classes):
    sid = session_with_classes["session"]["id"]
    empty = client.post(f"/sessions/{sid}/classes", json={"name": "Empty"}).json()

    # Should still return a valid (blank) collage
    resp = client.get(f"/sessions/{sid}/classes/{empty['id']}/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_class_preview_reflects_assignment_change(client, session_with_classes):
    """After reassigning samples, preview should update."""
    sid = session_with_classes["session"]["id"]
    cls_a = session_with_classes["class_a"]
    cls_b = session_with_classes["class_b"]

    # Get preview for class A
    resp1 = client.get(f"/sessions/{sid}/classes/{cls_a['id']}/preview")
    assert resp1.status_code == 200

    # Move one sample from A to B
    samples_a = client.get(f"/sessions/{sid}/classes/{cls_a['id']}/samples").json()["items"]
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [samples_a[0]["id"]], "class_id": cls_b["id"],
    })

    # Get preview again — should regenerate (content may differ)
    resp2 = client.get(f"/sessions/{sid}/classes/{cls_a['id']}/preview")
    assert resp2.status_code == 200

    # Verify class A now has 3 samples
    summary = client.get(f"/sessions/{sid}/classes/{cls_a['id']}/summary").json()
    assert summary["sample_count"] == 3


def test_real_images_class_preview(client, real_images_dir):
    """Test collage with real blood test images if available."""
    session = client.post("/sessions/", json={
        "name": "Blood Classes", "images_directory": real_images_dir,
    }).json()
    sid = session["id"]

    # Get Uncategorized class (all images are there)
    classes = client.get(f"/sessions/{sid}/classes").json()
    uncat = classes[0]

    resp = client.get(f"/sessions/{sid}/classes/{uncat['id']}/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
