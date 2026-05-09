import csv
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _create_test_image(path: Path, size=(100, 100)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [200, 200, 200]
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_with_data(client, tmp_path):
    """Session with segmentation, classes, and clusters."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Export Test",
        "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Run segmentation
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    # Extract features
    client.post(f"/sessions/{sid}/features/extract", json={})

    # Create a class and assign some samples
    cls = client.post(f"/sessions/{sid}/classes", json={
        "name": "TypeA", "color": "#FF0000",
    }).json()
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    client.post(f"/sessions/{sid}/samples/assign-class", json={
        "sample_ids": [s["id"] for s in samples[:3]],
        "class_id": cls["id"],
    })

    # Run clustering on the extracted morphological features
    client.post(f"/sessions/{sid}/clusters/run", json={
        "n_clusters": 2, "feature_method": "morphological",
    })

    return session


# ── Export all ─────────────────────────────────────────────────────────


def test_export_all(client, session_with_data, tmp_path):
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_output"

    resp = client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["classes_exported"] == 10
    assert data["clusters_exported"] > 0
    assert data["masks_exported"] == 10
    assert data["csv_rows"] == 10

    # Verify directory structure
    assert (export_dir / "classes").is_dir()
    assert (export_dir / "clusters").is_dir()
    assert (export_dir / "masks").is_dir()
    assert (export_dir / "parameters.csv").is_file()


def test_export_classes_structure(client, session_with_data, tmp_path):
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_classes"

    client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
        "include_clusters": False,
        "include_masks": False,
        "include_csv": False,
    })

    classes_dir = export_dir / "classes"
    assert classes_dir.is_dir()
    # Should have TypeA and Uncategorized folders
    subdirs = [d.name for d in classes_dir.iterdir() if d.is_dir()]
    assert "TypeA" in subdirs
    assert "Uncategorized" in subdirs

    # TypeA should have 3 images
    type_a_files = list((classes_dir / "TypeA").iterdir())
    assert len(type_a_files) == 3


def test_export_clusters_structure(client, session_with_data, tmp_path):
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_clusters"

    client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
        "include_classes": False,
        "include_masks": False,
        "include_csv": False,
    })

    clusters_dir = export_dir / "clusters"
    assert clusters_dir.is_dir()
    cluster_folders = list(clusters_dir.iterdir())
    # K-Means may collapse identical synthetic images into fewer clusters
    assert len(cluster_folders) >= 1


def test_export_masks(client, session_with_data, tmp_path):
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_masks"

    client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
        "include_classes": False,
        "include_clusters": False,
        "include_csv": False,
    })

    masks_dir = export_dir / "masks"
    assert masks_dir.is_dir()
    mask_files = list(masks_dir.glob("*_mask.png"))
    assert len(mask_files) == 10


def test_export_csv_content(client, session_with_data, tmp_path):
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_csv"

    client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
        "include_classes": False,
        "include_clusters": False,
        "include_masks": False,
    })

    csv_path = export_dir / "parameters.csv"
    assert csv_path.is_file()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 10
    # Check required columns
    assert "sample_id" in rows[0]
    assert "filename" in rows[0]
    assert "class_name" in rows[0]
    assert "cluster_ids" in rows[0]
    # Check mask attribute columns
    assert "area" in rows[0]
    assert "perimeter" in rows[0]
    assert "solidity" in rows[0]


def test_export_csv_class_names(client, session_with_data, tmp_path):
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_csv_classes"

    client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
        "include_classes": False,
        "include_clusters": False,
        "include_masks": False,
    })

    with open(export_dir / "parameters.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    class_names = [r["class_name"] for r in rows]
    assert class_names.count("TypeA") == 3
    assert class_names.count("Uncategorized") == 7


def test_export_selective(client, session_with_data, tmp_path):
    """Export only CSV, nothing else."""
    sid = session_with_data["id"]
    export_dir = tmp_path / "export_selective"

    resp = client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
        "include_classes": False,
        "include_clusters": False,
        "include_masks": False,
        "include_csv": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["classes_exported"] == 0
    assert data["clusters_exported"] == 0
    assert data["masks_exported"] == 0
    assert data["csv_rows"] == 10

    assert not (export_dir / "classes").exists()
    assert not (export_dir / "clusters").exists()
    assert not (export_dir / "masks").exists()
    assert (export_dir / "parameters.csv").is_file()


def test_export_no_clusters(client, tmp_path):
    """Export a session that has no clusters."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "No Clusters", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]
    export_dir = tmp_path / "export_no_clusters"

    resp = client.post(f"/sessions/{sid}/export/", json={
        "output_directory": str(export_dir),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters_exported"] == 0
    assert data["csv_rows"] == 3


def test_export_bad_directory(client, session_with_data):
    sid = session_with_data["id"]
    resp = client.post(f"/sessions/{sid}/export/", json={
        "output_directory": "/nonexistent/path/export",
    })
    assert resp.status_code == 400
