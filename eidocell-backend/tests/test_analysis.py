import numpy as np
import pytest
from PIL import Image


def _create_test_image(path, size=(100, 100), brightness=200):
    """Create image with a circle for segmentation."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [brightness, brightness, brightness]
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_with_masks(client, tmp_path):
    """Create a session, run segmentation so we have mask attributes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Varying brightness to get different intensity attributes
    for i in range(10):
        _create_test_image(img_dir / f"cell_{i:03d}.png", brightness=50 + i * 20)

    session = client.post("/sessions/", json={
        "name": "Analysis Test",
        "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    # Run segmentation to generate masks + attributes
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    return session


# ── Available parameters ────────────────────────────────────────────────


def test_list_parameters(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.get(f"/sessions/{sid}/analysis/parameters")
    assert resp.status_code == 200
    params = resp.json()
    assert "area" in params
    assert "mean_intensity" in params
    assert "solidity" in params


def test_list_parameters_no_masks(client, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (32, 32)).save(img_dir / "x.png")
    session = client.post("/sessions/", json={
        "name": "No Masks", "images_directory": str(img_dir),
    }).json()
    resp = client.get(f"/sessions/{session['id']}/analysis/parameters")
    assert resp.status_code == 200
    assert resp.json() == []


# ── Plot CRUD ───────────────────────────────────────────────────────────


def test_create_histogram_plot(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram",
        "parameters": {"x_variable": "area", "num_bins": 20},
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["chart_type"] == "histogram"
    assert "area" in data["name"]
    assert data["parameters"]["x_variable"] == "area"


def test_create_scatter_plot(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "mean_intensity"},
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["chart_type"] == "scatter"
    assert "area" in data["name"] and "mean_intensity" in data["name"]


def test_create_scatter_plot_with_color(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {
            "x_variable": "area", "y_variable": "perimeter",
            "color_variable": "class",
        },
    })
    assert resp.status_code == 201


def test_list_plots(client, session_with_masks):
    sid = session_with_masks["id"]
    client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    })
    client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter", "parameters": {"x_variable": "area", "y_variable": "perimeter"},
    })
    resp = client.get(f"/sessions/{sid}/analysis/plots")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_single_plot(client, session_with_masks):
    sid = session_with_masks["id"]
    created = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    resp = client.get(f"/sessions/{sid}/analysis/plots/{created['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == created["id"]
    assert data["chart_type"] == "histogram"
    assert data["parameters"]["x_variable"] == "area"


def test_get_single_plot_not_found(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.get(f"/sessions/{sid}/analysis/plots/nonexistent")
    assert resp.status_code == 404


def test_delete_plot(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()
    resp = client.delete(f"/sessions/{sid}/analysis/plots/{plot['id']}")
    assert resp.status_code == 204
    assert len(client.get(f"/sessions/{sid}/analysis/plots").json()) == 0


# ── Plot data ───────────────────────────────────────────────────────────


def test_get_histogram_data(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    resp = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chart_type"] == "histogram"
    assert len(data["data"]) > 0
    for point in data["data"]:
        assert "area" in point["values"]
        assert point["sample_id"]


def test_get_scatter_data(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "mean_intensity"},
    }).json()

    resp = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) > 0
    for point in data["data"]:
        assert "area" in point["values"]
        assert "mean_intensity" in point["values"]


# ── Gate CRUD ───────────────────────────────────────────────────────────


def test_create_interval_gate(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    # Get data to know the value range
    data = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data").json()
    areas = [p["values"]["area"] for p in data["data"]]
    mid = (min(areas) + max(areas)) / 2

    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval",
        "definition": {"min": min(areas) - 1, "max": mid},
        "parameters": ["area"],
        "color": "#FF0000",
    })
    assert resp.status_code == 201
    gate = resp.json()
    assert gate["gate_type"] == "interval"
    assert gate["sample_count"] > 0
    assert gate["percentage"] > 0


def test_create_rectangular_gate(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "mean_intensity"},
    }).json()

    data = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data").json()
    areas = [p["values"]["area"] for p in data["data"]]
    intensities = [p["values"]["mean_intensity"] for p in data["data"]]

    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "rectangular",
        "definition": {
            "x": min(areas) - 1, "y": min(intensities) - 1,
            "width": max(areas) - min(areas) + 2,
            "height": max(intensities) - min(intensities) + 2,
        },
        "parameters": ["area", "mean_intensity"],
    })
    assert resp.status_code == 201
    gate = resp.json()
    # Gate covers all data, so all samples should be inside
    assert gate["sample_count"] == len(data["data"])


def test_create_polygon_gate(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "perimeter"},
    }).json()

    data = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data").json()
    areas = [p["values"]["area"] for p in data["data"]]
    perimeters = [p["values"]["perimeter"] for p in data["data"]]

    # Large polygon covering all data
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "polygon",
        "definition": {
            "vertices": [
                [min(areas) - 10, min(perimeters) - 10],
                [max(areas) + 10, min(perimeters) - 10],
                [max(areas) + 10, max(perimeters) + 10],
                [min(areas) - 10, max(perimeters) + 10],
            ],
        },
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 201
    assert resp.json()["sample_count"] == len(data["data"])


def test_list_gates(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 99999},
        "parameters": ["area"],
    })
    client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 1},
        "parameters": ["area"],
    })

    # List per plot
    resp = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates")
    assert len(resp.json()) == 2

    # List all
    resp = client.get(f"/sessions/{sid}/analysis/gates")
    assert len(resp.json()) == 2


def test_update_gate(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 99999},
        "parameters": ["area"], "name": "Original",
    }).json()

    resp = client.patch(f"/sessions/{sid}/analysis/gates/{gate['id']}", json={
        "name": "Renamed Gate", "color": "#00FF00",
    })
    assert resp.status_code == 200
    assert resp.json()["name"] == "Renamed Gate"
    assert resp.json()["color"] == "#00FF00"


def test_delete_gate(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 99999},
        "parameters": ["area"],
    }).json()

    resp = client.delete(f"/sessions/{sid}/analysis/gates/{gate['id']}")
    assert resp.status_code == 204

    assert len(client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates").json()) == 0


def test_gate_population(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 99999},
        "parameters": ["area"], "is_active": False,
    }).json()

    resp = client.get(f"/sessions/{sid}/analysis/gates/{gate['id']}/population")
    assert resp.status_code == 200
    sample_ids = resp.json()
    assert len(sample_ids) == gate["sample_count"]


# ── Active samples filtering ───────────────────────────────────────────


def test_active_samples_default(client, session_with_masks):
    sid = session_with_masks["id"]
    resp = client.get(f"/sessions/{sid}/analysis/active-samples")
    assert resp.status_code == 200
    assert len(resp.json()) == 10  # all active


def test_gate_filters_active_samples(client, session_with_masks):
    sid = session_with_masks["id"]

    # Get plot data to find value range
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "mean_intensity"},
    }).json()
    data = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data").json()
    intensities = sorted(p["values"]["mean_intensity"] for p in data["data"])

    # Create an active gate that only includes samples with intensity above median
    median_val = intensities[len(intensities) // 2]
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval",
        "definition": {"min": median_val, "max": max(intensities) + 1},
        "parameters": ["mean_intensity"],
        "is_active": True,
    }).json()

    # Active samples should now be filtered
    active = client.get(f"/sessions/{sid}/analysis/active-samples").json()
    assert len(active) < 10
    assert len(active) == gate["sample_count"]


def test_reset_gates_reactivates_all(client, session_with_masks):
    sid = session_with_masks["id"]

    # Create a restrictive gate
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()
    client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": -1, "max": 0},
        "parameters": ["area"], "is_active": True,
    })

    # Should have 0 active samples (no area is negative)
    active = client.get(f"/sessions/{sid}/analysis/active-samples").json()
    assert len(active) == 0

    # Reset
    resp = client.post(f"/sessions/{sid}/analysis/reset-gates")
    assert resp.status_code == 200
    assert resp.json()["reactivated"] == 10

    active = client.get(f"/sessions/{sid}/analysis/active-samples").json()
    assert len(active) == 10


def test_inactive_gate_doesnt_filter(client, session_with_masks):
    sid = session_with_masks["id"]

    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()

    # Create a gate but set it inactive
    client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": -1, "max": 0},
        "parameters": ["area"], "is_active": False,
    })

    # All samples should still be active
    active = client.get(f"/sessions/{sid}/analysis/active-samples").json()
    assert len(active) == 10


def test_multiple_active_gates_union(client, session_with_masks):
    """Multiple active gates use OR logic: samples in ANY gate are active."""
    sid = session_with_masks["id"]

    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "mean_intensity"},
    }).json()

    # Get data ranges
    data = client.get(f"/sessions/{sid}/analysis/plots/{plot['id']}/data").json()
    intensities = sorted(p["values"]["mean_intensity"] for p in data["data"])
    mid = intensities[len(intensities) // 2]

    # Gate 1: lower half of intensities
    gate1 = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval",
        "definition": {"min": min(intensities) - 1, "max": mid},
        "parameters": ["mean_intensity"], "is_active": True,
    }).json()

    active_after_one = client.get(f"/sessions/{sid}/analysis/active-samples").json()
    assert len(active_after_one) == gate1["sample_count"]
    assert len(active_after_one) < 10  # not all samples

    # Gate 2: upper half of intensities
    gate2 = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval",
        "definition": {"min": mid, "max": max(intensities) + 1},
        "parameters": ["mean_intensity"], "is_active": True,
    }).json()

    # Union of both gates should cover all (or nearly all) samples
    active_after_two = client.get(f"/sessions/{sid}/analysis/active-samples").json()
    assert len(active_after_two) >= len(active_after_one)
    assert len(active_after_two) >= gate2["sample_count"]


def test_delete_plot_deletes_gates(client, session_with_masks):
    sid = session_with_masks["id"]

    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram", "parameters": {"x_variable": "area"},
    }).json()
    client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 99999},
        "parameters": ["area"],
    })

    client.delete(f"/sessions/{sid}/analysis/plots/{plot['id']}")

    # All gates should be gone
    assert len(client.get(f"/sessions/{sid}/analysis/gates").json()) == 0


# ── Gate validation ───────────────────────────────────────────────────


def _make_plot(client, sid):
    return client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "perimeter"},
    }).json()


def test_interval_gate_missing_min(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"max": 100},
        "parameters": ["area"],
    })
    assert resp.status_code == 422


def test_interval_gate_min_gt_max(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 100, "max": 10},
        "parameters": ["area"],
    })
    assert resp.status_code == 422


def test_interval_gate_wrong_param_count(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 100},
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 422


def test_rectangular_gate_missing_key(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "rectangular", "definition": {"x": 0, "y": 0, "width": 10},
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 422


def test_rectangular_gate_negative_size(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "rectangular",
        "definition": {"x": 0, "y": 0, "width": -10, "height": 10},
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 422


def test_polygon_gate_too_few_vertices(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "polygon",
        "definition": {"vertices": [[0, 0], [1, 1]]},
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 422


def test_polygon_gate_bad_vertex(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "polygon",
        "definition": {"vertices": [[0, 0], [1], [2, 2]]},
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 422


def test_unknown_gate_type(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    resp = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "elliptical",
        "definition": {"cx": 0, "cy": 0, "rx": 10, "ry": 5},
        "parameters": ["area", "perimeter"],
    })
    assert resp.status_code == 422


def test_update_gate_validates_definition(client, session_with_masks):
    sid = session_with_masks["id"]
    plot = _make_plot(client, sid)
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0, "max": 100},
        "parameters": ["area"],
    }).json()

    resp = client.patch(f"/sessions/{sid}/analysis/gates/{gate['id']}", json={
        "definition": {"min": 200, "max": 100},
    })
    assert resp.status_code == 400
