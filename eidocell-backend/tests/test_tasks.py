"""Tests for background task infrastructure and async endpoints."""

import time
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from core.task_manager import TaskManager, TaskStatus


def _create_test_image(path: Path, size=(100, 100)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    cy, cx = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = [200, 200, 200]
    Image.fromarray(img).save(path)


@pytest.fixture()
def session_for_tasks(client, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        _create_test_image(img_dir / f"cell_{i:03d}.png")

    session = client.post("/sessions/", json={
        "name": "Task Test",
        "images_directory": str(img_dir),
    }).json()
    return session


# ── TaskManager unit tests ─────────────────────────────────────────────


def test_task_manager_submit_and_complete():
    tm = TaskManager()

    def dummy_task(*, on_progress, **_):
        on_progress(1, 2, "half")
        on_progress(2, 2, "done")
        return {"result": "ok"}

    task_id = tm.submit("test", dummy_task)
    # Wait for completion
    for _ in range(50):
        task = tm.get_task(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            break
        time.sleep(0.05)

    task = tm.get_task(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert task.result == {"result": "ok"}
    assert task.progress == 2
    assert task.total == 2


def test_task_manager_failure():
    tm = TaskManager()

    def failing_task(*, on_progress, **_):
        raise ValueError("Something went wrong")

    task_id = tm.submit("fail test", failing_task)
    for _ in range(50):
        task = tm.get_task(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            break
        time.sleep(0.05)

    task = tm.get_task(task_id)
    assert task.status == TaskStatus.FAILED
    assert "Something went wrong" in task.error


def test_task_manager_list():
    tm = TaskManager()

    def noop(*, on_progress, **_):
        return None

    tm.submit("task1", noop)
    tm.submit("task2", noop)
    time.sleep(0.2)

    tasks = tm.list_tasks()
    assert len(tasks) == 2


def test_task_manager_cleanup():
    tm = TaskManager()

    def noop(*, on_progress, **_):
        return None

    for i in range(5):
        tm.submit(f"task{i}", noop)
    time.sleep(0.3)

    removed = tm.cleanup_completed(keep_last=2)
    assert removed == 3
    assert len(tm.list_tasks()) == 2


def test_task_to_dict():
    tm = TaskManager()

    def noop(*, on_progress, **_):
        on_progress(5, 10, "halfway")
        return {"x": 1}

    task_id = tm.submit("dict test", noop)
    time.sleep(0.3)

    task = tm.get_task(task_id)
    d = task.to_dict()
    assert d["id"] == task_id
    assert d["name"] == "dict test"
    assert d["status"] == "completed"
    assert d["percentage"] == 50.0
    assert d["result"] == {"x": 1}


# ── Tasks API ──────────────────────────────────────────────────────────


def test_list_tasks_empty(client):
    # Note: task manager is global singleton, may have tasks from other tests
    resp = client.get("/tasks/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_task_not_found(client):
    resp = client.get("/tasks/nonexistent")
    assert resp.status_code == 404


def test_cleanup_tasks(client, session_for_tasks):
    """POST /tasks/cleanup removes completed tasks."""
    sid = session_for_tasks["id"]

    # Run a quick sync segmentation to generate a completed async task
    resp = client.post(f"/sessions/{sid}/segmentation/run-async", json={
        "method": "otsu_intensity",
    })
    task_id = resp.json()["task_id"]

    # Wait for completion
    for _ in range(100):
        data = client.get(f"/tasks/{task_id}").json()
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    resp = client.post("/tasks/cleanup")
    assert resp.status_code == 200
    assert resp.json()["removed"] >= 0


# ── Async segmentation ────────────────────────────────────────────────


def test_segmentation_async(client, session_for_tasks):
    sid = session_for_tasks["id"]

    # Start async segmentation
    resp = client.post(f"/sessions/{sid}/segmentation/run-async", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    # Poll until complete
    for _ in range(100):
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert data["status"] == "completed"
    assert data["result"]["processed"] == 5
    assert data["result"]["total"] == 5

    # Verify masks were actually created
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    attrs = client.get(f"/sessions/{sid}/samples/{samples[0]['id']}/mask/attributes").json()
    assert "area" in attrs


# ── Async feature extraction ─────────────────────────────────────────


def test_feature_extraction_async(client, session_for_tasks):
    sid = session_for_tasks["id"]

    # Run segmentation first (sync is fine for test)
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    # Start async feature extraction
    resp = client.post(f"/sessions/{sid}/features/extract", json={
        "method": "morphological",
    })
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    # Poll until complete
    for _ in range(100):
        resp = client.get(f"/tasks/{task_id}")
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert data["status"] == "completed"
    assert data["result"]["processed"] == 5

    # Verify features file exists
    features_path = Path(session_for_tasks["session_folder"]) / "features" / "session_features.npy"
    assert features_path.exists()


def test_task_progress_tracking(client, session_for_tasks):
    sid = session_for_tasks["id"]

    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
    })

    resp = client.post(f"/sessions/{sid}/features/extract", json={})
    task_id = resp.json()["task_id"]

    # Wait for completion and check final progress
    for _ in range(100):
        resp = client.get(f"/tasks/{task_id}")
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert data["progress"] == data["total"]
    assert data["percentage"] == 100.0
