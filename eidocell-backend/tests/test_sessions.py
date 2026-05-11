import pytest


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_sessions_empty(client):
    resp = client.get("/sessions/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_session(client):
    resp = client.post("/sessions/", json={"name": "Blood Test"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Blood Test"
    assert data["sample_count"] == 0  # empty until import runs
    assert data["scale_factor"] == 1.0
    assert data["scale_units"] == "px"
    assert data["id"]
    assert data["session_folder"]
    assert data.get("images_directory") in (None, "")


def test_get_session(client):
    create_resp = client.post("/sessions/", json={"name": "Test Session"})
    session_id = create_resp.json()["id"]

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Session"
    assert data["sample_count"] == 0


def test_get_session_not_found(client):
    resp = client.get("/sessions/nonexistent")
    assert resp.status_code == 404


def test_update_session(client):
    create_resp = client.post("/sessions/", json={"name": "Original Name"})
    session_id = create_resp.json()["id"]

    resp = client.patch(f"/sessions/{session_id}", json={
        "name": "Renamed Session",
        "scale_factor": 0.5,
        "scale_units": "um",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Renamed Session"
    assert data["scale_factor"] == 0.5
    assert data["scale_units"] == "um"


def test_delete_session(client):
    create_resp = client.post("/sessions/", json={"name": "To Delete"})
    session_id = create_resp.json()["id"]

    resp = client.delete(f"/sessions/{session_id}")
    assert resp.status_code == 204

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 404


def test_list_sessions_after_create(client):
    client.post("/sessions/", json={"name": "Session A"})
    client.post("/sessions/", json={"name": "Session B"})

    resp = client.get("/sessions/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = [s["name"] for s in data]
    assert "Session A" in names
    assert "Session B" in names
