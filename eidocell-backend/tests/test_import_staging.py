"""Round-trip test for the import_staging Lance helper."""
from __future__ import annotations


import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_eidocell_home(tmp_path, monkeypatch):
    """Point EIDOCELL_HOME at a tmp dir so each test gets a fresh Lance dir."""
    monkeypatch.setenv("EIDOCELL_HOME", str(tmp_path))
    # Reload the lance module to pick up the new LANCE_DIR.
    import importlib

    from core import config as config_mod
    importlib.reload(config_mod)
    from core.storage import lance as lance_mod
    importlib.reload(lance_mod)
    from core.storage import import_staging as staging_mod
    importlib.reload(staging_mod)
    yield


def test_round_trip_single_channel():
    from core.storage import import_staging

    sid = "sess1"
    iid = "imp1"
    arr = np.random.randint(0, 255, (32, 64), dtype=np.uint8)
    n = import_staging.upsert_rows(sid, iid, [{
        "tmp_id": "t1",
        "group_stem": "blob",
        "source_paths": ["blob.png"],
        "channel_meta": {"order": ["BF"]},
        "array": arr,
        "parsed_attrs": {"area": 42.0},
    }])
    assert n == 1
    assert import_staging.count(sid, iid) == 1

    rows = list(import_staging.iter_rows(sid, iid))
    assert len(rows) == 1
    r = rows[0]
    assert r["tmp_id"] == "t1"
    assert r["group_stem"] == "blob"
    assert r["channel_meta"] == {"order": ["BF"]}
    assert r["parsed_attrs"] == {"area": 42.0}
    assert r["array"].shape == arr.shape
    assert (r["array"] == arr).all()


def test_round_trip_multi_channel_and_summary():
    from core.storage import import_staging

    sid = "sess2"
    iid = "imp2"
    arr_small = np.zeros((20, 30, 2), dtype=np.uint16) + 100
    arr_big = np.zeros((40, 50, 2), dtype=np.uint16) + 200
    import_staging.upsert_rows(sid, iid, [
        {"tmp_id": "a", "group_stem": "g1", "source_paths": ["a_ch1.png", "a_ch2.png"], "array": arr_small},
        {"tmp_id": "b", "group_stem": "g2", "source_paths": ["b_ch1.png", "b_ch2.png"], "array": arr_big},
    ])
    s = import_staging.shape_summary(sid, iid)
    assert s["count"] == 2
    assert s["min_h"] == 20 and s["max_h"] == 40 and s["mean_h"] == 30
    assert s["min_w"] == 30 and s["max_w"] == 50 and s["mean_w"] == 40
    # Longest-side stats: per-image max(H, W).
    # arr_small is 20×30 → longest=30; arr_big is 40×50 → longest=50.
    assert s["min_longest"] == 30
    assert s["max_longest"] == 50
    assert s["mean_longest"] == 40
    assert s["n_channels"] == 2

    import_staging.drop(sid, iid)
    assert import_staging.count(sid, iid) == 0
