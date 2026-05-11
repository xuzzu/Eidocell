from pathlib import Path

from core.image_io.grouping import group_paths, parse_filename


def test_parse_filename_no_suffix():
    assert parse_filename("blob.tif") == ("blob", None)


def test_parse_filename_ch1():
    assert parse_filename("cell001_ch1.png") == ("cell001", 1)


def test_parse_filename_ch01_uppercase():
    assert parse_filename("CELL_007_CH02.tif") == ("CELL_007", 2)


def test_parse_filename_short_form():
    assert parse_filename("cell_005_c3.jpeg") == ("cell_005", 3)


def test_group_paths_disabled_yields_singletons():
    paths = [Path(p) for p in ("a_ch1.png", "a_ch2.png", "b.png")]
    g = group_paths(paths, detect_channels=False)
    assert len(g) == 3
    assert all(len(x["paths"]) == 1 for x in g)


def test_group_paths_enabled_groups_by_stem():
    paths = [Path(p) for p in ("a_ch2.png", "a_ch1.png", "b.png")]
    g = group_paths(paths, detect_channels=True)
    assert len(g) == 2
    by_stem = {x["group_stem"]: x for x in g}
    assert [p.name for p in by_stem["a"]["paths"]] == ["a_ch1.png", "a_ch2.png"]
    assert by_stem["a"]["channel_indices"] == [0, 1]
    assert by_stem["b"]["channel_indices"] == [0]


def test_group_paths_singletons_keep_zero_index():
    g = group_paths([Path("solo.tif")], detect_channels=True)
    assert g[0]["channel_indices"] == [0]
