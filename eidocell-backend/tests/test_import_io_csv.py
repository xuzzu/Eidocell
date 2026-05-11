from textwrap import dedent

import pytest

from core.image_io.csv_linker import lookup, parse_csv
from core.image_io.errors import CsvLinkError


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(dedent(content).lstrip("\n"))
    return p


def test_parse_csv_extracts_numeric_only(tmp_path):
    csv = _write(tmp_path, "data.csv", """
        image,area,label,intensity
        cell_001.png,12.5,a,0.7
        cell_002.png,33.0,b,1.1
    """)
    rows = parse_csv(csv, filename_col="image")
    assert "cell_001.png" in rows
    assert rows["cell_001.png"] == {"area": 12.5, "intensity": 0.7}
    assert "label" not in rows["cell_001.png"]
    assert rows["cell_001"]["area"] == 12.5  # stem also indexed


def test_parse_csv_missing_filename_col(tmp_path):
    csv = _write(tmp_path, "bad.csv", """
        area
        1.0
    """)
    with pytest.raises(CsvLinkError, match="missing filename column"):
        parse_csv(csv, filename_col="image")


def test_lookup_fallback_to_stem(tmp_path):
    csv = _write(tmp_path, "x.csv", """
        name,v
        a,9.0
    """)
    rows = parse_csv(csv, filename_col="name")
    assert lookup(rows, "a") == {"v": 9.0}
    assert lookup(rows, "a.png") == {"v": 9.0}
    assert lookup(rows, "nope") == {}


def test_parse_csv_skips_nan_inf(tmp_path):
    csv = _write(tmp_path, "n.csv", """
        image,x,y
        a,nan,3.0
        b,inf,5.0
    """)
    rows = parse_csv(csv, filename_col="image")
    assert "x" not in rows["a"]
    assert rows["a"]["y"] == 3.0
    assert "x" not in rows["b"]
    assert rows["b"]["y"] == 5.0
