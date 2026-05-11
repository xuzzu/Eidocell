"""Validate the CIF/RIF parser using the example files in example-datasets.

The fixtures are real IDEAS exports symlinked from ``ifc-data-eval/test.{cif,rif}``;
we just need their existence and that we extract at least one well-formed image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.image_io.cif_rif import (
    _decompress_30817,
    _decompress_30818,
    count_objects,
    iter_objects,
    split_channels,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CIF_PATH = REPO_ROOT / "example-datasets" / "cif_example"
RIF_PATH = REPO_ROOT / "example-datasets" / "rif_example"
TEST_CHUNK = REPO_ROOT / "ifc-data-eval" / "test_chunk.bin"


def test_decompress_30818_round_trips_simple_run():
    # 4 pixels of 255 then 2 pixels of 0 → matches the docstring example.
    raw = np.array([255, 3, 0, 1], dtype=np.uint8)
    out = _decompress_30818(raw, 3, 2)
    # Reshape for inspection.
    assert out.shape == (2, 3)
    # Values should be the encoded 255s and 0s in the expected order.
    flat = out.T.ravel()
    assert flat.tolist() == [255, 255, 255, 255, 0, 0]


@pytest.mark.skipif(not TEST_CHUNK.is_file(), reason="test_chunk.bin not available")
def test_decompress_30817_known_chunk_shape():
    raw = np.frombuffer(TEST_CHUNK.read_bytes(), dtype=np.uint8).copy()
    img = _decompress_30817(raw, 516, 94)
    assert img.shape == (94, 516)
    # Multi-channel layout: split into 6 tiles.
    chans = split_channels(img, 6)
    assert len(chans) == 6
    assert all(c.shape == (94, 86) for c in chans)


@pytest.mark.skipif(not CIF_PATH.exists(), reason="cif_example not available")
def test_iter_objects_cif_yields_images():
    n = count_objects(CIF_PATH)
    assert n > 0
    # Pull the first object only — full enumeration could be slow.
    first = next(iter_objects(CIF_PATH))
    assert first["image"].ndim == 2
    assert first["image"].size > 0
    assert first["compression"] == 30817


@pytest.mark.skipif(not CIF_PATH.exists(), reason="cif_example not available")
def test_iter_objects_cif_pairs_masks():
    """IDEAS CIF interleaves image (30817) + mask (30818) IFDs per object.

    Image-IFD records should expose the paired mask as a binary (0/1) array of
    matching width/height, and ``count_objects`` should count image IFDs only.
    """
    first = next(iter_objects(CIF_PATH))
    assert first["mask"] is not None, "expected paired IDEAS mask on first object"
    assert first["mask"].shape == first["image"].shape
    assert set(np.unique(first["mask"]).tolist()) <= {0, 1}


@pytest.mark.skipif(not RIF_PATH.exists(), reason="rif_example not available")
def test_iter_objects_rif_yields_images():
    first = next(iter_objects(RIF_PATH))
    assert first["image"].ndim == 2
    assert first["image"].size > 0
