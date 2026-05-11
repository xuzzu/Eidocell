"""Unit tests for the preprocessing operations."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.preprocessing import ops, build_pipeline


def test_zscore_grayscale_unit_variance():
    img = (np.random.randn(64, 64) * 5 + 100).astype(np.float32)
    out, meta = ops.normalize_zscore(img, per_channel=True)
    assert abs(float(out.mean())) < 1e-3
    assert abs(float(out.std()) - 1.0) < 1e-2
    assert meta["per_channel"] is True


def test_zscore_per_channel():
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[..., 0] = np.random.randn(10, 10) * 2 + 5
    img[..., 1] = np.random.randn(10, 10) * 7 + 50
    out, _ = ops.normalize_zscore(img, per_channel=True)
    assert abs(float(out[..., 0].std()) - 1.0) < 0.05
    assert abs(float(out[..., 1].std()) - 1.0) < 0.05


def test_minmax_clamps_to_unit_range():
    img = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    out, meta = ops.normalize_minmax(img, per_channel=False)
    assert float(out.min()) == 0.0
    assert float(out.max()) == pytest.approx(1.0)
    assert meta["stats"][0]["min"] == 0.0


def test_pad_constant_centres_input():
    img = np.ones((4, 6), dtype=np.uint8) * 9
    out, meta = ops.pad_to(img, (8, 10), method="constant", constant_value=0)
    assert out.shape == (8, 10)
    assert meta["pad"] == [2, 2, 2, 2]
    # Centre block preserved.
    assert (out[2:6, 2:8] == 9).all()
    assert out[0, 0] == 0


def test_pad_replicate_keeps_input_block():
    img = np.full((4, 4), 7, dtype=np.uint8)
    out, _ = ops.pad_to(img, (8, 8), method="replicate")
    assert out.shape == (8, 8)
    assert (out == 7).all()  # replicate of constant input is constant


def test_pad_poisson_preserves_centre():
    img = np.full((4, 4), 50, dtype=np.uint8)
    out, _ = ops.pad_to(img, (8, 8), method="poisson")
    assert out.shape == (8, 8)
    # Centre block preserved.
    assert (out[2:6, 2:6] == 50).all()


def test_resize_changes_shape_only():
    img = np.zeros((10, 10), dtype=np.uint8)
    out, _ = ops.resize_to(img, (20, 30))
    assert out.shape == (20, 30)


def test_align_channels_recovers_shift():
    """Synthetic shift across two channels — alignment must reduce centre-MAE."""
    rng = np.random.default_rng(0)
    h, w = 64, 64
    ref = rng.normal(loc=128, scale=20, size=(h, w)).astype(np.float32)
    # cv2.warpAffine([[1,0,tx],[0,1,ty]]) shifts the image by (tx,ty).
    M = np.float32([[1, 0, 5.0], [0, 1, -3.0]])
    target = cv2.warpAffine(ref, M, (w, h))
    img = np.stack([ref, target], axis=-1)

    # Pre-alignment MAE in the centre patch (excludes border noise).
    centre_ref = ref[10:54, 10:54]
    pre = float(np.mean(np.abs(centre_ref - target[10:54, 10:54])))

    out, meta = ops.align_channels(img, reference_index=0, upsample_factor=10)
    dy, dx = meta["shifts"][1]
    # Detect a non-trivial shift was found.
    assert abs(dy) + abs(dx) >= 1.0

    post = float(np.mean(np.abs(centre_ref - out[10:54, 10:54, 1].astype(np.float32))))
    assert post < pre / 2  # alignment substantially reduces error


def test_align_channels_skips_single_channel():
    img = np.zeros((10, 10), dtype=np.float32)
    out, meta = ops.align_channels(img)
    assert meta["skipped"] is True
    assert out.shape == img.shape


def test_compensate_identity_is_noop():
    img = np.random.rand(8, 8, 3).astype(np.float32)
    out, _ = ops.compensate(img, np.eye(3))
    assert np.allclose(out, img, atol=1e-5)


def test_compensate_swaps_channels():
    img = np.zeros((4, 4, 2), dtype=np.float32)
    img[..., 0] = 1.0
    M = np.array([[0, 1], [1, 0]], dtype=np.float32)
    out, _ = ops.compensate(img, M)
    assert (out[..., 0] == 0.0).all()
    assert (out[..., 1] == 1.0).all()


# ── Pipeline integration ───────────────────────────────────────────────


def test_build_pipeline_skip_target_does_no_padding():
    p = build_pipeline({"target_shape_strategy": "none"})
    img = np.zeros((10, 12), dtype=np.uint8)
    out, meta = p.apply(img)
    assert out.shape == img.shape
    assert meta["steps"] == []


def test_build_pipeline_with_explicit_pad_and_zscore():
    p = build_pipeline({
        "target_shape_strategy": "explicit",
        "explicit_shape": [16, 16],
        "padding_method": "constant",
        "normalize": "zscore",
    })
    img = (np.random.randn(8, 8) * 4 + 50).astype(np.float32)
    out, meta = p.apply(img)
    assert out.shape == (16, 16)
    step_names = [s["step"] for s in meta["steps"]]
    assert step_names == ["pad_to", "normalize_zscore"]


def test_build_pipeline_mean_strategy_uses_summary():
    summary = {"count": 3, "min_h": 8, "max_h": 24, "mean_h": 16,
               "min_w": 10, "max_w": 26, "mean_w": 18, "n_channels": 1}
    p = build_pipeline(
        {"target_shape_strategy": "mean", "padding_method": "constant"},
        shape_summary=summary,
    )
    img = np.zeros((10, 12), dtype=np.uint8)
    out, _ = p.apply(img)
    assert out.shape == (16, 18)
