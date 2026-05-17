"""Unit tests for the preprocessing operations."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.preprocessing import ops, build_pipeline


def test_percentile_clip_and_rescale():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)  # 0..99
    out, meta = ops.normalize_percentile(img, low=1.0, high=99.0, per_channel=False)
    # Values below p1 / above p99 saturate; range is [0, 1].
    assert float(out.min()) == pytest.approx(0.0)
    assert float(out.max()) == pytest.approx(1.0)
    assert meta["stats"][0]["p_low"] == pytest.approx(np.percentile(img, 1))
    assert meta["stats"][0]["p_high"] == pytest.approx(np.percentile(img, 99))


def test_percentile_per_channel_independent_scales():
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[..., 0] = np.arange(100).reshape(10, 10)            # 0..99
    img[..., 1] = np.arange(100).reshape(10, 10) * 100.0    # 0..9900
    out, _ = ops.normalize_percentile(img, per_channel=True)
    # Both channels rescaled into the same [0, 1] band despite 100× scale gap.
    assert float(out[..., 0].max()) == pytest.approx(1.0)
    assert float(out[..., 1].max()) == pytest.approx(1.0)


def test_percentile_valid_mask_excludes_padded_margin():
    """Pixels outside ``valid`` shouldn't drag the low/high percentiles."""
    img = np.zeros((20, 20), dtype=np.float32)
    img[5:15, 5:15] = np.arange(100, dtype=np.float32).reshape(10, 10) + 50.0
    valid = np.zeros_like(img, dtype=np.uint8)
    valid[5:15, 5:15] = 1
    out_with, _ = ops.normalize_percentile(img, valid=valid, per_channel=False)
    out_without, _ = ops.normalize_percentile(img, per_channel=False)
    # With valid mask: the 0-padded margin pixels saturate at 0 but don't define
    # the low end. Without: the 0-padded pixels ARE the low end, compressing
    # the interior into a smaller fraction of the range.
    interior_with = out_with[5:15, 5:15]
    interior_without = out_without[5:15, 5:15]
    assert float(interior_with.max() - interior_with.min()) > \
           float(interior_without.max() - interior_without.min())


def test_bg_subtract_floor_and_high():
    img = np.linspace(10.0, 110.0, 100, dtype=np.float32).reshape(10, 10)
    out, meta = ops.normalize_bg_subtract(
        img, bg_percentile=10.0, high_percentile=99.0, per_channel=False
    )
    assert float(out.min()) == pytest.approx(0.0)
    # Output above bg-floor + span should land near 1.0; clipping floor at 0.
    assert float(out.max()) >= 0.99
    assert meta["stats"][0]["bg"] == pytest.approx(np.percentile(img, 10))


def test_bg_subtract_per_channel():
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[..., 0] = np.arange(100).reshape(10, 10) + 500.0
    img[..., 1] = np.arange(100).reshape(10, 10) + 5.0
    out, _ = ops.normalize_bg_subtract(img, per_channel=True)
    # Both channels brought down to a comparable floor near 0.
    assert float(out[..., 0].min()) == pytest.approx(0.0)
    assert float(out[..., 1].min()) == pytest.approx(0.0)


def test_background_subtraction_residual_rescaled_to_unit_p99():
    """The value at the p99 of the residual maps to ~1.0 in the output.

    (The max can exceed 1.0 — the op clips only at 0, not at 1 — so we test
    the p99 invariant, which is what the rescale is defined to do.)
    """
    rng = np.random.default_rng(0)
    img = rng.normal(loc=50.0, scale=2.0, size=(40, 40)).astype(np.float32)
    # Sprinkle a moderate blob covering many pixels so p99 lands within it.
    img[10:30, 10:30] += 80.0
    out, meta = ops.background_subtraction(img, radius=8, per_channel=True)
    assert out.shape == img.shape
    assert float(out.min()) >= 0.0
    # After rescale the p99 of the OUTPUT equals 1.0 (up to clipping at 0).
    assert float(np.percentile(out, 99)) == pytest.approx(1.0, abs=1e-2)
    assert meta["radius"] == 8
    assert meta["method"] == "tophat"
    assert "high" in meta["stats"][0]


def test_background_subtraction_default_radius_is_25():
    """Sanity: the new default matches the post-eval recommendation."""
    import inspect
    sig = inspect.signature(ops.background_subtraction)
    assert sig.parameters["radius"].default == 25


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


def test_pad_to_square_landscape_input():
    """Wider-than-tall input pads vertically to a square of side w."""
    img = np.ones((4, 8), dtype=np.uint8) * 9
    out, meta = ops.pad_to_square(img, method="constant", constant_value=0)
    assert out.shape == (8, 8)
    assert meta["target"] == [8, 8]
    # Original block centred in the output.
    assert (out[2:6, :] == 9).all()
    assert out[0, 0] == 0


def test_pad_to_square_portrait_input():
    """Taller-than-wide input pads horizontally to a square of side h."""
    img = np.ones((10, 4), dtype=np.uint8) * 7
    out, _ = ops.pad_to_square(img, method="constant", constant_value=0)
    assert out.shape == (10, 10)
    assert (out[:, 3:7] == 7).all()


def test_pad_to_square_already_square_is_noop():
    img = np.ones((6, 6), dtype=np.uint8) * 3
    out, meta = ops.pad_to_square(img, method="constant")
    assert out.shape == (6, 6)
    assert (out == 3).all()
    assert meta["pad"] == [0, 0, 0, 0]


def test_pad_to_square_multichannel_hwc():
    img = np.ones((4, 8, 3), dtype=np.uint8) * 5
    out, _ = ops.pad_to_square(img, method="constant", constant_value=0)
    assert out.shape == (8, 8, 3)
    assert (out[2:6, :, :] == 5).all()


def test_pad_to_square_poisson_preserves_centre():
    img = np.full((4, 8), 50, dtype=np.uint8)
    out, _ = ops.pad_to_square(img, method="poisson")
    assert out.shape == (8, 8)
    assert (out[2:6, :] == 50).all()


def test_pad_to_square_replicate_keeps_input_block():
    img = np.full((4, 8), 7, dtype=np.uint8)
    out, _ = ops.pad_to_square(img, method="replicate")
    assert out.shape == (8, 8)
    assert (out == 7).all()


def test_pad_to_square_rejects_unsupported_ndim():
    bad = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        ops.pad_to_square(bad)


def test_resize_changes_shape_only():
    img = np.zeros((10, 10), dtype=np.uint8)
    out, _ = ops.resize_to(img, (20, 30))
    assert out.shape == (20, 30)


def test_resize_handles_unsupported_dtype_int32():
    """cv2.resize doesn't accept int32 directly — resize_to must cast safely."""
    img = np.full((8, 8), 1000, dtype=np.int32)
    out, _ = ops.resize_to(img, (16, 16))
    assert out.shape == (16, 16)
    assert out.dtype == np.int32
    assert int(out.mean()) == 1000


def test_resize_handles_unsupported_dtype_bool():
    img = np.ones((4, 4), dtype=bool)
    out, _ = ops.resize_to(img, (8, 8))
    assert out.shape == (8, 8)
    assert out.dtype == bool
    assert out.all()


def test_resize_multichannel_unsupported_dtype():
    img = np.full((4, 6, 3), 500, dtype=np.int32)
    out, _ = ops.resize_to(img, (8, 12))
    assert out.shape == (8, 12, 3)
    assert out.dtype == np.int32


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


def test_build_pipeline_with_explicit_pad_and_percentile():
    p = build_pipeline({
        "target_shape_strategy": "explicit",
        "explicit_shape": [16, 16],
        "padding_method": "constant",
        "normalize": "percentile",
    })
    img = (np.random.randn(8, 8) * 4 + 50).astype(np.float32)
    out, meta = p.apply(img)
    assert out.shape == (16, 16)
    step_names = [s["step"] for s in meta["steps"]]
    assert step_names == ["pad_to", "normalize_percentile"]


def test_build_pipeline_with_bg_subtract():
    p = build_pipeline({
        "target_shape_strategy": "none",
        "normalize": "bg_subtract",
    })
    assert [s.name for s in p.steps] == ["normalize_bg_subtract"]


def test_build_pipeline_mean_strategy_uses_summary():
    """mean strategy uses the average of mean_h and mean_w as the square side."""
    summary = {"count": 3, "min_h": 8, "max_h": 24, "mean_h": 16,
               "min_w": 10, "max_w": 26, "mean_w": 18, "n_channels": 1}
    p = build_pipeline(
        {"target_shape_strategy": "mean", "padding_method": "constant"},
        shape_summary=summary,
    )
    img = np.zeros((10, 12), dtype=np.uint8)
    out, _ = p.apply(img)
    # (mean_h + mean_w) / 2 = (16 + 18) / 2 = 17 → square (17, 17)
    assert out.shape == (17, 17)


def test_resolve_target_shape_min_is_square():
    """min strategy takes the minimum across both axes — square output."""
    from core.preprocessing.pipeline import _resolve_target_shape
    summary = {"count": 3, "min_h": 8, "max_h": 24, "mean_h": 16,
               "min_w": 10, "max_w": 26, "mean_w": 18, "n_channels": 1}
    assert _resolve_target_shape("min", None, summary) == (8, 8)


def test_resolve_target_shape_max_is_square():
    """max strategy takes the maximum across both axes — square output."""
    from core.preprocessing.pipeline import _resolve_target_shape
    summary = {"count": 3, "min_h": 8, "max_h": 24, "mean_h": 16,
               "min_w": 10, "max_w": 26, "mean_w": 18, "n_channels": 1}
    assert _resolve_target_shape("max", None, summary) == (26, 26)


def test_resolve_target_shape_mean_is_square():
    from core.preprocessing.pipeline import _resolve_target_shape
    summary = {"count": 3, "min_h": 8, "max_h": 24, "mean_h": 16,
               "min_w": 10, "max_w": 26, "mean_w": 18, "n_channels": 1}
    assert _resolve_target_shape("mean", None, summary) == (17, 17)


def test_resolve_target_shape_explicit_passes_through_non_square():
    """Explicit shapes are honoured as-is (caller chose the aspect)."""
    from core.preprocessing.pipeline import _resolve_target_shape
    out = _resolve_target_shape("explicit", (64, 128), summary=None)
    assert out == (64, 128)


def test_resolve_target_shape_none_summary_returns_none():
    from core.preprocessing.pipeline import _resolve_target_shape
    assert _resolve_target_shape("mean", None, None) is None


def test_resolve_target_shape_already_square_summary():
    """When height stats equal width stats, the square output matches them."""
    from core.preprocessing.pipeline import _resolve_target_shape
    summary = {"count": 5, "min_h": 32, "max_h": 64, "mean_h": 48,
               "min_w": 32, "max_w": 64, "mean_w": 48, "n_channels": 1}
    assert _resolve_target_shape("min", None, summary) == (32, 32)
    assert _resolve_target_shape("max", None, summary) == (64, 64)
    assert _resolve_target_shape("mean", None, summary) == (48, 48)


def test_resolve_target_shape_mean_rounds_half_to_nearest():
    """Mean of an odd sum is rounded to the nearest integer."""
    from core.preprocessing.pipeline import _resolve_target_shape
    summary = {"count": 2, "min_h": 10, "max_h": 11, "mean_h": 10,
               "min_w": 11, "max_w": 11, "mean_w": 11, "n_channels": 1}
    # (10 + 11) / 2 = 10.5 → 10 (banker's rounding in Python's round())
    assert _resolve_target_shape("mean", None, summary) == (10, 10)


def test_resolve_target_shape_unknown_raises():
    from core.preprocessing.pipeline import _resolve_target_shape
    summary = {"count": 1, "min_h": 10, "max_h": 10, "mean_h": 10,
               "min_w": 10, "max_w": 10, "mean_w": 10, "n_channels": 1}
    with pytest.raises(ValueError):
        _resolve_target_shape("bogus", None, summary)


def test_resolve_target_shape_explicit_requires_value():
    from core.preprocessing.pipeline import _resolve_target_shape
    with pytest.raises(ValueError):
        _resolve_target_shape("explicit", None, summary=None)


# ── per_image_square strategy ───────────────────────────────────────────


def test_pipeline_per_image_square_no_post_resize():
    """per_image_square strategy emits a pad_to_square step and no global pad/resize."""
    p = build_pipeline({
        "target_shape_strategy": "per_image_square",
        "padding_method": "constant",
    })
    step_names = [s.name for s in p.steps]
    assert step_names == ["pad_to_square"]
    assert p.steps[0].params == {"method": "constant"}
    assert p.target_shape is None  # output shape varies per image


def test_pipeline_per_image_square_applies_per_image_shape():
    """Two different non-square inputs produce two different square outputs."""
    p = build_pipeline({
        "target_shape_strategy": "per_image_square",
        "padding_method": "constant",
    })
    img_a = np.zeros((4, 10), dtype=np.uint8)
    img_b = np.zeros((20, 8), dtype=np.uint8)
    out_a, _ = p.apply(img_a)
    out_b, _ = p.apply(img_b)
    assert out_a.shape == (10, 10)
    assert out_b.shape == (20, 20)


def test_pipeline_per_image_square_with_explicit_post_resize():
    """post_resize_strategy='explicit' appends a resize_to((N, N)) step."""
    p = build_pipeline({
        "target_shape_strategy": "per_image_square",
        "padding_method": "constant",
        "post_resize_strategy": "explicit",
        "post_resize_value": 32,
    })
    step_names = [s.name for s in p.steps]
    assert step_names == ["pad_to_square", "resize_to"]
    assert p.steps[1].params == {"target_shape": (32, 32)}
    assert p.target_shape == (32, 32)
    # End-to-end: any input ends up 32×32.
    img = np.zeros((4, 10), dtype=np.uint8)
    out, _ = p.apply(img)
    assert out.shape == (32, 32)


def test_pipeline_per_image_square_with_max_longest_post_resize():
    """post_resize_strategy='max_longest' reads from shape_summary."""
    summary = {
        "count": 3, "min_h": 4, "max_h": 20, "mean_h": 12,
        "min_w": 8, "max_w": 16, "mean_w": 12,
        "min_longest": 10, "max_longest": 24, "mean_longest": 17,
        "n_channels": 1,
    }
    p = build_pipeline(
        {
            "target_shape_strategy": "per_image_square",
            "padding_method": "constant",
            "post_resize_strategy": "max_longest",
        },
        shape_summary=summary,
    )
    step_names = [s.name for s in p.steps]
    assert step_names == ["pad_to_square", "resize_to"]
    assert p.steps[1].params == {"target_shape": (24, 24)}
    assert p.target_shape == (24, 24)


def test_pipeline_per_image_square_min_and_mean_longest():
    summary = {
        "count": 3, "min_h": 4, "max_h": 20, "mean_h": 12,
        "min_w": 8, "max_w": 16, "mean_w": 12,
        "min_longest": 10, "max_longest": 24, "mean_longest": 17,
        "n_channels": 1,
    }
    p_min = build_pipeline(
        {"target_shape_strategy": "per_image_square",
         "padding_method": "constant",
         "post_resize_strategy": "min_longest"},
        shape_summary=summary,
    )
    assert p_min.steps[1].params == {"target_shape": (10, 10)}
    p_mean = build_pipeline(
        {"target_shape_strategy": "per_image_square",
         "padding_method": "constant",
         "post_resize_strategy": "mean_longest"},
        shape_summary=summary,
    )
    assert p_mean.steps[1].params == {"target_shape": (17, 17)}


def test_pipeline_per_image_square_dataset_stat_without_summary_raises():
    """A dataset-stat post-resize without a shape_summary is a usage error."""
    with pytest.raises(ValueError, match="shape_summary"):
        build_pipeline({
            "target_shape_strategy": "per_image_square",
            "padding_method": "constant",
            "post_resize_strategy": "max_longest",
        })


def test_pipeline_per_image_square_keeps_normalize_after():
    """Order: pad_to_square → resize_to → normalize."""
    p = build_pipeline({
        "target_shape_strategy": "per_image_square",
        "padding_method": "constant",
        "post_resize_strategy": "explicit",
        "post_resize_value": 16,
        "normalize": "percentile",
    })
    assert [s.name for s in p.steps] == ["pad_to_square", "resize_to", "normalize_percentile"]
