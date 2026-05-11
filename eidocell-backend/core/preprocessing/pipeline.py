"""Composable preprocessing pipeline.

Builds an ordered list of (op_callable, params) from a user-friendly config
dict (the ``PreprocessingConfig`` from the import wizard) and applies it to
each image. ``apply`` returns ``(image, metadata)`` so the import service can
record per-step parameters and stats in ``preprocessing_metadata.json``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.preprocessing import ops as _ops


StepFn = Callable[..., "tuple[np.ndarray, dict]"]


@dataclass
class Step:
    name: str
    fn: StepFn
    params: dict


@dataclass
class Pipeline:
    pipeline_id: str
    steps: list[Step]
    target_shape: tuple[int, int] | None = None
    config: dict = field(default_factory=dict)

    def apply(self, arr: np.ndarray) -> tuple[np.ndarray, dict]:
        history: list[dict] = []
        cur = arr
        for step in self.steps:
            params = dict(step.params)
            cur, meta = step.fn(cur, **params)
            history.append({"step": step.name, "params": params, "metadata": meta})
        return cur, {
            "pipeline_id": self.pipeline_id,
            "input_shape": list(arr.shape),
            "output_shape": list(cur.shape),
            "steps": history,
        }


def _resolve_target_shape(strategy: str, explicit: tuple[int, int] | None, summary: dict | None) -> tuple[int, int] | None:
    """Resolve the preprocessing target shape.

    ``min`` / ``max`` / ``mean`` produce a **square** target whose side is the
    chosen statistic over **all sides** (heights and widths combined) in the
    dataset. ``explicit`` is honoured as-is and may be non-square.
    """
    if strategy == "explicit":
        if explicit is None:
            raise ValueError("explicit_shape required when target_shape_strategy='explicit'")
        return tuple(explicit)
    if not summary:
        return None
    if strategy == "min":
        side = min(summary["min_h"], summary["min_w"])
        return (side, side)
    if strategy == "max":
        side = max(summary["max_h"], summary["max_w"])
        return (side, side)
    if strategy == "mean":
        # Each image contributes one H and one W, weighted equally — average the
        # per-axis means.
        side = int(round((summary["mean_h"] + summary["mean_w"]) / 2))
        return (side, side)
    raise ValueError(f"unknown target_shape_strategy '{strategy}'")


def build_pipeline(config: dict, *, shape_summary: dict | None = None) -> Pipeline:
    """Build a Pipeline from the wizard's PreprocessingConfig dict.

    Recognised keys:
        target_shape_strategy: "min" | "max" | "mean" | "explicit" | "none"
        explicit_shape: [int, int] | None
        padding_method: "constant" | "poisson" | "replicate"
        resize: bool
        normalize: "none" | "minmax" | "zscore"
        align_channels: bool
        align_reference_index: int
        compensation_matrix: list[list[float]] | None
    """
    steps: list[Step] = []
    cfg = dict(config or {})
    strategy = cfg.get("target_shape_strategy", "none")
    explicit = cfg.get("explicit_shape")
    target: tuple[int, int] | None = None
    if strategy not in ("none", "per_image_square"):
        target = _resolve_target_shape(
            strategy,
            tuple(explicit) if explicit else None,
            shape_summary,
        )

    if cfg.get("background_subtraction"):
        steps.append(Step(
            name="background_subtraction",
            fn=_ops.background_subtraction,
            params={
                "radius": int(cfg.get("background_subtraction_radius", 50)),
                "per_channel": bool(cfg.get("background_subtraction_per_channel", True))
            },
        ))

    if cfg.get("align_channels"):
        steps.append(Step(
            name="align_channels",
            fn=_ops.align_channels,
            params={
                "reference_index": int(cfg.get("align_reference_index", 0)),
                "upsample_factor": int(cfg.get("align_upsample_factor", 10)),
            },
        ))

    if cfg.get("compensation_matrix"):
        steps.append(Step(
            name="compensate",
            fn=_ops.compensate,
            params={"matrix": cfg["compensation_matrix"]},
        ))

    if strategy == "per_image_square":
        steps.append(Step(
            name="pad_to_square",
            fn=_ops.pad_to_square,
            params={"method": cfg.get("padding_method", "constant")},
        ))
        post = cfg.get("post_resize_strategy", "none")
        if post != "none":
            if post == "explicit":
                value = cfg.get("post_resize_value")
                if value is None:
                    raise ValueError(
                        "post_resize_value required when post_resize_strategy='explicit'"
                    )
                n = int(value)
            else:
                if post not in ("min_longest", "max_longest", "mean_longest"):
                    raise ValueError(f"unknown post_resize_strategy '{post}'")
                if not shape_summary or post not in shape_summary:
                    raise ValueError(
                        f"shape_summary missing '{post}' for post_resize_strategy='{post}'"
                    )
                n = int(shape_summary[post])
            steps.append(Step(
                name="resize_to",
                fn=_ops.resize_to,
                params={"target_shape": (n, n)},
            ))
            target = (n, n)
    elif target is not None:
        if cfg.get("resize"):
            steps.append(Step(
                name="resize_to",
                fn=_ops.resize_to,
                params={"target_shape": target},
            ))
        else:
            steps.append(Step(
                name="pad_to",
                fn=_ops.pad_to,
                params={
                    "target_shape": target,
                    "method": cfg.get("padding_method", "constant"),
                },
            ))

    norm = cfg.get("normalize", "none")
    if norm == "minmax":
        steps.append(Step(name="normalize_minmax", fn=_ops.normalize_minmax, params={
            "per_channel": bool(cfg.get("normalize_per_channel", True)),
        }))
    elif norm == "zscore":
        steps.append(Step(name="normalize_zscore", fn=_ops.normalize_zscore, params={
            "per_channel": bool(cfg.get("normalize_per_channel", True)),
        }))

    return Pipeline(
        pipeline_id=uuid.uuid4().hex[:12],
        steps=steps,
        target_shape=target,
        config=cfg,
    )
