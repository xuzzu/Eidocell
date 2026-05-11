# Per-Image Square Padding (with optional global resize)

## Motivation

The current import preprocessing pipeline picks **one global target shape**
(`min` / `max` / `mean` / `explicit`) and either pads or resizes every image
to that shape. For IFC datasets where cells vary widely in size, a global
`max` target wastes a lot of space padding small cells, and `mean` distorts
the largest cells.

A common workflow is: make each image individually square (pad to its own
longest side), then optionally resize **all** images to a single resolution
for downstream batched models (CNNs, autoencoders). This spec adds that
mode without disturbing the existing global-target strategies.

## Scope

In scope:

- New `target_shape_strategy` value: `per_image_square`.
- New op `pad_to_square` reusing the three existing padding methods
  (constant / poisson / replicate).
- Optional follow-up resize to a single N×N, where N is either user-entered
  or a dataset statistic over per-image longest sides.
- Backend schema, pipeline, shape summary, op, tests.
- Frontend wizard UI and types.

Out of scope:

- Rectangular per-image targets (only square per-image padding is supported).
- Changes to existing global strategies (`min` / `max` / `mean` / `explicit`).
- Database migrations — `preprocessing_config` is stored as JSON in the
  `imports` table, so new keys deserialize without schema work.

## Architecture

```
existing flow:           ┌─────────────┐    ┌──────────────────┐
                         │ resolve     │ →  │ pad_to or        │
                         │ target_shape│    │ resize_to (one)  │
                         └─────────────┘    └──────────────────┘

new per_image_square     ┌──────────────────────────────────┐
flow:                    │ pad_to_square (per image, side = │
                         │ max(H, W) for that image)        │
                         └──────────────────────────────────┘
                                       ↓ optional
                         ┌──────────────────────────────────┐
                         │ resize_to((N, N))  — N from      │
                         │ explicit input or shape_summary  │
                         └──────────────────────────────────┘
```

The pipeline is already structured as a list of per-image ops, so the
per-image-square step slots in at the same position as the existing global
pad/resize step. Other steps (background subtraction, channel alignment,
compensation, normalization) are unaffected.

## Schema

`eidocell-backend/schemas/imports.py`:

```python
TargetShapeStrategy = Literal[
    "none", "min", "max", "mean", "explicit", "per_image_square"
]
PostResizeStrategy = Literal[
    "none", "explicit", "min_longest", "max_longest", "mean_longest"
]

class PreprocessingConfig(BaseModel):
    # ... existing fields unchanged ...
    post_resize_strategy: PostResizeStrategy = "none"
    post_resize_value: int | None = None   # used only when strategy == "explicit"
```

`eidocell-ui/src/types/imports.ts` mirrors the additions:

```ts
export type TargetShapeStrategy =
  | 'none' | 'min' | 'max' | 'mean' | 'explicit' | 'per_image_square'
export type PostResizeStrategy =
  | 'none' | 'explicit' | 'min_longest' | 'max_longest' | 'mean_longest'

export interface PreprocessingConfig {
  // ... existing ...
  post_resize_strategy: PostResizeStrategy
  post_resize_value: number | null
}
```

`DEFAULT_PREPROCESSING` gains `post_resize_strategy: 'none'` and
`post_resize_value: null`.

Semantics:

- `post_resize_*` are **ignored** unless `target_shape_strategy === 'per_image_square'`.
- The legacy `resize: bool` and `explicit_shape` are unchanged; they are
  hidden in the UI and ignored by the pipeline when the new strategy is
  selected. No validator change is needed.

## Backend

### New op — `core/preprocessing/ops.py`

```python
def pad_to_square(
    arr: np.ndarray,
    *,
    method: str = "constant",
    constant_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Pad ``arr`` to a square whose side equals max(H, W). Centred."""
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    side = max(h, w)
    return pad_to(arr, (side, side), method=method, constant_value=constant_value)
```

`pad_to` already handles the 2-D ↔ HWC promote/restore internally, so
`pad_to_square` is a pure thin wrapper. Metadata payload matches `pad_to`
so consumers don't need to branch. Already-square inputs short-circuit
through `pad_to`'s existing zero-pad fast path.

Add `"pad_to_square"` to `__all__`.

### Shape summary — `core/storage/import_staging.py`

Extend the returned dict with three keys derived from per-image longest sides:

```python
longest = np.maximum(heights, widths)
"min_longest":  int(longest.min()),
"max_longest":  int(longest.max()),
"mean_longest": int(round(longest.mean())),
```

### Pipeline builder — `core/preprocessing/pipeline.py`

In `build_pipeline`, replace the current `if target is not None:` block with
a branch on strategy:

```python
if strategy == "per_image_square":
    steps.append(Step(
        name="pad_to_square",
        fn=_ops.pad_to_square,
        params={"method": cfg.get("padding_method", "constant")},
    ))
    post = cfg.get("post_resize_strategy", "none")
    if post != "none":
        if post == "explicit":
            n = int(cfg["post_resize_value"])
        else:
            key = {"min_longest": "min_longest",
                   "max_longest": "max_longest",
                   "mean_longest": "mean_longest"}[post]
            if not shape_summary or key not in shape_summary:
                raise ValueError(
                    f"shape_summary missing '{key}' for post_resize_strategy='{post}'"
                )
            n = int(shape_summary[key])
        steps.append(Step(
            name="resize_to",
            fn=_ops.resize_to,
            params={"target_shape": (n, n)},
        ))
        target = (n, n)
elif target is not None:
    # ... existing global pad/resize logic, unchanged ...
```

`Pipeline.target_shape` is set to `(N, N)` when post-resize is enabled, else
`None` (per-image-square outputs vary in size).

`_resolve_target_shape` does not need to know about the new strategy — the
new branch resolves its own targets.

## Frontend

In `eidocell-ui/src/views/ImportWizardView.vue`, extend the existing
`Target shape` block:

1. Add `<option value="per_image_square">Per-image square (pad to longest side)</option>`
   to the strategy dropdown.
2. When `preproc.target_shape_strategy === 'per_image_square'`:
   - Hide the "Resize (otherwise pad)" checkbox and the `explicit_shape`
     H/W inputs.
   - Show the existing padding-method selector (`constant` / `poisson` /
     `replicate`).
   - Show a new "Resize after" select bound to `preproc.post_resize_strategy`:
     - `None`
     - `Specify N×N`              → `explicit`
     - `Match smallest longest side` → `min_longest`
     - `Match mean longest side`     → `mean_longest`
     - `Match largest longest side`  → `max_longest`
   - When `'explicit'`: a single integer input bound to
     `preproc.post_resize_value`, rendered as "N × N".

Update the wizard's summary line (~L514) to render
`per-image square · pad <method>[ · resize N×N]` when the new strategy is
active.

## Tests

`eidocell-backend/tests/test_preprocessing_ops.py`:

- `pad_to_square` op: square output for non-square input, longest-side
  preservation, all three padding methods, multi-channel HWC, already-square
  inputs (no-op shape, metadata still returned).

`eidocell-backend/tests/test_preprocessing_ops.py` (pipeline builder):

- `per_image_square` strategy → pipeline contains `pad_to_square`, no
  global `pad_to` / `resize_to`.
- `per_image_square` + `post_resize_strategy='explicit'` with
  `post_resize_value=64` → pipeline ends with `resize_to((64, 64))`.
- `per_image_square` + `post_resize_strategy='max_longest'` with a stubbed
  `shape_summary={"max_longest": 96, ...}` → pipeline ends with
  `resize_to((96, 96))`.
- `per_image_square` + `post_resize_strategy='max_longest'` with no summary
  → raises `ValueError`.

`eidocell-backend/tests/test_import_service.py`:

- End-to-end: stage three images of different shapes
  (e.g. 10×20, 30×30, 40×15), run import with `per_image_square` +
  `post_resize_strategy='max_longest'`, assert all outputs are 40×40 and
  per-image metadata in `preprocessing_metadata.json` records both steps.

`eidocell-backend/tests/test_import_staging.py`:

- Assert `min_longest` / `max_longest` / `mean_longest` keys appear in
  `shape_summary` output.

No frontend tests — the wizard has no existing test harness; the new UI
is conditional rendering of existing primitives.

## Backwards Compatibility

- All new schema fields have defaults; existing serialized
  `preprocessing_config` JSON loads unchanged.
- The legacy `resize: bool` and `explicit_shape` fields keep their
  semantics; they're ignored only when the new strategy is selected.
- No DB migration required (preprocessing config is stored as JSON in
  `imports.preprocessing_config`).
