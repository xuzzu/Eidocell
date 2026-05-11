# Per-Image Square Padding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `target_shape_strategy='per_image_square'` mode that pads each image to its own longest side, with an optional follow-up resize to a single N×N (user-entered or dataset-stat over longest sides).

**Architecture:** Add a new `pad_to_square` op that delegates to the existing `pad_to`. Extend the shape-summary helper with longest-side stats. Branch the pipeline builder on the new strategy. Mirror schema + types in the frontend. Add a conditional UI block in the import wizard.

**Tech Stack:** Python 3.11 / FastAPI / numpy / OpenCV / pytest (backend); Vue 3 + TypeScript / Vite (frontend).

**Spec:** `docs/superpowers/specs/2026-05-11-per-image-square-padding-design.md`

---

## File Map

**Backend — modify:**
- `eidocell-backend/core/preprocessing/ops.py` — add `pad_to_square`
- `eidocell-backend/core/preprocessing/pipeline.py` — branch on new strategy in `build_pipeline`
- `eidocell-backend/core/storage/import_staging.py` — add longest-side stats to `shape_summary`
- `eidocell-backend/schemas/imports.py` — extend `TargetShapeStrategy` literal, add `PostResizeStrategy`, add two fields to `PreprocessingConfig`

**Backend — tests modify:**
- `eidocell-backend/tests/test_preprocessing_ops.py` — new tests for op + pipeline builder
- `eidocell-backend/tests/test_import_staging.py` — assert new summary keys
- `eidocell-backend/tests/test_import_service.py` — end-to-end test

**Frontend — modify:**
- `eidocell-ui/src/types/imports.ts` — extend types and defaults
- `eidocell-ui/src/views/ImportWizardView.vue` — extend wizard UI

---

## Task 1: `pad_to_square` op + tests

**Files:**
- Modify: `eidocell-backend/core/preprocessing/ops.py`
- Test: `eidocell-backend/tests/test_preprocessing_ops.py`

Working dir for all backend bash steps: `eidocell-backend/`.

- [ ] **Step 1: Write failing tests for `pad_to_square`**

Append at the bottom of `eidocell-backend/tests/test_preprocessing_ops.py`, BEFORE the existing pipeline tests block — i.e., immediately after the last `test_pad_*` test (around line 60). Place these adjacent to the other padding tests:

```python
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
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd eidocell-backend && poetry run pytest tests/test_preprocessing_ops.py -k pad_to_square -v
```

Expected: all 7 tests FAIL with `AttributeError: module 'core.preprocessing.ops' has no attribute 'pad_to_square'`.

- [ ] **Step 3: Implement `pad_to_square` in `ops.py`**

In `eidocell-backend/core/preprocessing/ops.py`, insert this function immediately after `pad_to` (before `resize_to`):

```python
def pad_to_square(
    arr: np.ndarray,
    *,
    method: str = "constant",
    constant_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Pad ``arr`` to a square whose side equals max(H, W). Centred.

    Thin wrapper over ``pad_to`` that derives the target shape per-image
    from the input's longest side. Used by the ``per_image_square`` import
    strategy.
    """
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    side = max(h, w)
    return pad_to(arr, (side, side), method=method, constant_value=constant_value)
```

Also add `"pad_to_square"` to the `__all__` list at the bottom of the file. The current list is:

```python
__all__ = [
    "normalize_minmax",
    "normalize_zscore",
    "pad_to",
    "resize_to",
    "align_channels",
    "compensate",
    "background_subtraction",
]
```

Replace it with:

```python
__all__ = [
    "normalize_minmax",
    "normalize_zscore",
    "pad_to",
    "pad_to_square",
    "resize_to",
    "align_channels",
    "compensate",
    "background_subtraction",
]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd eidocell-backend && poetry run pytest tests/test_preprocessing_ops.py -k pad_to_square -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Run the full preprocessing test file to confirm no regressions**

```bash
cd eidocell-backend && poetry run pytest tests/test_preprocessing_ops.py -v
```

Expected: all existing tests still PASS in addition to the new ones.

- [ ] **Step 6: Commit**

```bash
git add eidocell-backend/core/preprocessing/ops.py eidocell-backend/tests/test_preprocessing_ops.py
git commit -m "Add pad_to_square op for per-image square padding"
```

---

## Task 2: Longest-side stats in `shape_summary`

**Files:**
- Modify: `eidocell-backend/core/storage/import_staging.py:165-182`
- Test: `eidocell-backend/tests/test_import_staging.py`

- [ ] **Step 1: Extend the existing summary test to assert new keys**

Open `eidocell-backend/tests/test_import_staging.py`. Find `test_round_trip_multi_channel_and_summary` (around line 53). After the existing `assert s["min_w"] == 30 ...` line (line 67), add these assertions:

```python
    # Longest-side stats: per-image max(H, W).
    # arr_small is 20×30 → longest=30; arr_big is 40×50 → longest=50.
    assert s["min_longest"] == 30
    assert s["max_longest"] == 50
    assert s["mean_longest"] == 40
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd eidocell-backend && poetry run pytest tests/test_import_staging.py::test_round_trip_multi_channel_and_summary -v
```

Expected: FAIL with `KeyError: 'min_longest'`.

- [ ] **Step 3: Add longest-side stats to `shape_summary`**

Open `eidocell-backend/core/storage/import_staging.py`. Replace the body of `shape_summary` (currently lines 165-182). Here is the new full function:

```python
def shape_summary(session_id: str, import_id: str) -> dict:
    """Return aggregate shape stats over staged rows for preprocessing planning."""
    try:
        table = open_table(session_id, import_id, create_if_missing=False)
    except FileNotFoundError:
        return {"count": 0}
    arrow = table.search().select(["height", "width", "n_channels"]).to_arrow()
    if arrow.num_rows == 0:
        return {"count": 0}
    heights = np.asarray(arrow.column("height").to_pylist(), dtype=np.int32)
    widths = np.asarray(arrow.column("width").to_pylist(), dtype=np.int32)
    chans = np.asarray(arrow.column("n_channels").to_pylist(), dtype=np.int32)
    longest = np.maximum(heights, widths)
    return {
        "count": int(arrow.num_rows),
        "min_h": int(heights.min()), "max_h": int(heights.max()), "mean_h": int(round(heights.mean())),
        "min_w": int(widths.min()),  "max_w": int(widths.max()),  "mean_w": int(round(widths.mean())),
        "min_longest":  int(longest.min()),
        "max_longest":  int(longest.max()),
        "mean_longest": int(round(longest.mean())),
        "n_channels": int(chans.max()),
    }
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cd eidocell-backend && poetry run pytest tests/test_import_staging.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add eidocell-backend/core/storage/import_staging.py eidocell-backend/tests/test_import_staging.py
git commit -m "Add longest-side stats to shape_summary"
```

---

## Task 3: Schema fields for new strategy + post-resize

**Files:**
- Modify: `eidocell-backend/schemas/imports.py:7-26`

This task is type-system-only — no logic changes, no new tests. We verify with a one-shot Python import.

- [ ] **Step 1: Extend literals and add fields**

Open `eidocell-backend/schemas/imports.py`. Replace the literal definitions and `PreprocessingConfig` class. The current top of the file (lines 7-26) reads:

```python
SourceKind = Literal["folder", "cif", "rif"]
TargetShapeStrategy = Literal["none", "min", "max", "mean", "explicit"]
NormalizeStrategy = Literal["none", "minmax", "zscore"]
PaddingMethod = Literal["constant", "poisson", "replicate"]


class PreprocessingConfig(BaseModel):
    target_shape_strategy: TargetShapeStrategy = "none"
    explicit_shape: list[int] | None = None  # [H, W]
    padding_method: PaddingMethod = "constant"
    resize: bool = False
    normalize: NormalizeStrategy = "none"
    normalize_per_channel: bool = True
    background_subtraction: bool = False
    background_subtraction_radius: int = 50
    background_subtraction_per_channel: bool = True
    align_channels: bool = False
    align_reference_index: int = 0
    align_upsample_factor: int = 10
    compensation_matrix: list[list[float]] | None = None
```

Replace it with:

```python
SourceKind = Literal["folder", "cif", "rif"]
TargetShapeStrategy = Literal[
    "none", "min", "max", "mean", "explicit", "per_image_square"
]
NormalizeStrategy = Literal["none", "minmax", "zscore"]
PaddingMethod = Literal["constant", "poisson", "replicate"]
PostResizeStrategy = Literal[
    "none", "explicit", "min_longest", "max_longest", "mean_longest"
]


class PreprocessingConfig(BaseModel):
    target_shape_strategy: TargetShapeStrategy = "none"
    explicit_shape: list[int] | None = None  # [H, W]
    padding_method: PaddingMethod = "constant"
    resize: bool = False
    normalize: NormalizeStrategy = "none"
    normalize_per_channel: bool = True
    background_subtraction: bool = False
    background_subtraction_radius: int = 50
    background_subtraction_per_channel: bool = True
    align_channels: bool = False
    align_reference_index: int = 0
    align_upsample_factor: int = 10
    compensation_matrix: list[list[float]] | None = None
    post_resize_strategy: PostResizeStrategy = "none"
    post_resize_value: int | None = None
```

- [ ] **Step 2: Smoke-test the schema loads with new fields**

```bash
cd eidocell-backend && poetry run python -c "
from schemas.imports import PreprocessingConfig
c = PreprocessingConfig(target_shape_strategy='per_image_square', post_resize_strategy='explicit', post_resize_value=64)
print(c.model_dump())
"
```

Expected output (printed dict): includes `'target_shape_strategy': 'per_image_square'`, `'post_resize_strategy': 'explicit'`, `'post_resize_value': 64`, plus the defaults for everything else.

- [ ] **Step 3: Verify back-compat — old config still loads**

```bash
cd eidocell-backend && poetry run python -c "
from schemas.imports import PreprocessingConfig
c = PreprocessingConfig.model_validate({'target_shape_strategy': 'max', 'padding_method': 'poisson'})
assert c.post_resize_strategy == 'none'
assert c.post_resize_value is None
print('OK')
"
```

Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
git add eidocell-backend/schemas/imports.py
git commit -m "Add per_image_square strategy and post-resize fields to schema"
```

---

## Task 4: Pipeline builder branch + tests

**Files:**
- Modify: `eidocell-backend/core/preprocessing/pipeline.py:78-162`
- Test: `eidocell-backend/tests/test_preprocessing_ops.py`

- [ ] **Step 1: Write failing tests for the new pipeline branch**

Append to `eidocell-backend/tests/test_preprocessing_ops.py` (at the bottom of the file, after `test_resolve_target_shape_explicit_requires_value`):

```python
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
        "normalize": "zscore",
    })
    assert [s.name for s in p.steps] == ["pad_to_square", "resize_to", "normalize_zscore"]
```

- [ ] **Step 2: Run the new tests — confirm they fail**

```bash
cd eidocell-backend && poetry run pytest tests/test_preprocessing_ops.py -k per_image_square -v
```

Expected: all 7 tests FAIL — most with `KeyError`, `AssertionError`, or because the pipeline currently emits no steps for the unknown strategy.

- [ ] **Step 3: Modify `build_pipeline` to handle the new strategy**

Open `eidocell-backend/core/preprocessing/pipeline.py`. The current target-resolution + pad/resize block (lines 92-145) reads:

```python
    steps: list[Step] = []
    cfg = dict(config or {})
    strategy = cfg.get("target_shape_strategy", "none")
    explicit = cfg.get("explicit_shape")
    target = None
    if strategy != "none":
        target = _resolve_target_shape(
            strategy,
            tuple(explicit) if explicit else None,
            shape_summary,
        )

    if cfg.get("background_subtraction"):
        steps.append(Step(
            ...
        ))

    if cfg.get("align_channels"):
        steps.append(Step(
            ...
        ))

    if cfg.get("compensation_matrix"):
        steps.append(Step(
            ...
        ))

    if target is not None:
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
```

Two surgical edits:

**Edit A**: Skip the `_resolve_target_shape` call for the new strategy (it doesn't have a single global target). Replace the `target = None` / `if strategy != "none": ...` block (current lines 95-101) with:

```python
    target: tuple[int, int] | None = None
    if strategy not in ("none", "per_image_square"):
        target = _resolve_target_shape(
            strategy,
            tuple(explicit) if explicit else None,
            shape_summary,
        )
```

**Edit B**: Replace the trailing `if target is not None:` block (currently the last 16 lines of the function, before the `norm = ...` block — lines 130-145) with the version below. This adds the `per_image_square` branch ahead of the existing `pad_to` / `resize_to` branch:

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
                value = cfg.get("post_resize_value")
                if value is None:
                    raise ValueError(
                        "post_resize_value required when post_resize_strategy='explicit'"
                    )
                n = int(value)
            else:
                key_map = {
                    "min_longest": "min_longest",
                    "max_longest": "max_longest",
                    "mean_longest": "mean_longest",
                }
                if post not in key_map:
                    raise ValueError(f"unknown post_resize_strategy '{post}'")
                key = key_map[post]
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
```

Don't change anything below that block (normalization stays after).

- [ ] **Step 4: Run new tests — confirm they pass**

```bash
cd eidocell-backend && poetry run pytest tests/test_preprocessing_ops.py -k per_image_square -v
```

Expected: all 7 PASS.

- [ ] **Step 5: Run the full preprocessing test file — confirm no regressions**

```bash
cd eidocell-backend && poetry run pytest tests/test_preprocessing_ops.py -v
```

Expected: every test in the file PASSes (including the existing global-strategy tests).

- [ ] **Step 6: Commit**

```bash
git add eidocell-backend/core/preprocessing/pipeline.py eidocell-backend/tests/test_preprocessing_ops.py
git commit -m "Wire per_image_square strategy into pipeline builder"
```

---

## Task 5: End-to-end test via import service

**Files:**
- Test: `eidocell-backend/tests/test_import_service.py`

- [ ] **Step 1: Write failing end-to-end test**

Append this test at the bottom of `eidocell-backend/tests/test_import_service.py`:

```python
def test_import_folder_per_image_square_with_post_resize(client, tmp_path):
    """per_image_square + post_resize_strategy=max_longest produces uniformly sized N×N outputs."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    # Three images with different shapes and longest sides 12, 30, 40.
    shapes = [(10, 12), (30, 8), (40, 15)]
    for i, hw in enumerate(shapes):
        arr = np.full((*hw, 3), 20 + i * 30, dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"cell_{i:03d}.png")

    sid = _create_session(client)
    resp = client.post(f"/sessions/{sid}/imports/", json={
        "source_kind": "folder",
        "source_path": str(img_dir),
        "channel_grouping": False,
        "preprocessing": {
            "target_shape_strategy": "per_image_square",
            "padding_method": "constant",
            "post_resize_strategy": "max_longest",
            "normalize": "none",
        },
    })
    assert resp.status_code == 202, resp.text
    info = _wait_for_task(client, resp.json()["task_id"])
    assert info["status"] == "completed", info

    detail = client.get(f"/sessions/{sid}/imports/{resp.json()['import_id']}").json()
    assert detail["sample_count"] == 3

    # Every output is square at max_longest = 40.
    from core.storage import images as image_store
    samples = client.post(f"/sessions/{sid}/samples/list", json={}).json()["items"]
    for s in samples:
        arr = image_store.read_array(sid, s["id"])
        assert arr is not None
        h, w = arr.shape[:2]
        assert (h, w) == (40, 40), f"sample {s['id']} has shape {(h, w)}"

    # Pipeline metadata records both steps.
    sess = client.get(f"/sessions/{sid}").json()
    meta_path = Path(sess["session_folder"]) / "imports" / f"{resp.json()['import_id']}.json"
    text = meta_path.read_text()
    assert "pad_to_square" in text
    assert "resize_to" in text
```

- [ ] **Step 2: Run the new test to confirm it passes**

```bash
cd eidocell-backend && poetry run pytest tests/test_import_service.py::test_import_folder_per_image_square_with_post_resize -v
```

Expected: PASS (since Tasks 1–4 are already merged, all the underlying machinery is in place).

If it FAILs, inspect the error: the most likely cause is `shape_summary` not being passed into `build_pipeline` by the import service. Check `eidocell-backend/services/import_service.py` for the call site and confirm it's already supplying `shape_summary`. If it's not, that's an additional surgical fix; otherwise the test simply verifies the full chain.

- [ ] **Step 3: Run the full import-service test file — confirm no regressions**

```bash
cd eidocell-backend && poetry run pytest tests/test_import_service.py -v
```

Expected: all tests in the file PASS.

- [ ] **Step 4: Commit**

```bash
git add eidocell-backend/tests/test_import_service.py
git commit -m "Add end-to-end test for per_image_square import flow"
```

---

## Task 6: Frontend types + defaults

**Files:**
- Modify: `eidocell-ui/src/types/imports.ts`

- [ ] **Step 1: Extend the type definitions and defaults**

Open `eidocell-ui/src/types/imports.ts`. Make three edits.

**Edit A** — extend the `TargetShapeStrategy` union (line 2) and add a new `PostResizeStrategy` type. Replace:

```ts
export type TargetShapeStrategy = 'none' | 'min' | 'max' | 'mean' | 'explicit'
export type NormalizeStrategy = 'none' | 'minmax' | 'zscore'
export type PaddingMethod = 'constant' | 'poisson' | 'replicate'
```

with:

```ts
export type TargetShapeStrategy =
  | 'none' | 'min' | 'max' | 'mean' | 'explicit' | 'per_image_square'
export type NormalizeStrategy = 'none' | 'minmax' | 'zscore'
export type PaddingMethod = 'constant' | 'poisson' | 'replicate'
export type PostResizeStrategy =
  | 'none' | 'explicit' | 'min_longest' | 'max_longest' | 'mean_longest'
```

**Edit B** — extend the `PreprocessingConfig` interface (currently lines 14-25). Replace:

```ts
export interface PreprocessingConfig {
  target_shape_strategy: TargetShapeStrategy
  explicit_shape?: [number, number] | null
  padding_method: PaddingMethod
  resize: boolean
  normalize: NormalizeStrategy
  normalize_per_channel: boolean
  align_channels: boolean
  align_reference_index: number
  align_upsample_factor: number
  compensation_matrix?: number[][] | null
}
```

with:

```ts
export interface PreprocessingConfig {
  target_shape_strategy: TargetShapeStrategy
  explicit_shape?: [number, number] | null
  padding_method: PaddingMethod
  resize: boolean
  normalize: NormalizeStrategy
  normalize_per_channel: boolean
  align_channels: boolean
  align_reference_index: number
  align_upsample_factor: number
  compensation_matrix?: number[][] | null
  post_resize_strategy: PostResizeStrategy
  post_resize_value: number | null
}
```

**Edit C** — extend `DEFAULT_PREPROCESSING` (currently lines 67-78). Replace:

```ts
export const DEFAULT_PREPROCESSING: PreprocessingConfig = {
  target_shape_strategy: 'mean',
  explicit_shape: null,
  padding_method: 'poisson',
  resize: false,
  normalize: 'zscore',
  normalize_per_channel: true,
  align_channels: false,
  align_reference_index: 0,
  align_upsample_factor: 10,
  compensation_matrix: null,
}
```

with:

```ts
export const DEFAULT_PREPROCESSING: PreprocessingConfig = {
  target_shape_strategy: 'mean',
  explicit_shape: null,
  padding_method: 'poisson',
  resize: false,
  normalize: 'zscore',
  normalize_per_channel: true,
  align_channels: false,
  align_reference_index: 0,
  align_upsample_factor: 10,
  compensation_matrix: null,
  post_resize_strategy: 'none',
  post_resize_value: null,
}
```

- [ ] **Step 2: Type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no errors. (If errors mention `post_resize_strategy` being missing from object literals, Task 7 will fix the wizard view; until then, type-check may still pass because `DEFAULT_PREPROCESSING` is the only literal that has to populate every field.)

- [ ] **Step 3: Commit**

```bash
git add eidocell-ui/src/types/imports.ts
git commit -m "Add per_image_square types and post-resize defaults"
```

---

## Task 7: Import wizard UI

**Files:**
- Modify: `eidocell-ui/src/views/ImportWizardView.vue` (target-shape block ~L404-444, summary line ~L514)

- [ ] **Step 1: Add the new option to the strategy dropdown**

Find the strategy `<select>` (around line 407). The current options block is:

```html
<select v-model="preproc.target_shape_strategy" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
  <option value="none">No reshaping</option>
  <option value="min">Match smallest image</option>
  <option value="mean">Match average size</option>
  <option value="max">Match largest image</option>
  <option value="explicit">Specify exact shape</option>
</select>
```

Replace with:

```html
<select v-model="preproc.target_shape_strategy" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
  <option value="none">No reshaping</option>
  <option value="min">Match smallest image</option>
  <option value="mean">Match average size</option>
  <option value="max">Match largest image</option>
  <option value="explicit">Specify exact shape</option>
  <option value="per_image_square">Per-image square (pad to longest side)</option>
</select>
```

- [ ] **Step 2: Gate the existing explicit-shape inputs and pad/resize block on the strategy**

The current explicit-shape block reads:

```html
<div v-if="preproc.target_shape_strategy === 'explicit'" class="flex gap-2">
```

No change needed there — `per_image_square` already won't show the H/W inputs.

The current resize/pad toggle (line 431) reads:

```html
<div v-if="preproc.target_shape_strategy !== 'none'">
```

Change it to also hide for `per_image_square`:

```html
<div v-if="preproc.target_shape_strategy !== 'none' && preproc.target_shape_strategy !== 'per_image_square'">
```

- [ ] **Step 3: Add a new per-image-square block**

Immediately AFTER the closing `</div>` of the `<div v-if="preproc.target_shape_strategy !== 'none' && ...">` block from Step 2 (which is the `<!-- Resize vs pad -->` section, ending around line 444), insert this new block:

```html
<!-- Per-image square: padding method + optional global resize -->
<div v-if="preproc.target_shape_strategy === 'per_image_square'" class="space-y-3">
  <div>
    <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Padding method</span></label>
    <select v-model="preproc.padding_method" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
      <option value="constant">Constant (zero)</option>
      <option value="poisson">Poisson noise (recommended for IFC)</option>
      <option value="replicate">Replicate edge</option>
    </select>
  </div>

  <div>
    <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Resize after</span></label>
    <select v-model="preproc.post_resize_strategy" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
      <option value="none">None (keep per-image sizes)</option>
      <option value="explicit">Specify N×N</option>
      <option value="min_longest">Match smallest longest side</option>
      <option value="mean_longest">Match mean longest side</option>
      <option value="max_longest">Match largest longest side</option>
    </select>
  </div>

  <div v-if="preproc.post_resize_strategy === 'explicit'" class="flex items-center gap-2">
    <input
      type="number" min="1" placeholder="N"
      :value="preproc.post_resize_value ?? ''"
      @input="preproc.post_resize_value = Number(($event.target as HTMLInputElement).value) || null"
      class="input input-bordered rounded-[2px] w-32 font-mono text-sm focus:outline-neutral"
    />
    <span class="font-mono text-xs text-neutral/60">× N (square)</span>
  </div>
</div>
```

- [ ] **Step 4: Update the review-step summary line**

Find the preprocessing summary line (around line 514):

```html
{{ preproc.target_shape_strategy }} shape · {{ preproc.resize ? 'resize' : 'pad ' + preproc.padding_method }} ·
{{ preproc.normalize === 'none' ? 'no norm' : preproc.normalize }}
<span v-if="preproc.align_channels"> · align</span>
```

Replace with:

```html
<template v-if="preproc.target_shape_strategy === 'per_image_square'">
  per-image square · pad {{ preproc.padding_method }}
  <template v-if="preproc.post_resize_strategy !== 'none'">
    · resize
    <template v-if="preproc.post_resize_strategy === 'explicit'">{{ preproc.post_resize_value ?? '?' }}×{{ preproc.post_resize_value ?? '?' }}</template>
    <template v-else>{{ preproc.post_resize_strategy.replace('_longest', '') }} longest</template>
  </template>
</template>
<template v-else>
  {{ preproc.target_shape_strategy }} shape · {{ preproc.resize ? 'resize' : 'pad ' + preproc.padding_method }}
</template>
· {{ preproc.normalize === 'none' ? 'no norm' : preproc.normalize }}
<span v-if="preproc.align_channels"> · align</span>
```

- [ ] **Step 5: Type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Manual smoke-test in the browser**

Start the backend and frontend in two terminals:

```bash
# terminal 1
cd eidocell-backend && poetry run uvicorn main:app --reload
```

```bash
# terminal 2
cd eidocell-ui && npm run dev
```

In the Electron/dev window:

1. Create a new session (any name).
2. Open the import wizard, point it at a folder of test images (a few PNGs of varying sizes — the `tmp_path` images from `test_import_service.py` or any handful of differently-shaped images).
3. Enable preprocessing. Confirm the `Target shape` dropdown now contains `Per-image square (pad to longest side)`.
4. Select that option. Verify the H/W explicit inputs and the "Resize (otherwise pad)" checkbox disappear, and the padding-method + "Resize after" controls appear.
5. Set `Resize after` to `Match largest longest side`. Step through to the review screen; confirm the summary reads e.g. `per-image square · pad poisson · resize max longest · zscore`.
6. Submit the import. After it completes, open the gallery and confirm samples render and look correctly square-padded.
7. Repeat step 4–6 with `Resize after = Specify N×N`, value 32. Confirm the summary reads `per-image square · pad poisson · resize 32×32 · zscore`. After import, every sample should be 32×32 in the gallery.

- [ ] **Step 7: Commit**

```bash
git add eidocell-ui/src/views/ImportWizardView.vue
git commit -m "Add per-image square padding controls to import wizard"
```

---

## Task 8: Final full-test sweep

- [ ] **Step 1: Run the entire backend test suite**

```bash
cd eidocell-backend && poetry run pytest
```

Expected: all tests PASS.

- [ ] **Step 2: Run frontend type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Confirm clean working tree**

```bash
git status
```

Expected: `working tree clean` (all task commits already landed).

---

## Self-review notes

- All spec sections (architecture, schema, backend op, pipeline, shape summary, frontend types, frontend UI, tests, back-compat) are covered by Tasks 1–7; Task 8 is a final verification sweep.
- No placeholders, TBDs, or "handle edge cases" steps — every step contains exact code or exact commands.
- Type names are consistent across tasks: `pad_to_square` (op), `PostResizeStrategy` (type), `post_resize_strategy` / `post_resize_value` (fields), `min_longest` / `max_longest` / `mean_longest` (summary keys).
- The schema test in Task 3 (`PreprocessingConfig.model_validate(...)`) intentionally tests both new field presence and back-compat of pre-existing fields.
- The pipeline-builder negative test (`shape_summary missing`) is covered by `test_pipeline_per_image_square_dataset_stat_without_summary_raises`.
