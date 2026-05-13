# Mask-toggle gating and axis rebind — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disable the gallery MASK VIEW toggle when the active session has zero extracted masks, and rebind gate axis parameters positionally when a plot's x/y variable changes (FCS Express semantics), with a dismissible REBOUND badge in the population tree.

**Architecture:**
- Item 1 piggybacks a `session_has_any_mask: bool` aggregate on the existing `SamplePage` response. The FilterBar reads this through the gallery Pinia store and renders the button as `disabled` with a tooltip when false. No new endpoint.
- Item 2 adds a `Gate.rebound_at` timestamp column (via STORAGE_VERSION bump — no migration code) and a backend `_rebind_plot_gates` helper called from `update_plot`. The frontend renders a dismissible REBOUND chip in `PopulationTreeNode.vue` and exposes a `clear_rebound` flag on `GateUpdate`.

**Tech Stack:**
- Backend: FastAPI, SQLAlchemy 2.0, SQLite, pytest.
- Frontend: Vue 3 `<script setup>`, Pinia, Tailwind + DaisyUI, TypeScript.
- Repository conventions: `routers/` → `services/` → `models/`; `Base.metadata.create_all()` + STORAGE_VERSION-wipe policy (no Alembic).

---

## File Structure

**Backend — modify:**
- `eidocell-backend/core/config.py` — bump `STORAGE_VERSION` (7 → 8).
- `eidocell-backend/models/models.py` — add `Gate.rebound_at: Mapped[datetime | None]`.
- `eidocell-backend/schemas/workspace/gallery.py` — add `SamplePage.session_has_any_mask: bool`.
- `eidocell-backend/schemas/workspace/analysis.py` — add `GateOut.rebound_at`, `GateUpdate.clear_rebound`.
- `eidocell-backend/services/workspace/gallery_service.py` — `list_samples` returns `(items, total, session_has_any_mask)`.
- `eidocell-backend/routers/workspace/gallery.py` — unpack new tuple, pass into `SamplePage`.
- `eidocell-backend/services/workspace/analysis_service.py` — new `_rebind_plot_gates`, extend `update_plot`, extend `update_gate` with `clear_rebound`, include `rebound_at` in `_gate_to_out`.
- `eidocell-backend/routers/workspace/analysis.py` — thread `clear_rebound` from `GateUpdate` into the service call.

**Backend — tests:**
- `eidocell-backend/tests/test_gallery.py` — new tests for `session_has_any_mask`.
- `eidocell-backend/tests/test_analysis.py` — new tests for rebind + `clear_rebound`.

**Frontend — modify:**
- `eidocell-ui/src/types/gallery.ts` — `SamplePage.session_has_any_mask`.
- `eidocell-ui/src/types/analysis.ts` — `GateOut.rebound_at`, `GateUpdate.clear_rebound`.
- `eidocell-ui/src/stores/gallery.ts` — new `sessionHasAnyMask` ref, populate on fetch.
- `eidocell-ui/src/views/workspace/GalleryView.vue` — pass `mask-available` prop to FilterBar.
- `eidocell-ui/src/components/gallery/FilterBar.vue` — accept `maskAvailable`, render disabled state + tooltip, force-off when flips false.
- `eidocell-ui/src/components/analysis/PopulationTreeNode.vue` — render dismissible REBOUND chip.

---

## Item 1 — Mask toggle gating

### Task 1: Backend test — `session_has_any_mask` aggregate

**Files:**
- Test: `eidocell-backend/tests/test_gallery.py`

- [ ] **Step 1: Inspect existing test patterns**

Read the top of `eidocell-backend/tests/test_gallery.py` to see fixtures used (e.g. how a session is created, how segmentation is run). Look at `test_segmentation.py:209` (`test_gallery_has_mask_after_segmentation`) for the standard "create session, run segmentation, verify masks present" pattern.

- [ ] **Step 2: Add the failing test**

Append the following to `eidocell-backend/tests/test_gallery.py`:

```python
def test_samples_page_session_has_any_mask_false_when_no_masks(client, tmp_path):
    """SamplePage.session_has_any_mask is False on a fresh session with no segmentation."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    from PIL import Image
    for i in range(3):
        Image.new("RGB", (32, 32)).save(img_dir / f"x_{i}.png")
    session = client.post("/sessions/", json={
        "name": "no-masks", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]

    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "offset": 0, "limit": 100, "sort_by": "filename", "sort_order": "asc",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_has_any_mask"] is False


def test_samples_page_session_has_any_mask_true_after_segmentation(client, tmp_path):
    """After segmentation creates at least one Mask row, the flag becomes True."""
    from PIL import Image
    import numpy as np
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[8:24, 8:24] = 200
        Image.fromarray(arr).save(img_dir / f"x_{i}.png")
    session = client.post("/sessions/", json={
        "name": "with-masks", "images_directory": str(img_dir),
    }).json()
    sid = session["id"]
    client.post(f"/sessions/{sid}/segmentation/run", json={
        "method": "otsu_intensity",
        "params": {"distance_from_center": 80, "min_component_size": 10},
    })

    resp = client.post(f"/sessions/{sid}/samples/list", json={
        "offset": 0, "limit": 100, "sort_by": "filename", "sort_order": "asc",
    })
    assert resp.status_code == 200
    assert resp.json()["session_has_any_mask"] is True
```

- [ ] **Step 3: Run the tests and verify they fail**

```bash
cd eidocell-backend && poetry run pytest tests/test_gallery.py::test_samples_page_session_has_any_mask_false_when_no_masks tests/test_gallery.py::test_samples_page_session_has_any_mask_true_after_segmentation -v
```

Expected: FAIL — `KeyError: 'session_has_any_mask'` (or `assert None is False`) because the schema field doesn't exist yet.

- [ ] **Step 4: Commit**

```bash
git add eidocell-backend/tests/test_gallery.py
git commit -m "test(gallery): add session_has_any_mask aggregate tests"
```

---

### Task 2: Backend implementation — `session_has_any_mask` aggregate

**Files:**
- Modify: `eidocell-backend/schemas/workspace/gallery.py:21-26` (SamplePage)
- Modify: `eidocell-backend/services/workspace/gallery_service.py:30-101` (list_samples signature + return)
- Modify: `eidocell-backend/routers/workspace/gallery.py:24-39` (unpack new tuple)

- [ ] **Step 1: Add field to `SamplePage` schema**

Edit `eidocell-backend/schemas/workspace/gallery.py`:

```python
class SamplePage(BaseModel):
    items: list[SampleOut]
    total: int
    offset: int
    limit: int
    session_has_any_mask: bool = False
```

- [ ] **Step 2: Compute the flag in `list_samples`**

Edit `eidocell-backend/services/workspace/gallery_service.py`. Change the signature (line 30) from:

```python
def list_samples(db: DbSession, session_id: str, params: SampleListParams) -> tuple[list[dict], int]:
```

to:

```python
def list_samples(
    db: DbSession, session_id: str, params: SampleListParams
) -> tuple[list[dict], int, bool]:
```

Compute `session_has_any_mask` early in the function (immediately after the docstring, before the `base_query = ...` block):

```python
    session_has_any_mask = (
        db.query(Mask.id)
        .join(Sample, Mask.sample_id == Sample.id)
        .filter(Sample.session_id == session_id)
        .first()
        is not None
    )
```

Update **both** return statements:

- Line 49 `return [], 0` becomes `return [], 0, session_has_any_mask`.
- Line 101 (end-of-function) `return items, total` becomes `return items, total, session_has_any_mask`.

- [ ] **Step 3: Update the router to thread the flag through**

Edit `eidocell-backend/routers/workspace/gallery.py`:

```python
@router.post("/samples/list", response_model=SamplePage)
def list_samples(
    session_id: str,
    params: SampleListParams,
    db: DbSession = Depends(get_db),
):
    """List samples with filtering, sorting, and pagination.
    Uses POST because filter conditions can be complex."""
    session_service.get_session(db, session_id)
    items, total, session_has_any_mask = gallery_service.list_samples(db, session_id, params)
    return SamplePage(
        items=items,
        total=total,
        offset=params.offset,
        limit=params.limit,
        session_has_any_mask=session_has_any_mask,
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd eidocell-backend && poetry run pytest tests/test_gallery.py::test_samples_page_session_has_any_mask_false_when_no_masks tests/test_gallery.py::test_samples_page_session_has_any_mask_true_after_segmentation -v
```

Expected: PASS, both tests.

- [ ] **Step 5: Run the full gallery test module to confirm no regressions**

```bash
cd eidocell-backend && poetry run pytest tests/test_gallery.py -v
```

Expected: all tests in the module PASS.

- [ ] **Step 6: Commit**

```bash
git add eidocell-backend/schemas/workspace/gallery.py eidocell-backend/services/workspace/gallery_service.py eidocell-backend/routers/workspace/gallery.py
git commit -m "feat(gallery): expose session_has_any_mask on SamplePage"
```

---

### Task 3: Frontend — types + store

**Files:**
- Modify: `eidocell-ui/src/types/gallery.ts:14-19`
- Modify: `eidocell-ui/src/stores/gallery.ts`

- [ ] **Step 1: Add the field to the TypeScript type**

Edit `eidocell-ui/src/types/gallery.ts`:

```typescript
export interface SamplePage {
  items: SampleOut[]
  total: number
  offset: number
  limit: number
  session_has_any_mask: boolean
}
```

- [ ] **Step 2: Add `sessionHasAnyMask` to the store**

Edit `eidocell-ui/src/stores/gallery.ts`. In the state block near `const maskVersion = ref(0)`, add:

```typescript
const sessionHasAnyMask = ref(false)
```

In `$reset()`, add:

```typescript
sessionHasAnyMask.value = false
```

In `fetchSamples()`, after the `samples.value = page.items` line, add:

```typescript
sessionHasAnyMask.value = page.session_has_any_mask
```

In `loadNextPage()`, after `total.value = page.total`, add the same line so pagination keeps the flag fresh:

```typescript
sessionHasAnyMask.value = page.session_has_any_mask
```

Add `sessionHasAnyMask` to the `return` object:

```typescript
  return {
    samples, total, offset, limit, sortBy, sortOrder, filters,
    sessionHasAnyMask,
    maskVersion, inspectMode, openSimilarityDialog,
    // ... rest unchanged
```

- [ ] **Step 3: Type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no new errors. (Existing errors, if any, are out of scope.)

- [ ] **Step 4: Commit**

```bash
git add eidocell-ui/src/types/gallery.ts eidocell-ui/src/stores/gallery.ts
git commit -m "feat(gallery-ui): plumb session_has_any_mask through store"
```

---

### Task 4: Frontend — FilterBar disabled state

**Files:**
- Modify: `eidocell-ui/src/views/workspace/GalleryView.vue` (pass prop)
- Modify: `eidocell-ui/src/components/gallery/FilterBar.vue` (accept prop, render disabled, force-off on flip)

- [ ] **Step 1: Pass the prop in `GalleryView.vue`**

Edit `eidocell-ui/src/views/workspace/GalleryView.vue`. The existing FilterBar binding at lines 97-104 becomes:

```vue
        <FilterBar
          :zoom-level="zoomLevel"
          :mask-view="maskViewEnabled"
          :mask-available="gallery.sessionHasAnyMask"
          :inspect-mode="gallery.inspectMode"
          @update:zoom-level="zoomLevel = $event"
          @update:mask-view="maskViewEnabled = $event"
          @update:inspect-mode="gallery.inspectMode = $event"
        />
```

- [ ] **Step 2: Accept the prop and render disabled state in `FilterBar.vue`**

Edit `eidocell-ui/src/components/gallery/FilterBar.vue`. Extend the `defineProps` interface (around line 9):

```typescript
const props = defineProps<{
  zoomLevel: number
  maskView: boolean
  maskAvailable: boolean
  inspectMode: boolean
}>()
```

At the top of the `<script setup>`, after the `emit` declaration, add the force-off watch and a `watch` import:

```typescript
import { computed, ref, watch } from 'vue'
// ... existing imports

// If masks disappear (e.g. session change), turn the toggle off so the
// SampleCardGrid doesn't render in a half-broken state.
watch(() => props.maskAvailable, (available) => {
  if (!available && props.maskView) {
    emit('update:maskView', false)
  }
})
```

(The existing `import { computed, ref } from 'vue'` line at the top already exists — replace it with `import { computed, ref, watch } from 'vue'`.)

Replace the existing MASK VIEW button (lines 112-120) with a wrapped, disabled-aware version:

```vue
    <!-- MASK VIEW toggle -->
    <div
      class="tooltip tooltip-bottom"
      :data-tip="props.maskAvailable ? undefined : 'No masks extracted yet — run segmentation first'"
    >
      <button
        class="h-8 px-4 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity duration-200"
        :class="[
          !props.maskAvailable
            ? 'opacity-30 cursor-not-allowed'
            : (props.maskView ? 'opacity-100' : 'opacity-50 hover:opacity-80'),
        ]"
        :disabled="!props.maskAvailable"
        @click="props.maskAvailable && emit('update:maskView', !props.maskView)"
      >
        <component :is="props.maskView && props.maskAvailable ? Eye : EyeOff" class="w-4 h-4" />
        MASK VIEW
      </button>
    </div>
```

- [ ] **Step 3: Type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no new errors.

- [ ] **Step 4: Manual smoke test**

Start backend and frontend:

```bash
cd eidocell-backend && poetry run uvicorn main:app --reload &
cd eidocell-ui && npm run dev
```

Manually verify, in a browser:
1. Create a new session, import a small folder of images, wait for previews.
2. On Gallery view, MASK VIEW button is greyed/disabled and hovering shows the tooltip.
3. Run segmentation; after completion, navigate back to Gallery (or it refetches automatically).
4. MASK VIEW button is now enabled and toggles correctly.

- [ ] **Step 5: Commit**

```bash
git add eidocell-ui/src/views/workspace/GalleryView.vue eidocell-ui/src/components/gallery/FilterBar.vue
git commit -m "feat(gallery-ui): disable MASK VIEW toggle when session has no masks"
```

---

## Item 2 — Axis rebind on plot parameter change

### Task 5: Backend — schema column + storage version bump

**Files:**
- Modify: `eidocell-backend/core/config.py:6`
- Modify: `eidocell-backend/models/models.py` (Gate class, around lines 171-198)

- [ ] **Step 1: Bump STORAGE_VERSION**

Edit `eidocell-backend/core/config.py`:

```python
STORAGE_VERSION = 8
```

(The existing value is 7. This is the wipe-on-bump trigger per the project's storage policy.)

- [ ] **Step 2: Add `rebound_at` to the `Gate` model**

Edit `eidocell-backend/models/models.py`. Inside `class Gate(Base):`, after the `source_gate_ids` line (around line 189), add:

```python
    rebound_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

(The class already imports `datetime` and `DateTime` — confirm at the top of the file.)

- [ ] **Step 3: Commit**

```bash
git add eidocell-backend/core/config.py eidocell-backend/models/models.py
git commit -m "feat(analysis): add Gate.rebound_at column (STORAGE_VERSION bump)"
```

---

### Task 6: Backend test — rebind on axis change

**Files:**
- Test: `eidocell-backend/tests/test_analysis.py`

- [ ] **Step 1: Add failing tests**

Append the following at the end of `eidocell-backend/tests/test_analysis.py`:

```python
# ── Axis rebind ─────────────────────────────────────────────────────────


def test_axis_rebind_histogram_updates_gate_parameters(client, session_with_masks):
    """Changing histogram x_variable rebinds parameters[0] and stamps rebound_at."""
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram",
        "parameters": {"x_variable": "area", "num_bins": 20},
    }).json()
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval",
        "definition": {"min": 0.0, "max": 1e9},
        "parameters": ["area"],
    }).json()
    assert gate["rebound_at"] is None

    client.patch(f"/sessions/{sid}/analysis/plots/{plot['id']}", json={
        "parameters": {"x_variable": "mean_intensity", "num_bins": 20},
    })

    refreshed = next(g for g in client.get(f"/sessions/{sid}/analysis/gates").json()
                     if g["id"] == gate["id"])
    assert refreshed["parameters"] == ["mean_intensity"]
    assert refreshed["definition"] == {"min": 0.0, "max": 1e9}
    assert refreshed["rebound_at"] is not None


def test_axis_rebind_scatter_xy(client, session_with_masks):
    """Changing both axes on a scatter plot rebinds both gate slots."""
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "mean_intensity"},
    }).json()
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "rectangular",
        "definition": {"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
        "parameters": ["area", "mean_intensity"],
    }).json()

    client.patch(f"/sessions/{sid}/analysis/plots/{plot['id']}", json={
        "parameters": {"x_variable": "solidity", "y_variable": "eccentricity"},
    })

    refreshed = next(g for g in client.get(f"/sessions/{sid}/analysis/gates").json()
                     if g["id"] == gate["id"])
    assert refreshed["parameters"] == ["solidity", "eccentricity"]
    assert refreshed["rebound_at"] is not None


def test_axis_rebind_only_x(client, session_with_masks):
    """Changing only x leaves y slot untouched."""
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "scatter",
        "parameters": {"x_variable": "area", "y_variable": "mean_intensity"},
    }).json()
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "rectangular",
        "definition": {"x": 0.0, "y": 0.0, "width": 50.0, "height": 50.0},
        "parameters": ["area", "mean_intensity"],
    }).json()

    client.patch(f"/sessions/{sid}/analysis/plots/{plot['id']}", json={
        "parameters": {"x_variable": "solidity", "y_variable": "mean_intensity"},
    })

    refreshed = next(g for g in client.get(f"/sessions/{sid}/analysis/gates").json()
                     if g["id"] == gate["id"])
    assert refreshed["parameters"] == ["solidity", "mean_intensity"]


def test_axis_rebind_noop_when_axes_unchanged(client, session_with_masks):
    """Updating non-axis params (e.g. num_bins) does NOT stamp rebound_at."""
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram",
        "parameters": {"x_variable": "area", "num_bins": 20},
    }).json()
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval",
        "definition": {"min": 0.0, "max": 1e9},
        "parameters": ["area"],
    }).json()

    client.patch(f"/sessions/{sid}/analysis/plots/{plot['id']}", json={
        "parameters": {"x_variable": "area", "num_bins": 50},
    })

    refreshed = next(g for g in client.get(f"/sessions/{sid}/analysis/gates").json()
                     if g["id"] == gate["id"])
    assert refreshed["parameters"] == ["area"]
    assert refreshed["rebound_at"] is None


def test_axis_rebind_skips_boolean_and_other_plots(client, session_with_masks):
    """Boolean gates and gates on a different plot are unaffected by axis change."""
    sid = session_with_masks["id"]
    plot_a = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram",
        "parameters": {"x_variable": "area", "num_bins": 20},
    }).json()
    plot_b = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram",
        "parameters": {"x_variable": "mean_intensity", "num_bins": 20},
    }).json()
    gate_a = client.post(f"/sessions/{sid}/analysis/plots/{plot_a['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0.0, "max": 10.0},
        "parameters": ["area"],
    }).json()
    gate_b = client.post(f"/sessions/{sid}/analysis/plots/{plot_b['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0.0, "max": 10.0},
        "parameters": ["mean_intensity"],
    }).json()
    boolean = client.post(f"/sessions/{sid}/analysis/boolean-gates", json={
        "name": "A AND B", "operator": "AND",
        "source_gate_ids": [gate_a["id"], gate_b["id"]],
    }).json()

    client.patch(f"/sessions/{sid}/analysis/plots/{plot_a['id']}", json={
        "parameters": {"x_variable": "solidity", "num_bins": 20},
    })

    gates = {g["id"]: g for g in client.get(f"/sessions/{sid}/analysis/gates").json()}
    assert gates[gate_a["id"]]["parameters"] == ["solidity"]
    assert gates[gate_a["id"]]["rebound_at"] is not None
    assert gates[gate_b["id"]]["parameters"] == ["mean_intensity"]
    assert gates[gate_b["id"]]["rebound_at"] is None
    assert gates[boolean["id"]]["parameters"] == []
    assert gates[boolean["id"]]["rebound_at"] is None


def test_clear_rebound(client, session_with_masks):
    """update_gate with clear_rebound=True nulls rebound_at."""
    sid = session_with_masks["id"]
    plot = client.post(f"/sessions/{sid}/analysis/plots", json={
        "chart_type": "histogram",
        "parameters": {"x_variable": "area", "num_bins": 20},
    }).json()
    gate = client.post(f"/sessions/{sid}/analysis/plots/{plot['id']}/gates", json={
        "gate_type": "interval", "definition": {"min": 0.0, "max": 10.0},
        "parameters": ["area"],
    }).json()
    client.patch(f"/sessions/{sid}/analysis/plots/{plot['id']}", json={
        "parameters": {"x_variable": "mean_intensity", "num_bins": 20},
    })

    resp = client.patch(f"/sessions/{sid}/analysis/gates/{gate['id']}", json={
        "clear_rebound": True,
    })
    assert resp.status_code == 200
    assert resp.json()["rebound_at"] is None
```

- [ ] **Step 2: Verify tests fail**

```bash
cd eidocell-backend && poetry run pytest tests/test_analysis.py -k "axis_rebind or clear_rebound" -v
```

Expected: FAIL — either `KeyError: 'rebound_at'`, schema/field-missing errors, or assertions that gate.parameters never change.

- [ ] **Step 3: Commit**

```bash
git add eidocell-backend/tests/test_analysis.py
git commit -m "test(analysis): axis rebind and clear_rebound behavior"
```

---

### Task 7: Backend — schemas + service rebind logic

**Files:**
- Modify: `eidocell-backend/schemas/workspace/analysis.py` (GateOut, GateUpdate)
- Modify: `eidocell-backend/services/workspace/analysis_service.py` (update_plot, _rebind_plot_gates, update_gate, _gate_to_out)
- Modify: `eidocell-backend/routers/workspace/analysis.py` (thread clear_rebound)

- [ ] **Step 1: Extend `GateOut` and `GateUpdate` schemas**

Edit `eidocell-backend/schemas/workspace/analysis.py`. Update `GateUpdate` (around lines 146-152):

```python
class GateUpdate(BaseModel):
    name: str | None = None
    color: str | None = None
    definition: dict | None = None
    parent_gate_id: str | None = None
    clear_rebound: bool = False
```

Update `GateOut` (around lines 174-188):

```python
class GateOut(BaseModel):
    id: str
    plot_id: str | None = None
    name: str
    gate_type: str
    definition: dict
    color: str
    parameters: list[str]
    sample_count: int = 0
    percentage: float = 0.0
    parent_gate_id: str | None = None
    operator: str | None = None
    source_gate_ids: list[str] | None = None
    rebound_at: datetime | None = None

    model_config = {"from_attributes": True}
```

(`datetime` is already imported at the top of the file.)

- [ ] **Step 2: Add `_rebind_plot_gates` and extend `update_plot` in the service**

Edit `eidocell-backend/services/workspace/analysis_service.py`. At the top of the file, ensure `datetime` is imported:

```python
from datetime import datetime
```

(If a `datetime` import is already present, leave it.)

Replace the existing `update_plot` (lines 88-115) with:

```python
def update_plot(
    db: DbSession, plot_id: str,
    name: str | None = None,
    parameters: dict | None = None,
) -> dict:
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    if name is not None:
        plot.name = name

    if parameters is not None:
        if plot.chart_type == "histogram":
            if "x_variable" not in parameters:
                raise HTTPException(status_code=400, detail="Histogram requires x_variable")
        else:
            if "x_variable" not in parameters or "y_variable" not in parameters:
                raise HTTPException(
                    status_code=400,
                    detail=f"{plot.chart_type.title()} requires x_variable and y_variable",
                )
        _rebind_plot_gates(db, plot, plot.parameters or {}, parameters)
        plot.parameters = parameters

    db.commit()
    db.refresh(plot)
    _update_active_samples(db, plot.session_id)
    gate_count = db.query(func.count(Gate.id)).filter(Gate.plot_id == plot.id).scalar()
    return _plot_to_out(plot, gate_count)


def _rebind_plot_gates(
    db: DbSession, plot: Plot, old_params: dict, new_params: dict
) -> None:
    """Positionally rebind a plot's geometric gates when axes change.

    FCS Express semantics: gate.definition coordinates are preserved verbatim; only
    the parameter names referenced by each slot are rewritten to the plot's new
    axes. Boolean gates and gates on other plots are untouched.
    """
    new_x = new_params.get("x_variable")
    new_y = new_params.get("y_variable")
    old_x = old_params.get("x_variable")
    old_y = old_params.get("y_variable")
    if new_x == old_x and new_y == old_y:
        return

    gates = (
        db.query(Gate)
        .filter(Gate.plot_id == plot.id, Gate.gate_type != "boolean")
        .all()
    )
    is_1d = plot.chart_type == "histogram"
    stamp = datetime.utcnow()
    for g in gates:
        params = list(g.parameters or [])
        if is_1d:
            if new_x and len(params) >= 1:
                params[0] = new_x
        else:
            if new_x and len(params) >= 1:
                params[0] = new_x
            if new_y and len(params) >= 2:
                params[1] = new_y
        g.parameters = params
        g.rebound_at = stamp
```

- [ ] **Step 3: Extend `update_gate` with `clear_rebound`**

Edit `eidocell-backend/services/workspace/analysis_service.py`. Change the `update_gate` signature (around line 425):

```python
def update_gate(
    db: DbSession,
    gate_id: str,
    name: str | None,
    color: str | None,
    definition: dict | None,
    parent_gate_id: str | None = None,
    update_parent: bool = False,
    clear_rebound: bool = False,
) -> dict:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    if name is not None:
        gate.name = name
    if color is not None:
        gate.color = color
    # ... existing definition + update_parent blocks UNCHANGED ...

    if clear_rebound:
        gate.rebound_at = None

    db.commit()
    db.refresh(gate)
    pop = _compute_gate_population(db, gate)
    total = _session_total_samples(db, gate.session_id)
    _update_active_samples(db, gate.session_id)
    return _gate_to_out(gate, len(pop), total)
```

(Only the signature and the new `if clear_rebound:` block are added — leave the existing `if definition is not None:` and `if update_parent:` blocks exactly as they are.)

- [ ] **Step 4: Include `rebound_at` in `_gate_to_out`**

Replace `_gate_to_out` (around lines 535-549):

```python
def _gate_to_out(gate: Gate, sample_count: int, total: int) -> dict:
    return {
        "id": gate.id,
        "plot_id": gate.plot_id,
        "name": gate.name,
        "gate_type": gate.gate_type,
        "definition": gate.definition,
        "color": gate.color,
        "parameters": gate.parameters,
        "sample_count": sample_count,
        "percentage": (sample_count / total * 100) if total > 0 else 0,
        "parent_gate_id": gate.parent_gate_id,
        "operator": gate.operator,
        "source_gate_ids": gate.source_gate_ids,
        "rebound_at": gate.rebound_at,
    }
```

- [ ] **Step 5: Thread `clear_rebound` through the router**

Edit `eidocell-backend/routers/workspace/analysis.py:148-154`. Replace the existing handler with:

```python
@router.patch("/gates/{gate_id}", response_model=GateOut)
def update_gate(session_id: str, gate_id: str, data: GateUpdate, db: DbSession = Depends(get_db)):
    return analysis_service.update_gate(
        db, gate_id, data.name, data.color, data.definition,
        parent_gate_id=data.parent_gate_id,
        update_parent="parent_gate_id" in data.model_fields_set,
        clear_rebound=data.clear_rebound,
    )
```

- [ ] **Step 6: Run the rebind tests and verify they pass**

```bash
cd eidocell-backend && poetry run pytest tests/test_analysis.py -k "axis_rebind or clear_rebound" -v
```

Expected: PASS, all six tests.

- [ ] **Step 7: Run the full analysis test module to confirm no regressions**

```bash
cd eidocell-backend && poetry run pytest tests/test_analysis.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add eidocell-backend/schemas/workspace/analysis.py eidocell-backend/services/workspace/analysis_service.py eidocell-backend/routers/workspace/analysis.py
git commit -m "feat(analysis): rebind plot gates positionally on axis change"
```

---

### Task 8: Frontend — types

**Files:**
- Modify: `eidocell-ui/src/types/analysis.ts`

- [ ] **Step 1: Add `rebound_at` to `GateOut`**

Edit `eidocell-ui/src/types/analysis.ts`. Update `GateOut`:

```typescript
export interface GateOut {
  id: string
  plot_id: string | null
  name: string
  gate_type: GateType
  definition: Record<string, unknown>
  color: string
  parameters: string[]
  sample_count: number
  percentage: number
  parent_gate_id?: string | null
  operator?: BooleanOperator | null
  source_gate_ids?: string[] | null
  rebound_at?: string | null
}
```

Update `GateUpdate`:

```typescript
export interface GateUpdate {
  name?: string
  color?: string
  definition?: Record<string, unknown>
  parent_gate_id?: string | null
  clear_rebound?: boolean
}
```

- [ ] **Step 2: Type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add eidocell-ui/src/types/analysis.ts
git commit -m "feat(analysis-ui): add rebound_at + clear_rebound types"
```

---

### Task 9: Frontend — REBOUND chip in PopulationTreeNode

**Files:**
- Modify: `eidocell-ui/src/components/analysis/PopulationTreeNode.vue`

- [ ] **Step 1: Add an emit for clearing rebound and a click handler**

Edit `eidocell-ui/src/components/analysis/PopulationTreeNode.vue`. Extend the `defineEmits` block (around lines 20-28):

```typescript
const emit = defineEmits<{
  toggleExpand: [id: string]
  select: [id: string]
  delete: [id: string]
  dragStart: [id: string]
  dragEnd: []
  reparent: [sourceId: string, newParentId: string | null]
  contextmenu: [id: string, name: string, e: MouseEvent]
  clearRebound: [id: string]
}>()
```

Add a computed near the existing computeds (around line 33):

```typescript
const isRebound = computed(() => Boolean(props.node.rebound_at))
```

- [ ] **Step 2: Render the chip in the row**

In the `<template>`, immediately after the gate name `<span>` that ends with `</span>` for the boolean operator badge (around lines 168-175 — the one ending with `</span>` after the operator chip), insert a new REBOUND chip:

```vue
      <!-- Rebound badge: appears when an axis change retargeted this gate's params. -->
      <button
        v-if="isRebound && !isRoot && !isBoolean"
        class="ml-1 px-1 h-3.5 flex items-center text-[8px] font-bold tracking-widest rounded-[2px] bg-amber-100 text-amber-700 hover:bg-amber-200 transition-colors shrink-0"
        title="Gate axes were rebound after a plot parameter change. Click to dismiss."
        @click.stop="emit('clearRebound', node.id)"
      >REBOUND</button>
```

- [ ] **Step 3: Forward the new emit through the recursive child binding**

In the recursive `<PopulationTreeNode>` block (around lines 198-216), add the new event forwarder:

```vue
        <PopulationTreeNode
          v-for="(child, idx) in node.children"
          :key="child.id"
          :node="child"
          :depth="depth + 1"
          :is-last="idx === node.children.length - 1"
          :expanded-ids="expandedIds"
          :plot-names="plotNames"
          :dragging-id="draggingId"
          :dragging-descendants="draggingDescendants"
          :selected-id="selectedId"
          @toggle-expand="(id) => emit('toggleExpand', id)"
          @select="(id) => emit('select', id)"
          @delete="(id) => emit('delete', id)"
          @drag-start="(id) => emit('dragStart', id)"
          @drag-end="emit('dragEnd')"
          @reparent="(s, p) => emit('reparent', s, p)"
          @contextmenu="(id, name, e) => emit('contextmenu', id, name, e)"
          @clear-rebound="(id) => emit('clearRebound', id)"
        />
```

- [ ] **Step 4: Handle the emit at the PopulationTree level**

Open `eidocell-ui/src/components/analysis/PopulationTree.vue`. `useAnalysisStore` and `analysis` are already imported (verified at lines 4–10).

Add this handler in the `<script setup>` near the other gate handlers (e.g. after the existing `analysis.updateGate(...)` rename handler around line 32):

```typescript
async function onClearRebound(gateId: string) {
  await analysis.updateGate(gateId, { clear_rebound: true })
}
```

The template has **two** `<PopulationTreeNode>` instances (hierarchical tree around line 147, and the boolean flat list around line 173). Add `@clear-rebound="onClearRebound"` to **both**, immediately after the existing `@contextmenu="onNodeContextmenu"` line:

```vue
        @contextmenu="onNodeContextmenu"
        @clear-rebound="onClearRebound"
```

- [ ] **Step 5: Type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no new errors.

- [ ] **Step 6: Manual smoke test**

With backend + frontend running:
1. Create a session, run segmentation.
2. Open Analysis view, create a histogram plot on `area`, draw an interval gate.
3. Open PlotSettings, change x to `mean_intensity`, apply.
4. Verify: the gate's REBOUND chip appears next to its name in PopulationTree, and the gate's parameters in the API response (open devtools) show `mean_intensity` (the new axis).
5. Click the REBOUND chip → it disappears.

- [ ] **Step 7: Commit**

```bash
git add eidocell-ui/src/components/analysis/PopulationTreeNode.vue eidocell-ui/src/components/analysis/PopulationTree.vue
git commit -m "feat(analysis-ui): REBOUND chip on rebound gates with click-to-dismiss"
```

---

## Verification (final)

- [ ] **Step 1: Full backend test suite**

```bash
cd eidocell-backend && poetry run pytest
```

Expected: all tests PASS. New tests for `session_has_any_mask`, `axis_rebind_*`, and `clear_rebound` are green; nothing else regressed.

- [ ] **Step 2: Frontend type-check**

```bash
cd eidocell-ui && npx vue-tsc --noEmit
```

Expected: no new errors.

- [ ] **Step 3: Frontend build**

```bash
cd eidocell-ui && npm run build
```

Expected: clean build.

- [ ] **Step 4: End-to-end manual run**

Start backend (which will wipe `~/.eidocell` on first start because STORAGE_VERSION bumped) and frontend. In the browser:

- **Item 1:** Create a session with images, do not run segmentation → MASK VIEW disabled with tooltip. Run segmentation → MASK VIEW enables.
- **Item 2:** Create a 2D scatter plot, draw a rectangular gate, change both axes via PlotSettings → gate's params update; REBOUND chip appears in PopulationTree. Click chip → chip disappears, gate's `rebound_at` cleared.
