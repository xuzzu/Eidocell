# Mask-toggle gating and gate rebind on plot axis change

## Scope

Two unrelated fixes bundled because each is small and they touch independent code paths:

1. **MASK VIEW toggle**: disable the FilterBar toggle when the active session has zero extracted masks; show a tooltip explaining why.
2. **Axis rebind**: when a plot's `x_variable` or `y_variable` changes, rebind that plot's gates positionally (the gate's `parameters[]` slot gets the new axis name; coordinates stay verbatim). A persistent "rebound" badge surfaces on each rebound gate until the user dismisses it.

A third item ("gated populations as training data") was discussed and explicitly cut from this spec.

## Item 1 — Mask toggle gating

### Current state

`eidocell-ui/src/components/gallery/FilterBar.vue:113` renders an always-enabled MASK VIEW button. Each `SampleOut` carries a per-sample `has_mask: bool` already, but the toggle has no session-level awareness. If the session has zero masks, clicking the toggle does nothing visible (the per-sample `has_mask` check in `SampleCardGrid.vue:74` filters everything out).

### Behavior

When the active session has zero masks:
- The MASK VIEW button renders disabled (greyed, non-interactive cursor).
- Hovering shows a tooltip: `No masks extracted yet — run segmentation first`.
- When mask state changes (e.g. segmentation finishes), the disabled state lifts on the next gallery refresh — no manual page reload required.

When at least one mask exists, behavior is unchanged from today.

### Signal source

Add a single `session_has_any_mask: bool` field to the existing gallery samples list response (`SamplesPage` in `schemas/workspace/gallery.py`). Computed once per request as:

```python
session_has_any_mask = db.query(
    exists().where(
        Mask.sample_id == Sample.id,
        Sample.session_id == session_id,
    )
).scalar()
```

This avoids a second round-trip, piggybacks on a call that's already made on session load and after every filter/sort/segmentation completion, and is reactive through existing flows.

### Frontend wiring

- `eidocell-ui/src/stores/gallery.ts` — store the boolean from the page response in a new `sessionHasAnyMask: Ref<boolean>`. Reset to `false` on `$reset()` and on fresh `fetchSamples()`.
- `eidocell-ui/src/views/workspace/GalleryView.vue` — pass `gallery.sessionHasAnyMask` to FilterBar as a new prop `maskAvailable`.
- `eidocell-ui/src/components/gallery/FilterBar.vue` — accept `maskAvailable`. When false:
  - Add `disabled` to the button.
  - Set `cursor-not-allowed`, drop the opacity to a clearly-disabled state.
  - Wrap in DaisyUI tooltip `tooltip-bottom` with `data-tip="No masks extracted yet — run segmentation first"`.
  - If `maskView` is currently true and `maskAvailable` flips to false, emit `update:maskView` with `false` once so the parent doesn't render a broken on-state.

### Out of scope

- Per-channel mask availability (mask exists for some channels but not others). The toggle stays a session-level on/off.
- Mask presence on Classes/Clusters views — those have their own `has_mask` paths and aren't gated by this toggle today.

## Item 2 — Axis rebind on plot parameter change

### Current state

`analysis_service.update_plot` (lines 88–115) accepts a new `parameters` dict on a plot but never touches that plot's gates. Each gate stores its own `parameters: list[str]` and `definition: dict` (coordinate values). After axis change:

- The plot re-renders against new attributes.
- The gates still reference their original parameters, so they're rendered against the **old** axes — invisible on the new plot, but their populations still compute correctly against whatever those old axes were.

This is confusing: users think the gate is "on the plot" but it isn't, and changing axes silently strands the gate.

### Behavior

When `update_plot` is called and `parameters['x_variable']` or `parameters['y_variable']` differs from the plot's current value:

- For every gate where `gate.plot_id == this_plot.id` and `gate.gate_type != 'boolean'`:
  - Rewrite `gate.parameters` positionally:
    - 1D plots (histogram): `parameters[0]` ← new `x_variable`.
    - 2D plots: `parameters[0]` ← new `x_variable`, `parameters[1]` ← new `y_variable`.
  - Leave `gate.definition` untouched (FCS Express semantics — coordinates are interpreted in the new param's data domain).
  - Set `gate.rebound_at = now()`.
- Commit atomically with the plot update.
- After rebind, `_update_active_samples(session_id)` runs so the selected population reflects new gate populations.

Boolean gates and gates on other plots are unaffected.

### Schema change

Add nullable column `rebound_at: DateTime | None` to the `gates` table. The project uses the STORAGE_VERSION wipe-on-bump policy (no incremental migrations):
1. Add column to `Gate` model in `models/models.py`.
2. Bump `STORAGE_VERSION` in `core/config.py` so the next backend start wipes `~/.eidocell` and recreates the schema via `Base.metadata.create_all()`.

Add `rebound_at: datetime | None = None` to the `GateOut` schema in `schemas/workspace/analysis.py`.

### Backend change

`analysis_service.update_plot` becomes:

```python
def update_plot(db, plot_id, name=None, parameters=None) -> dict:
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(404, "Plot not found")

    if name is not None:
        plot.name = name

    if parameters is not None:
        # existing chart_type validation ...
        old_params = plot.parameters or {}
        _rebind_plot_gates(db, plot, old_params, parameters)
        plot.parameters = parameters

    db.commit()
    db.refresh(plot)
    _update_active_samples(db, plot.session_id)
    gate_count = db.query(func.count(Gate.id)).filter(Gate.plot_id == plot.id).scalar()
    return _plot_to_out(plot, gate_count)
```

`_rebind_plot_gates` (new helper):

```python
def _rebind_plot_gates(db, plot, old_params, new_params):
    """Positional rebind of geometric gates on this plot when axes change."""
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
    now = datetime.utcnow()
    for g in gates:
        params = list(g.parameters or [])
        if is_1d:
            if len(params) >= 1 and new_x:
                params[0] = new_x
        else:
            if len(params) >= 1 and new_x:
                params[0] = new_x
            if len(params) >= 2 and new_y:
                params[1] = new_y
        g.parameters = params
        g.rebound_at = now
```

`update_gate` is extended with a "dismiss rebound" path: if the request body sets `clear_rebound: True`, set `gate.rebound_at = None` and commit. Add `clear_rebound: bool = False` to `GateUpdate`.

### Frontend wiring

- `eidocell-ui/src/types/analysis.ts` — add `rebound_at: string | null` to `GateOut` and `clear_rebound?: boolean` to `GateUpdate`.
- `eidocell-ui/src/components/analysis/PopulationTreeNode.vue` — when `gate.rebound_at` is set, render a small chip next to the gate name: `REBOUND` in the project's `text-[9px] font-bold tracking-widest uppercase` style. Clicking the chip calls `analysis.updateGate(gate.id, { clear_rebound: true })` to dismiss.
- `eidocell-ui/src/components/analysis/PlotWidget.vue` — when rendering a gate overlay on a plot, if `gate.rebound_at` is set, the gate's stroke could optionally render dashed; not strictly required, but mentioned here as a follow-up if the chip alone proves insufficient feedback. **For this spec, only the chip in the population tree.**

### Edge cases

- **Same param on both axes** (`x_variable == y_variable`): rebind still positional; gate keeps its `[x, y]` shape with both slots equal. No special handling.
- **Only one axis changes**: only that slot rebinds; the other gate `parameters` slot stays as-is. Already handled by the per-slot check.
- **Histogram → scatter conversion** (or vice versa): out of scope. `chart_type` change isn't supported by `update_plot` today and won't be introduced here.
- **Gate's coordinates outside new param's domain**: gate population becomes 0 (or nearly so). This matches FCS Express; the badge tells the user something changed, and they can redraw.
- **Child gates of a rebound parent on the same plot**: also rebound (they're in the gate list, same `plot_id`). Children of a rebound gate that live on a *different* plot are untouched.
- **Active selection points at a rebound gate**: `_update_active_samples` already recomputes after the commit; new population (possibly empty) takes effect.
- **Multiple consecutive axis changes**: each call stamps `rebound_at = now`. The badge stays until dismissed.

## Testing

### Backend (pytest)

- `test_gallery_session_has_any_mask`: session with no masks → flag false; after Mask rows added → flag true.
- `test_axis_rebind_histogram`: create histogram plot + interval gate; change `x_variable`; assert gate `parameters[0]` updated, `definition` unchanged, `rebound_at` non-null.
- `test_axis_rebind_scatter_xy`: scatter plot + rectangular gate; change both axes; assert both `parameters` slots updated.
- `test_axis_rebind_only_x`: change only x; assert only `parameters[0]` rebound, `parameters[1]` untouched.
- `test_axis_rebind_skips_boolean`: boolean gate present in session; not rebound.
- `test_axis_rebind_does_not_touch_other_plot`: two plots, gates on each; change axes on plot A; gates on plot B untouched.
- `test_clear_rebound`: `update_gate(clear_rebound=True)` clears `rebound_at`.
- `test_axis_rebind_active_samples_recompute`: gate is the selected population; after rebind, `_update_active_samples` produces the new (possibly empty) set.

### Frontend (manual)

- Open a session with no segmentation run → MASK VIEW button disabled with tooltip.
- Run segmentation → after completion, button enables.
- Create a histogram + interval gate; change x via PlotSettingsPopover → gate now lists new x param; REBOUND chip appears in PopulationTree; click chip → chip disappears.
- Same for 2D scatter with rectangular gate; change both axes.

## Files touched

Backend:
- `eidocell-backend/models/models.py` — add `Gate.rebound_at`.
- `eidocell-backend/main.py` — add migration check.
- `eidocell-backend/schemas/workspace/analysis.py` — `GateOut.rebound_at`, `GateUpdate.clear_rebound`.
- `eidocell-backend/schemas/workspace/gallery.py` — `SamplesPage.session_has_any_mask`.
- `eidocell-backend/services/workspace/analysis_service.py` — `update_plot`, new `_rebind_plot_gates`, `update_gate` clear-rebound path.
- `eidocell-backend/services/workspace/gallery_service.py` — populate `session_has_any_mask` in list response.
- `eidocell-backend/tests/` — new tests as above.

Frontend:
- `eidocell-ui/src/types/gallery.ts` — `SamplesPage.session_has_any_mask`.
- `eidocell-ui/src/types/analysis.ts` — `GateOut.rebound_at`, `GateUpdate.clear_rebound`.
- `eidocell-ui/src/stores/gallery.ts` — store `sessionHasAnyMask`.
- `eidocell-ui/src/views/workspace/GalleryView.vue` — pass prop to FilterBar.
- `eidocell-ui/src/components/gallery/FilterBar.vue` — disabled + tooltip + auto-off behavior.
- `eidocell-ui/src/components/analysis/PopulationTreeNode.vue` — REBOUND chip + dismiss handler.
