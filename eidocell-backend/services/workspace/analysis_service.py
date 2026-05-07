"""Analysis service: plots, gates, and active sample filtering."""

import logging
import math
import random
import struct

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from models.models import Sample, SampleClass, Mask, Plot, Gate, Session, sample_clusters
from schemas.workspace.analysis import _validate_gate_definition

logger = logging.getLogger("eidocell.analysis")


# ── Plot CRUD ───────────────────────────────────────────────────────────


def create_plot(
    db: DbSession, session_id: str,
    chart_type: str, parameters: dict, name: str | None,
    parent_gate_id: str | None = None,
) -> dict:
    valid_chart_types = ("histogram", "scatter", "density", "contour")
    if chart_type not in valid_chart_types:
        raise HTTPException(status_code=400, detail=f"chart_type must be one of {valid_chart_types}")

    if chart_type == "histogram":
        if "x_variable" not in parameters:
            raise HTTPException(status_code=400, detail="Histogram requires x_variable")
        if not name:
            name = f"Histogram: {parameters['x_variable']}"
    else:
        if "x_variable" not in parameters or "y_variable" not in parameters:
            raise HTTPException(status_code=400, detail=f"{chart_type.title()} requires x_variable and y_variable")
        if not name:
            label = chart_type.title()
            name = f"{label}: {parameters['x_variable']} vs {parameters['y_variable']}"

    if parent_gate_id is not None:
        parent = (
            db.query(Gate)
            .filter(Gate.id == parent_gate_id, Gate.session_id == session_id)
            .first()
        )
        if not parent:
            raise HTTPException(status_code=404, detail="Parent gate not found")

    plot = Plot(
        session_id=session_id,
        name=name,
        chart_type=chart_type,
        parameters=parameters,
        parent_gate_id=parent_gate_id,
    )
    db.add(plot)
    db.commit()
    db.refresh(plot)
    return _plot_to_out(plot, 0)


def list_plots(db: DbSession, session_id: str) -> list[dict]:
    plots = db.query(Plot).filter(Plot.session_id == session_id).order_by(Plot.created_at).all()
    if not plots:
        return []
    plot_ids = [p.id for p in plots]
    gate_counts = dict(
        db.query(Gate.plot_id, func.count(Gate.id))
        .filter(Gate.plot_id.in_(plot_ids))
        .group_by(Gate.plot_id)
        .all()
    )
    return [_plot_to_out(p, gate_counts.get(p.id, 0)) for p in plots]


def get_plot(db: DbSession, plot_id: str) -> dict:
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")
    gate_count = db.query(func.count(Gate.id)).filter(Gate.plot_id == plot.id).scalar()
    return _plot_to_out(plot, gate_count)


def delete_plot(db: DbSession, plot_id: str) -> None:
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")
    session_id = plot.session_id
    # Cascade: every gate created on this plot, plus its hierarchical
    # descendants, plus any boolean gate that referenced any of them.
    for gate_id in [g.id for g in plot.gates]:
        _delete_gate_subtree(db, gate_id)
    db.delete(plot)
    db.commit()
    _clear_selection_if_dangling(db, session_id)
    _update_active_samples(db, session_id)


def _plot_to_out(plot: Plot, gate_count: int) -> dict:
    return {
        "id": plot.id,
        "name": plot.name,
        "chart_type": plot.chart_type,
        "parameters": plot.parameters,
        "parent_gate_id": plot.parent_gate_id,
        "created_at": plot.created_at,
        "gate_count": gate_count,
    }


# ── Plot data ───────────────────────────────────────────────────────────


def _query_plot_rows(db: DbSession, plot: Plot, axes: list[str]):
    """Shared query for plot data: returns list of (sample, mask, class) tuples.

    Returns ALL session samples regardless of is_active so the plot keeps a
    stable axis range as gates are added. If the plot inherits from a parent
    gate, results are restricted to that population.
    """
    query = (
        db.query(Sample, Mask, SampleClass)
        .join(Mask, Sample.id == Mask.sample_id)
        .outerjoin(SampleClass, Sample.class_id == SampleClass.id)
        .filter(Sample.session_id == plot.session_id)
    )
    if plot.parent_gate_id:
        parent = db.query(Gate).filter(Gate.id == plot.parent_gate_id).first()
        if parent:
            parent_ids = _compute_gate_population(db, parent)
            if not parent_ids:
                return []
            query = query.filter(Sample.id.in_(parent_ids))
    return query.all()


def get_plot_data(db: DbSession, plot_id: str, *, max_points: int = 0) -> dict:
    """Return data points for rendering a plot.

    If max_points > 0 and total exceeds it, uses reservoir sampling to downsample.
    """
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    params = plot.parameters
    chart_type = plot.chart_type

    # Determine which axes we need
    if chart_type == "histogram":
        axes = [params["x_variable"]]
    else:
        axes = [params["x_variable"], params["y_variable"]]

    color_var = params.get("color_variable")

    rows = _query_plot_rows(db, plot, axes)

    data = []
    for sample, mask, sample_class in rows:
        attrs = mask.attributes or {}
        values = {}
        skip = False
        for axis in axes:
            if axis in attrs:
                values[axis] = attrs[axis]
            else:
                skip = True
                break
        if skip:
            continue

        point = {
            "sample_id": sample.id,
            "values": values,
            "class_name": sample_class.name if sample_class else None,
            "class_color": sample_class.color if sample_class else None,
            "cluster_ids": [],
        }

        if color_var == "cluster":
            cluster_ids = (
                db.query(sample_clusters.c.cluster_id)
                .filter(sample_clusters.c.sample_id == sample.id)
                .all()
            )
            point["cluster_ids"] = [cid for (cid,) in cluster_ids]

        data.append(point)

    total = len(data)

    # Downsample for rendering if requested
    if max_points > 0 and total > max_points:
        data = random.sample(data, max_points)

    return {
        "plot_id": plot.id,
        "chart_type": chart_type,
        "parameters": params,
        "data": data,
        "total": total,
    }


def get_plot_data_binary(db: DbSession, plot_id: str, *, max_points: int = 50000) -> dict:
    """Return plot data as binary Float32Arrays + metadata for fast frontend transfer.

    Returns:
        {
            "meta": {"plot_id", "chart_type", "axes", "total", "returned"},
            "sample_ids": [...],
            "x_data": bytes (float32),
            "y_data": bytes | None (float32),
            "colors": [...],  # hex color strings per point
        }
    """
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    params = plot.parameters
    chart_type = plot.chart_type

    if chart_type == "histogram":
        axes = [params["x_variable"]]
    else:
        axes = [params["x_variable"], params["y_variable"]]

    rows = _query_plot_rows(db, plot, axes)

    sample_ids = []
    x_vals = []
    y_vals = []
    colors = []

    for sample, mask, sample_class in rows:
        attrs = mask.attributes or {}
        skip = False
        axis_values = []
        for axis in axes:
            if axis in attrs:
                axis_values.append(float(attrs[axis]))
            else:
                skip = True
                break
        if skip:
            continue

        sample_ids.append(sample.id)
        x_vals.append(axis_values[0])
        if len(axis_values) > 1:
            y_vals.append(axis_values[1])
        colors.append(sample_class.color if sample_class else "#64748b")

    total = len(sample_ids)

    # Downsample
    if max_points > 0 and total > max_points:
        indices = random.sample(range(total), max_points)
        indices.sort()
        sample_ids = [sample_ids[i] for i in indices]
        x_vals = [x_vals[i] for i in indices]
        if y_vals:
            y_vals = [y_vals[i] for i in indices]
        colors = [colors[i] for i in indices]

    returned = len(sample_ids)

    x_data = struct.pack(f"{returned}f", *x_vals) if x_vals else b""
    y_data = struct.pack(f"{returned}f", *y_vals) if y_vals else None

    return {
        "meta": {
            "plot_id": plot.id,
            "chart_type": chart_type,
            "axes": axes,
            "total": total,
            "returned": returned,
        },
        "sample_ids": sample_ids,
        "x_data": x_data,
        "y_data": y_data,
        "colors": colors,
    }


def batch_plot_data(db: DbSession, plot_ids: list[str], *, max_points: int = 0) -> dict[str, dict]:
    """Fetch data for multiple plots in a single call.

    Returns a dict mapping plot_id → PlotData dict.
    """
    result = {}
    for pid in plot_ids:
        try:
            result[pid] = get_plot_data(db, pid, max_points=max_points)
        except HTTPException:
            continue  # skip missing plots
    return result


def list_available_parameters(db: DbSession, session_id: str) -> list[str]:
    """Return list of mask attribute names available for plotting."""
    # Get one mask with attributes to discover available keys
    mask = (
        db.query(Mask)
        .join(Sample, Mask.sample_id == Sample.id)
        .filter(Sample.session_id == session_id, Mask.attributes.isnot(None))
        .first()
    )
    if not mask or not mask.attributes:
        return []
    return sorted(mask.attributes.keys())


# ── Gate CRUD ───────────────────────────────────────────────────────────


def create_gate(
    db: DbSession, session_id: str, plot_id: str,
    gate_type: str, definition: dict, parameters: list[str],
    name: str | None, color: str,
    parent_gate_id: str | None = None,
) -> dict:
    if gate_type == "boolean":
        raise HTTPException(
            status_code=400,
            detail="Boolean gates must be created via /boolean-gates",
        )

    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    # If the plot itself inherits from a population, gates drawn on it
    # automatically nest under that parent unless an explicit parent was given.
    if parent_gate_id is None and plot.parent_gate_id is not None:
        parent_gate_id = plot.parent_gate_id

    if parent_gate_id:
        parent = db.query(Gate).filter(Gate.id == parent_gate_id).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent gate not found")
        # Boolean parents are allowed: child population becomes
        # boolean.population ∩ child.filter, which is the natural meaning when a
        # plot inherits from a boolean and a gate is then drawn on it.

    if not name:
        name = f"Gate {gate_type[:4].title()}"

    gate = Gate(
        plot_id=plot_id,
        session_id=session_id,
        name=name,
        gate_type=gate_type,
        definition=definition,
        color=color,
        parameters=parameters,
        is_active=False,
        parent_gate_id=parent_gate_id,
    )
    db.add(gate)
    db.commit()
    db.refresh(gate)

    sample_ids = _compute_gate_population(db, gate)
    total = _session_total_samples(db, session_id)
    return _gate_to_out(gate, len(sample_ids), total)


def create_boolean_gate(
    db: DbSession, session_id: str,
    name: str, operator: str, source_gate_ids: list[str], color: str,
) -> dict:
    if operator not in ("AND", "OR"):
        raise HTTPException(status_code=400, detail="Operator must be AND or OR")
    if len(source_gate_ids) != 2 or source_gate_ids[0] == source_gate_ids[1]:
        raise HTTPException(
            status_code=400,
            detail="Boolean gates require two distinct source gates",
        )

    sources = (
        db.query(Gate)
        .filter(Gate.id.in_(source_gate_ids), Gate.session_id == session_id)
        .all()
    )
    if len(sources) != 2:
        raise HTTPException(status_code=404, detail="Source gate(s) not found")

    gate = Gate(
        plot_id=None,
        session_id=session_id,
        name=name,
        gate_type="boolean",
        definition={},
        color=color,
        parameters=[],
        is_active=False,
        parent_gate_id=None,
        operator=operator,
        source_gate_ids=list(source_gate_ids),
    )
    db.add(gate)
    db.commit()
    db.refresh(gate)

    pop = _compute_gate_population(db, gate)
    total = _session_total_samples(db, session_id)
    return _gate_to_out(gate, len(pop), total)


def list_gates(db: DbSession, session_id: str, plot_id: str | None = None) -> list[dict]:
    query = db.query(Gate).filter(Gate.session_id == session_id)
    if plot_id:
        query = query.filter(Gate.plot_id == plot_id)
    gates = query.all()

    total = _session_total_samples(db, session_id)

    results = []
    for g in gates:
        pop = _compute_gate_population(db, g)
        results.append(_gate_to_out(g, len(pop), total))
    return results


def update_gate(db: DbSession, gate_id: str, name: str | None, color: str | None,
                definition: dict | None,
                parent_gate_id: str | None = None,
                update_parent: bool = False) -> dict:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    if name is not None:
        gate.name = name
    if color is not None:
        gate.color = color
    if definition is not None:
        if gate.gate_type == "boolean":
            raise HTTPException(status_code=400, detail="Boolean gates have no geometric definition")
        try:
            _validate_gate_definition(gate.gate_type, definition, gate.parameters)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        gate.definition = definition

    if update_parent:
        if gate.gate_type == "boolean":
            raise HTTPException(status_code=400, detail="Boolean gates cannot be reparented")
        if parent_gate_id is not None:
            if parent_gate_id == gate.id:
                raise HTTPException(status_code=400, detail="Gate cannot be its own parent")
            target = db.query(Gate).filter(Gate.id == parent_gate_id).first()
            if not target or target.session_id != gate.session_id:
                raise HTTPException(status_code=400, detail="Invalid parent gate")
            cursor: Gate | None = target
            seen: set[str] = set()
            while cursor is not None and cursor.id not in seen:
                if cursor.id == gate.id:
                    raise HTTPException(status_code=400, detail="Cycle detected: target is a descendant of this gate")
                seen.add(cursor.id)
                cursor = (
                    db.query(Gate).filter(Gate.id == cursor.parent_gate_id).first()
                    if cursor.parent_gate_id else None
                )
        gate.parent_gate_id = parent_gate_id

    db.commit()
    db.refresh(gate)

    pop = _compute_gate_population(db, gate)
    total = _session_total_samples(db, gate.session_id)

    # Definition or parent change can shift populations of this gate, its
    # descendants, and any boolean depending on it. Refresh the active-sample
    # mirror so the gallery view tracks the selected gate's new population
    # without forcing a manual reselection.
    _update_active_samples(db, gate.session_id)

    return _gate_to_out(gate, len(pop), total)


def _delete_gate_subtree(db: DbSession, gate_id: str) -> set[str]:
    """Delete a gate, its hierarchical descendants, and any boolean gate that
    transitively references any of them. Returns the set of deleted gate IDs."""
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        return set()
    session_id = gate.session_id

    # BFS hierarchical descendants.
    to_delete: set[str] = {gate.id}
    frontier: list[str] = [gate.id]
    while frontier:
        children = (
            db.query(Gate.id)
            .filter(Gate.parent_gate_id.in_(frontier))
            .all()
        )
        next_frontier = [cid for (cid,) in children if cid not in to_delete]
        to_delete.update(next_frontier)
        frontier = next_frontier

    # Iterate to fixed point: any boolean gate referencing a doomed gate also dies.
    while True:
        booleans = (
            db.query(Gate)
            .filter(
                Gate.session_id == session_id,
                Gate.gate_type == "boolean",
                Gate.id.notin_(to_delete),
            )
            .all()
        )
        added = False
        for b in booleans:
            sources = b.source_gate_ids or []
            if any(s in to_delete for s in sources):
                to_delete.add(b.id)
                added = True
        if not added:
            break

    db.query(Gate).filter(Gate.id.in_(to_delete)).delete(synchronize_session="fetch")
    db.commit()
    return to_delete


def delete_gate(db: DbSession, gate_id: str) -> None:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")
    session_id = gate.session_id
    _delete_gate_subtree(db, gate_id)
    _clear_selection_if_dangling(db, session_id)
    _update_active_samples(db, session_id)


def get_gate_population(db: DbSession, gate_id: str) -> list[str]:
    """Return sample IDs that fall within a gate."""
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")
    return _compute_gate_population(db, gate)


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
    }


def _session_total_samples(db: DbSession, session_id: str) -> int:
    return db.query(func.count(Sample.id)).filter(Sample.session_id == session_id).scalar() or 0


# ── Gate population computation ─────────────────────────────────────────


def _compute_gate_population(
    db: DbSession, gate: Gate, _seen: set[str] | None = None
) -> list[str]:
    """Compute which sample IDs fall within a gate.

    Hierarchical gate: parent population (recursive) intersected with this gate's
    geometric filter. Boolean gate: AND/OR of two source gates' populations.
    `_seen` tracks gates currently being evaluated up-stack to break cycles.
    """
    seen = (_seen or set()) | {gate.id}

    if gate.gate_type == "boolean":
        sources = gate.source_gate_ids or []
        if len(sources) != 2 or any(s in seen for s in sources):
            return []
        a = db.query(Gate).filter(Gate.id == sources[0]).first()
        b = db.query(Gate).filter(Gate.id == sources[1]).first()
        if not a or not b:
            return []
        pop_a = set(_compute_gate_population(db, a, seen))
        pop_b = set(_compute_gate_population(db, b, seen))
        if gate.operator == "AND":
            return list(pop_a & pop_b)
        return list(pop_a | pop_b)

    axes = gate.parameters
    gate_type = gate.gate_type
    defn = gate.definition

    parent_ids: set[str] | None = None
    if gate.parent_gate_id and gate.parent_gate_id not in seen:
        parent_gate = db.query(Gate).filter(Gate.id == gate.parent_gate_id).first()
        if parent_gate:
            parent_ids = set(_compute_gate_population(db, parent_gate, seen))
            if not parent_ids:
                return []

    # Population evaluation must NOT depend on Sample.is_active — that field
    # mirrors the currently-selected population, so filtering by it would make
    # every other gate's count shrink to "samples also in the selection."
    query = (
        db.query(Sample.id, Mask.attributes)
        .join(Mask, Sample.id == Mask.sample_id)
        .filter(Sample.session_id == gate.session_id)
    )
    if parent_ids is not None:
        query = query.filter(Sample.id.in_(parent_ids))

    matching_ids: list[str] = []
    for sample_id, attrs in query.all():
        if not attrs:
            continue
        vals = []
        skip = False
        for axis in axes:
            if axis in attrs:
                vals.append(attrs[axis])
            else:
                skip = True
                break
        if skip:
            continue

        if _point_in_gate(gate_type, defn, vals):
            matching_ids.append(sample_id)

    return matching_ids


def _point_in_gate(gate_type: str, definition: dict, values: list[float]) -> bool:
    if gate_type == "interval":
        return definition["min"] <= values[0] <= definition["max"]

    elif gate_type == "rectangular":
        x, y = values[0], values[1]
        rx, ry = definition["x"], definition["y"]
        rw, rh = definition["width"], definition["height"]
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    elif gate_type == "polygon":
        x, y = values[0], values[1]
        vertices = definition["vertices"]
        return _point_in_polygon(x, y, vertices)

    elif gate_type == "ellipse":
        x, y = values[0], values[1]
        cx, cy = definition["cx"], definition["cy"]
        rx, ry = definition["rx"], definition["ry"]
        angle = math.radians(definition.get("angle", 0))
        dx, dy = x - cx, y - cy
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        nx = cos_a * dx + sin_a * dy
        ny = -sin_a * dx + cos_a * dy
        return (nx / rx) ** 2 + (ny / ry) ** 2 <= 1

    elif gate_type == "quadrant":
        x, y = values[0], values[1]
        # Default: Q1 (upper-right) — x > threshold AND y > threshold
        return x >= definition["x_threshold"] and y >= definition["y_threshold"]

    return False


def _point_in_polygon(x: float, y: float, vertices: list[list[float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ── Active samples filtering ───────────────────────────────────────────


def _clear_selection_if_dangling(db: DbSession, session_id: str) -> None:
    """Null out Session.selected_gate_id if the referenced gate no longer exists."""
    sess = db.query(Session).filter(Session.id == session_id).first()
    if not sess or sess.selected_gate_id is None:
        return
    exists = db.query(Gate.id).filter(Gate.id == sess.selected_gate_id).first()
    if not exists:
        sess.selected_gate_id = None
        db.commit()


def _update_active_samples(db: DbSession, session_id: str) -> None:
    """Set Sample.is_active to mirror the selected gate's population, or all
    samples active when no gate is selected."""
    sess = db.query(Session).filter(Session.id == session_id).first()
    if not sess:
        return

    if sess.selected_gate_id is None:
        db.query(Sample).filter(Sample.session_id == session_id).update({"is_active": True})
        db.commit()
        return

    gate = db.query(Gate).filter(Gate.id == sess.selected_gate_id).first()
    if not gate:
        sess.selected_gate_id = None
        db.query(Sample).filter(Sample.session_id == session_id).update({"is_active": True})
        db.commit()
        return

    active_ids = set(_compute_gate_population(db, gate))
    if active_ids:
        db.query(Sample).filter(
            Sample.session_id == session_id, Sample.id.in_(active_ids)
        ).update({"is_active": True}, synchronize_session="fetch")
        db.query(Sample).filter(
            Sample.session_id == session_id, ~Sample.id.in_(active_ids)
        ).update({"is_active": False}, synchronize_session="fetch")
    else:
        db.query(Sample).filter(Sample.session_id == session_id).update({"is_active": False})
    db.commit()


def select_population(db: DbSession, session_id: str, gate_id: str | None) -> dict:
    """Set the session's selected gate (or clear with None) and refresh
    Sample.is_active to mirror its population."""
    sess = db.query(Session).filter(Session.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    if gate_id is not None:
        gate = (
            db.query(Gate)
            .filter(Gate.id == gate_id, Gate.session_id == session_id)
            .first()
        )
        if not gate:
            raise HTTPException(status_code=404, detail="Gate not found")

    sess.selected_gate_id = gate_id
    db.commit()
    _update_active_samples(db, session_id)
    return {"selected_gate_id": gate_id}


def get_selected_population(db: DbSession, session_id: str) -> dict:
    sess = db.query(Session).filter(Session.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"selected_gate_id": sess.selected_gate_id}


def get_active_sample_ids(db: DbSession, session_id: str) -> list[str]:
    """Return IDs of currently active samples."""
    rows = (
        db.query(Sample.id)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .all()
    )
    return [r[0] for r in rows]


def reset_active_samples(db: DbSession, session_id: str) -> int:
    """Clear the selected population so all samples become active again."""
    select_population(db, session_id, None)
    return (
        db.query(func.count(Sample.id))
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .scalar()
        or 0
    )


# ── Population tree ──────────���────────────────────────────────────────


def get_population_tree(db: DbSession, session_id: str) -> dict:
    """Return the full population tree for a session.

    Shape: { root: <synthetic 'All Events' node with children>, booleans: [...] }
    Hierarchical gates nest under root; boolean gates are returned in a flat list.
    """
    all_gates = db.query(Gate).filter(Gate.session_id == session_id).all()
    total = _session_total_samples(db, session_id)

    gate_pops: dict[str, int] = {}
    for g in all_gates:
        gate_pops[g.id] = len(_compute_gate_population(db, g))

    gate_map: dict[str, dict] = {}
    for g in all_gates:
        node = _gate_to_out(g, gate_pops[g.id], total)
        node["children"] = []
        gate_map[g.id] = node

    booleans: list[dict] = []
    hierarchical_roots: list[dict] = []
    for g in all_gates:
        node = gate_map[g.id]
        if g.gate_type == "boolean":
            booleans.append(node)
            continue
        if g.parent_gate_id and g.parent_gate_id in gate_map:
            parent_node = gate_map[g.parent_gate_id]
            parent_count = gate_pops.get(g.parent_gate_id, 0)
            if parent_count > 0:
                node["percentage"] = (gate_pops[g.id] / parent_count) * 100
            parent_node["children"].append(node)
        else:
            hierarchical_roots.append(node)

    root = {
        "id": "__root__",
        "plot_id": None,
        "name": "All Events",
        "gate_type": "root",
        "definition": {},
        "color": "#475569",
        "parameters": [],
        "sample_count": total,
        "percentage": 100.0 if total > 0 else 0.0,
        "parent_gate_id": None,
        "operator": None,
        "source_gate_ids": None,
        "children": hierarchical_roots,
    }

    return {"root": root, "booleans": booleans}
