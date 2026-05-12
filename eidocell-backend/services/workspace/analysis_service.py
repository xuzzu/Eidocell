"""Analysis service: plots, gates, and active sample filtering."""

import logging
import math
import random
import struct
from datetime import datetime

import numpy as np
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from core.storage import mask_attrs as lance_mask_attrs
from models.models import Gate, Plot, Sample, SampleClass, Session, sample_clusters
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

    FCS Express semantics: gate.definition coordinates are preserved verbatim;
    only the parameter names referenced by each slot are rewritten to the plot's
    new axes. Boolean gates and gates on other plots are untouched.
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


def delete_plot(db: DbSession, plot_id: str) -> None:
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")
    session_id = plot.session_id
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


def _fetch_plot_rows(
    db: DbSession, plot: Plot, axes: list[str]
) -> tuple[list[Sample], dict[str, SampleClass | None], dict[str, dict]]:
    """Fetch all session samples (or restricted by parent gate) joined with class info,
    plus per-sample attrs for the requested axes."""
    query = (
        db.query(Sample, SampleClass)
        .outerjoin(SampleClass, Sample.class_id == SampleClass.id)
        .filter(Sample.session_id == plot.session_id)
    )
    if plot.parent_gate_id:
        parent = db.query(Gate).filter(Gate.id == plot.parent_gate_id).first()
        if parent:
            parent_ids = _compute_gate_population(db, parent)
            if not parent_ids:
                return [], {}, {}
            query = query.filter(Sample.id.in_(parent_ids))
    rows = query.all()
    samples = [s for s, _ in rows]
    class_by_sid = {s.id: cls for s, cls in rows}
    attrs_by_sid = lance_mask_attrs.get_attrs_bulk(plot.session_id, [s.id for s in samples])
    return samples, class_by_sid, attrs_by_sid


def get_plot_data(db: DbSession, plot_id: str, *, max_points: int = 0) -> dict:
    """Return data points for rendering a plot.

    If max_points > 0 and total exceeds it, uses reservoir sampling to downsample.
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

    color_var = params.get("color_variable")
    samples, class_by_sid, attrs_by_sid = _fetch_plot_rows(db, plot, axes)

    data = []
    for sample in samples:
        attrs = attrs_by_sid.get(sample.id) or {}
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
        cls = class_by_sid.get(sample.id)
        point = {
            "sample_id": sample.id,
            "values": values,
            "class_name": cls.name if cls else None,
            "class_color": cls.color if cls else None,
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
    """Return plot data as binary Float32Arrays + metadata."""
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    params = plot.parameters
    chart_type = plot.chart_type

    if chart_type == "histogram":
        axes = [params["x_variable"]]
    else:
        axes = [params["x_variable"], params["y_variable"]]

    samples, class_by_sid, attrs_by_sid = _fetch_plot_rows(db, plot, axes)

    sample_ids: list[str] = []
    x_vals: list[float] = []
    y_vals: list[float] = []
    colors: list[str] = []

    for sample in samples:
        attrs = attrs_by_sid.get(sample.id) or {}
        skip = False
        axis_values: list[float] = []
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
        cls = class_by_sid.get(sample.id)
        colors.append(cls.color if cls else "#64748b")

    total = len(sample_ids)

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
    result = {}
    for pid in plot_ids:
        try:
            result[pid] = get_plot_data(db, pid, max_points=max_points)
        except HTTPException:
            continue
    return result


def list_available_parameters(db: DbSession, session_id: str) -> list[str]:
    """Return list of mask attribute names available for plotting."""
    return sorted(lance_mask_attrs.list_attribute_names(session_id))


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

    if parent_gate_id is None and plot.parent_gate_id is not None:
        parent_gate_id = plot.parent_gate_id

    if parent_gate_id:
        parent = db.query(Gate).filter(Gate.id == parent_gate_id).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent gate not found")

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
                update_parent: bool = False,
                clear_rebound: bool = False) -> dict:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    if name is not None:
        gate.name = name
    if color is not None:
        gate.color = color
    if clear_rebound:
        gate.rebound_at = None
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
    _update_active_samples(db, gate.session_id)
    return _gate_to_out(gate, len(pop), total)


def _delete_gate_subtree(db: DbSession, gate_id: str) -> set[str]:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        return set()
    session_id = gate.session_id

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
        "rebound_at": gate.rebound_at,
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

    parent_ids: set[str] | None = None
    if gate.parent_gate_id and gate.parent_gate_id not in seen:
        parent_gate = db.query(Gate).filter(Gate.id == gate.parent_gate_id).first()
        if parent_gate:
            parent_ids = set(_compute_gate_population(db, parent_gate, seen))
            if not parent_ids:
                return []

    # Limit candidate set to samples in this session (and parent gate, if any)
    sample_q = db.query(Sample.id).filter(Sample.session_id == gate.session_id)
    if parent_ids is not None:
        sample_q = sample_q.filter(Sample.id.in_(parent_ids))
    candidate_ids = [sid for (sid,) in sample_q.all()]
    if not candidate_ids:
        return []

    return _evaluate_geometric_gate(gate, candidate_ids)


def _evaluate_geometric_gate(gate: Gate, candidate_ids: list[str]) -> list[str]:
    axes = list(gate.parameters or [])
    gate_type = gate.gate_type
    defn = gate.definition or {}
    valid_columns = set(lance_mask_attrs.ATTRIBUTE_NAMES)
    for axis in axes:
        if axis not in valid_columns:
            return []

    if gate_type == "interval":
        axis = axes[0]
        lo = float(defn["min"])
        hi = float(defn["max"])
        where = f"{axis} BETWEEN {lo} AND {hi}"
        return lance_mask_attrs.query_predicate(
            gate.session_id, where, sample_ids=candidate_ids
        )

    if gate_type == "rectangular":
        ax_x, ax_y = axes[0], axes[1]
        x = float(defn["x"]); y = float(defn["y"])
        w = float(defn["width"]); h = float(defn["height"])
        where = (
            f"{ax_x} BETWEEN {x} AND {x + w} "
            f"AND {ax_y} BETWEEN {y} AND {y + h}"
        )
        return lance_mask_attrs.query_predicate(
            gate.session_id, where, sample_ids=candidate_ids
        )

    if gate_type == "quadrant":
        ax_x, ax_y = axes[0], axes[1]
        xt = float(defn["x_threshold"]); yt = float(defn["y_threshold"])
        where = f"{ax_x} >= {xt} AND {ax_y} >= {yt}"
        return lance_mask_attrs.query_predicate(
            gate.session_id, where, sample_ids=candidate_ids
        )

    if gate_type == "ellipse":
        ax_x, ax_y = axes[0], axes[1]
        sids, cols = lance_mask_attrs.fetch_columns(
            gate.session_id, [ax_x, ax_y], sample_ids=candidate_ids
        )
        if not sids:
            return []
        cx = float(defn["cx"]); cy = float(defn["cy"])
        rx = float(defn["rx"]); ry = float(defn["ry"])
        angle = math.radians(defn.get("angle", 0))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = cols[ax_x] - cx
        dy = cols[ax_y] - cy
        nx = cos_a * dx + sin_a * dy
        ny = -sin_a * dx + cos_a * dy
        mask = (nx / rx) ** 2 + (ny / ry) ** 2 <= 1
        return [sid for sid, m in zip(sids, mask) if m]

    if gate_type == "polygon":
        ax_x, ax_y = axes[0], axes[1]
        sids, cols = lance_mask_attrs.fetch_columns(
            gate.session_id, [ax_x, ax_y], sample_ids=candidate_ids
        )
        if not sids:
            return []
        vertices = defn.get("vertices") or []
        if len(vertices) < 3:
            return []
        return _polygon_contains_vec(sids, cols[ax_x], cols[ax_y], vertices)

    return []


def _polygon_contains_vec(
    sample_ids: list[str], xs: np.ndarray, ys: np.ndarray, vertices: list[list[float]]
) -> list[str]:
    """Vectorized ray-casting point-in-polygon."""
    n = len(vertices)
    inside = np.zeros(xs.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        cond1 = (yi > ys) != (yj > ys)
        denom = (yj - yi)
        denom_safe = np.where(denom == 0, 1.0, denom)
        x_intersect = (xj - xi) * (ys - yi) / denom_safe + xi
        cond2 = xs < x_intersect
        flip = cond1 & cond2 & (denom != 0)
        inside ^= flip
        j = i
    return [sid for sid, m in zip(sample_ids, inside.tolist()) if m]


# ── Active samples filtering ───────────────────────────────────────────


def _clear_selection_if_dangling(db: DbSession, session_id: str) -> None:
    sess = db.query(Session).filter(Session.id == session_id).first()
    if not sess or sess.selected_gate_id is None:
        return
    exists = db.query(Gate.id).filter(Gate.id == sess.selected_gate_id).first()
    if not exists:
        sess.selected_gate_id = None
        db.commit()


def _update_active_samples(db: DbSession, session_id: str) -> None:
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
    rows = (
        db.query(Sample.id)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .all()
    )
    return [r[0] for r in rows]


def reset_active_samples(db: DbSession, session_id: str) -> int:
    select_population(db, session_id, None)
    return (
        db.query(func.count(Sample.id))
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .scalar()
        or 0
    )


# ── Population tree ─────────────────────────────────────────────────────


def get_population_tree(db: DbSession, session_id: str) -> dict:
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
