"""Analysis service: plots, gates, and active sample filtering."""

import logging
import math
import random
import struct

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from models.models import Sample, SampleClass, Mask, Plot, Gate, sample_clusters
from schemas.workspace.analysis import _validate_gate_definition

logger = logging.getLogger("eidocell.analysis")


# ── Plot CRUD ───────────────────────────────────────────────────────────


def create_plot(db: DbSession, session_id: str, chart_type: str, parameters: dict, name: str | None) -> dict:
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

    plot = Plot(
        session_id=session_id,
        name=name,
        chart_type=chart_type,
        parameters=parameters,
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
    db.delete(plot)
    db.commit()


def _plot_to_out(plot: Plot, gate_count: int) -> dict:
    return {
        "id": plot.id,
        "name": plot.name,
        "chart_type": plot.chart_type,
        "parameters": plot.parameters,
        "created_at": plot.created_at,
        "gate_count": gate_count,
    }


# ── Plot data ───────────────────────────────────────────────────────────


def _query_plot_rows(db: DbSession, plot: Plot, axes: list[str]):
    """Shared query for plot data: returns list of (sample, mask, class) tuples."""
    rows = (
        db.query(Sample, Mask, SampleClass)
        .join(Mask, Sample.id == Mask.sample_id)
        .outerjoin(SampleClass, Sample.class_id == SampleClass.id)
        .filter(Sample.session_id == plot.session_id, Sample.is_active == True)
        .all()
    )
    return rows


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
    name: str | None, color: str, is_active: bool,
    parent_gate_id: str | None = None,
) -> dict:
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

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
        is_active=is_active,
        parent_gate_id=parent_gate_id,
    )
    db.add(gate)
    db.commit()
    db.refresh(gate)

    # Compute population
    sample_ids = _compute_gate_population(db, gate)
    total = db.query(func.count(Sample.id)).filter(
        Sample.session_id == session_id, Sample.is_active == True
    ).scalar()

    # Update active samples if gate is active
    if is_active:
        _update_active_samples(db, session_id)

    return _gate_to_out(gate, len(sample_ids), total)


def list_gates(db: DbSession, session_id: str, plot_id: str | None = None) -> list[dict]:
    query = db.query(Gate).filter(Gate.session_id == session_id)
    if plot_id:
        query = query.filter(Gate.plot_id == plot_id)
    gates = query.all()

    total = db.query(func.count(Sample.id)).filter(
        Sample.session_id == session_id, Sample.is_active == True
    ).scalar()

    results = []
    for g in gates:
        pop = _compute_gate_population(db, g)
        results.append(_gate_to_out(g, len(pop), total))
    return results


def update_gate(db: DbSession, gate_id: str, name: str | None, color: str | None,
                definition: dict | None, is_active: bool | None) -> dict:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    if name is not None:
        gate.name = name
    if color is not None:
        gate.color = color
    if definition is not None:
        try:
            _validate_gate_definition(gate.gate_type, definition, gate.parameters)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        gate.definition = definition
    if is_active is not None:
        gate.is_active = is_active

    db.commit()
    db.refresh(gate)

    pop = _compute_gate_population(db, gate)
    total = db.query(func.count(Sample.id)).filter(
        Sample.session_id == gate.session_id, Sample.is_active == True
    ).scalar()

    if is_active is not None:
        _update_active_samples(db, gate.session_id)

    return _gate_to_out(gate, len(pop), total)


def delete_gate(db: DbSession, gate_id: str) -> None:
    gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")
    session_id = gate.session_id
    db.delete(gate)
    db.commit()
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
        "is_active": gate.is_active,
        "sample_count": sample_count,
        "percentage": (sample_count / total * 100) if total > 0 else 0,
        "parent_gate_id": gate.parent_gate_id,
    }


# ── Gate population computation ─────────────────────────────────────────


def _compute_gate_population(db: DbSession, gate: Gate) -> list[str]:
    """Compute which sample IDs fall within a gate based on mask attributes.

    If the gate has a parent, only samples within the parent population are considered.
    """
    axes = gate.parameters
    gate_type = gate.gate_type
    defn = gate.definition

    # If hierarchical: restrict to parent population
    parent_ids = None
    if gate.parent_gate_id:
        parent_gate = db.query(Gate).filter(Gate.id == gate.parent_gate_id).first()
        if parent_gate:
            parent_ids = set(_compute_gate_population(db, parent_gate))

    # Get all samples with masks in this session
    query = (
        db.query(Sample.id, Mask.attributes)
        .join(Mask, Sample.id == Mask.sample_id)
        .filter(Sample.session_id == gate.session_id, Sample.is_active == True)
    )
    # If parent gate, only check parent's samples
    if parent_ids is not None:
        if len(parent_ids) == 0:
            return []
        query = query.filter(Sample.id.in_(parent_ids))

    rows = query.all()

    matching_ids = []
    for sample_id, attrs in rows:
        if not attrs:
            continue
        # Extract axis values
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


def _update_active_samples(db: DbSession, session_id: str) -> None:
    """Update is_active on samples based on active gates (OR union logic).

    If no active gates exist, all samples are active.
    If active gates exist, only samples in ANY gate are active.
    """
    active_gates = (
        db.query(Gate)
        .filter(Gate.session_id == session_id, Gate.is_active == True)
        .all()
    )

    if not active_gates:
        # No gates — all samples active
        db.query(Sample).filter(Sample.session_id == session_id).update({"is_active": True})
        db.commit()
        return

    # Compute union of all gate populations
    active_ids = set()
    for gate in active_gates:
        pop = _compute_gate_population(db, gate)
        active_ids.update(pop)

    # Update is_active: True for those in gates, False for others
    db.query(Sample).filter(
        Sample.session_id == session_id, Sample.id.in_(active_ids)
    ).update({"is_active": True}, synchronize_session="fetch")

    db.query(Sample).filter(
        Sample.session_id == session_id, ~Sample.id.in_(active_ids)
    ).update({"is_active": False}, synchronize_session="fetch")

    db.commit()


def get_active_sample_ids(db: DbSession, session_id: str) -> list[str]:
    """Return IDs of currently active samples."""
    rows = (
        db.query(Sample.id)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .all()
    )
    return [r[0] for r in rows]


def reset_active_samples(db: DbSession, session_id: str) -> int:
    """Deactivate all gates and reactivate all samples."""
    db.query(Gate).filter(Gate.session_id == session_id).update({"is_active": False})
    updated = db.query(Sample).filter(Sample.session_id == session_id).update({"is_active": True})
    db.commit()
    return updated


# ── Population tree ──────────���────────────────────────────────────────


def get_population_tree(db: DbSession, session_id: str) -> list[dict]:
    """Return all gates for a session arranged as a tree.

    Each node: {id, name, gate_type, color, is_active, plot_id, sample_count,
                percentage, parent_gate_id, children: [...]}
    """
    all_gates = db.query(Gate).filter(Gate.session_id == session_id).all()
    if not all_gates:
        return []

    total = db.query(func.count(Sample.id)).filter(
        Sample.session_id == session_id, Sample.is_active == True
    ).scalar()

    # Compute population for each gate
    gate_pops = {}
    for g in all_gates:
        pop = _compute_gate_population(db, g)
        gate_pops[g.id] = len(pop)

    # Build tree
    gate_map = {}
    for g in all_gates:
        node = _gate_to_out(g, gate_pops[g.id], total)
        node["children"] = []
        gate_map[g.id] = node

    roots = []
    for g in all_gates:
        node = gate_map[g.id]
        if g.parent_gate_id and g.parent_gate_id in gate_map:
            parent_node = gate_map[g.parent_gate_id]
            # Percentage relative to parent
            parent_count = gate_pops.get(g.parent_gate_id, 0)
            if parent_count > 0:
                node["percentage"] = (gate_pops[g.id] / parent_count) * 100
            parent_node["children"].append(node)
        else:
            roots.append(node)

    return roots
