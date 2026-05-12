import math
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, model_validator


# ── Plots ────────────────────────────────────────────────────────────────


class HistogramParams(BaseModel):
    x_variable: str
    num_bins: int = 30
    show_mean: bool = True
    relative_frequency: bool = False


class ScatterParams(BaseModel):
    x_variable: str
    y_variable: str
    color_variable: str | None = None  # "class", "cluster", or None
    trendline: str | None = "none"  # "none", "global", "per_group"


class PlotCreate(BaseModel):
    name: str | None = None  # auto-generated if not provided
    chart_type: str  # "histogram", "scatter", "density", "contour"
    parameters: dict
    parent_gate_id: str | None = None  # restrict plot to this population


class PlotUpdate(BaseModel):
    name: str | None = None
    parameters: dict | None = None


class PlotOut(BaseModel):
    id: str
    name: str
    chart_type: str
    parameters: dict
    parent_gate_id: str | None = None
    created_at: datetime
    gate_count: int = 0

    model_config = {"from_attributes": True}


class PlotDataPoint(BaseModel):
    sample_id: str
    values: dict  # {axis_name: value}
    class_name: str | None = None
    class_color: str | None = None
    cluster_ids: list[str] = []


class PlotData(BaseModel):
    plot_id: str
    chart_type: str
    parameters: dict
    data: list[PlotDataPoint]
    total: int = 0  # total before downsampling


# ── Gates ────────────────────────────────────────────────────────────────


def _check_finite(value: float, field: str) -> None:
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        raise ValueError(f"'{field}' must be a finite number, got {value}")


def _validate_gate_definition(gate_type: str, definition: dict, parameters: list[str]) -> None:
    """Validate gate definition structure matches gate_type."""
    if gate_type == "interval":
        for key in ("min", "max"):
            if key not in definition:
                raise ValueError(f"Interval gate requires '{key}' in definition")
            _check_finite(definition[key], key)
        if definition["min"] > definition["max"]:
            raise ValueError("Interval gate 'min' must be <= 'max'")
        if len(parameters) != 1:
            raise ValueError("Interval gate requires exactly 1 parameter (axis)")

    elif gate_type == "rectangular":
        for key in ("x", "y", "width", "height"):
            if key not in definition:
                raise ValueError(f"Rectangular gate requires '{key}' in definition")
            _check_finite(definition[key], key)
        if definition["width"] <= 0 or definition["height"] <= 0:
            raise ValueError("Rectangular gate width and height must be positive")
        if len(parameters) != 2:
            raise ValueError("Rectangular gate requires exactly 2 parameters (axes)")

    elif gate_type == "polygon":
        if "vertices" not in definition:
            raise ValueError("Polygon gate requires 'vertices' in definition")
        vertices = definition["vertices"]
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError("Polygon gate requires at least 3 vertices")
        for i, v in enumerate(vertices):
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError(f"Vertex {i} must be a [x, y] pair")
            _check_finite(v[0], f"vertex[{i}][0]")
            _check_finite(v[1], f"vertex[{i}][1]")
        if len(parameters) != 2:
            raise ValueError("Polygon gate requires exactly 2 parameters (axes)")

    elif gate_type == "ellipse":
        for key in ("cx", "cy", "rx", "ry"):
            if key not in definition:
                raise ValueError(f"Ellipse gate requires '{key}' in definition")
            _check_finite(definition[key], key)
        if definition["rx"] <= 0 or definition["ry"] <= 0:
            raise ValueError("Ellipse gate rx and ry must be positive")
        if "angle" in definition:
            _check_finite(definition["angle"], "angle")
        if len(parameters) != 2:
            raise ValueError("Ellipse gate requires exactly 2 parameters (axes)")

    elif gate_type == "quadrant":
        for key in ("x_threshold", "y_threshold"):
            if key not in definition:
                raise ValueError(f"Quadrant gate requires '{key}' in definition")
            _check_finite(definition[key], key)
        if len(parameters) != 2:
            raise ValueError("Quadrant gate requires exactly 2 parameters (axes)")

    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


class GateCreate(BaseModel):
    name: str | None = None
    gate_type: Literal["rectangular", "polygon", "interval", "ellipse", "quadrant"]
    definition: dict
    color: str = "#FF0000"
    parameters: list[str]
    parent_gate_id: str | None = None

    @model_validator(mode="after")
    def validate_definition(self):
        _validate_gate_definition(self.gate_type, self.definition, self.parameters)
        return self


class GateUpdate(BaseModel):
    name: str | None = None
    color: str | None = None
    definition: dict | None = None
    # parent_gate_id: explicit None = move to root; absent = leave unchanged.
    # Use model_fields_set to distinguish at the service layer.
    parent_gate_id: str | None = None
    clear_rebound: bool = False


class BooleanGateCreate(BaseModel):
    name: str
    operator: Literal["AND", "OR"]
    source_gate_ids: list[str]
    color: str = "#9333EA"

    @model_validator(mode="after")
    def validate_sources(self):
        if len(self.source_gate_ids) != 2:
            raise ValueError("Boolean gates require exactly 2 source gates")
        if self.source_gate_ids[0] == self.source_gate_ids[1]:
            raise ValueError("Boolean gate source gates must be distinct")
        return self


class SelectPopulationRequest(BaseModel):
    gate_id: str | None = None


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
