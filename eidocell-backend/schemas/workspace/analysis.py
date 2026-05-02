import math
from datetime import datetime
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
    chart_type: str  # "histogram" or "scatter"
    parameters: dict  # HistogramParams or ScatterParams as dict


class PlotOut(BaseModel):
    id: str
    name: str
    chart_type: str
    parameters: dict
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
    gate_type: str  # "rectangular", "polygon", "interval", "ellipse", "quadrant"
    definition: dict  # depends on type:
    # rectangular: {"x": float, "y": float, "width": float, "height": float}
    # polygon: {"vertices": [[x, y], ...]}
    # interval: {"min": float, "max": float}
    # ellipse: {"cx": float, "cy": float, "rx": float, "ry": float, "angle": float}
    # quadrant: {"x_threshold": float, "y_threshold": float}
    color: str = "#FF0000"
    parameters: list[str]  # axis names this gate operates on
    is_active: bool = True
    parent_gate_id: str | None = None  # for hierarchical gating

    @model_validator(mode="after")
    def validate_definition(self):
        _validate_gate_definition(self.gate_type, self.definition, self.parameters)
        return self


class GateUpdate(BaseModel):
    name: str | None = None
    color: str | None = None
    definition: dict | None = None
    is_active: bool | None = None


class GateOut(BaseModel):
    id: str
    plot_id: str
    name: str
    gate_type: str
    definition: dict
    color: str
    parameters: list[str]
    is_active: bool
    sample_count: int = 0
    percentage: float = 0.0
    parent_gate_id: str | None = None

    model_config = {"from_attributes": True}
