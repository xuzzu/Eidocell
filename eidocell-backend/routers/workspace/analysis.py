import struct

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.workspace.analysis import (
    PlotCreate, PlotOut, PlotData,
    GateCreate, GateUpdate, GateOut,
)
from services.workspace import analysis_service

router = APIRouter(prefix="/sessions/{session_id}/analysis", tags=["analysis"])


# ── Available parameters ────────────────────────────────────────────────


@router.get("/parameters", response_model=list[str])
def list_available_parameters(session_id: str, db: DbSession = Depends(get_db)):
    """List mask attribute names available for plot axes."""
    return analysis_service.list_available_parameters(db, session_id)


# ── Plots ───────────────────────────────────────────────────────────────


@router.post("/plots", response_model=PlotOut, status_code=201)
def create_plot(session_id: str, data: PlotCreate, db: DbSession = Depends(get_db)):
    return analysis_service.create_plot(db, session_id, data.chart_type, data.parameters, data.name)


@router.get("/plots", response_model=list[PlotOut])
def list_plots(session_id: str, db: DbSession = Depends(get_db)):
    return analysis_service.list_plots(db, session_id)


@router.get("/plots/{plot_id}", response_model=PlotOut)
def get_plot(session_id: str, plot_id: str, db: DbSession = Depends(get_db)):
    return analysis_service.get_plot(db, plot_id)


@router.get("/plots/{plot_id}/data", response_model=PlotData)
def get_plot_data(
    session_id: str, plot_id: str,
    max_points: int = Query(0, ge=0, description="Downsample to N points (0 = no limit)"),
    db: DbSession = Depends(get_db),
):
    """Return data points for rendering this plot."""
    return analysis_service.get_plot_data(db, plot_id, max_points=max_points)


@router.get("/plots/{plot_id}/data/binary")
def get_plot_data_binary(
    session_id: str, plot_id: str,
    max_points: int = Query(50000, ge=0),
    db: DbSession = Depends(get_db),
):
    """Return plot data as binary for high-performance rendering.

    Response format: JSON metadata header + binary float32 arrays packed as:
    - 4 bytes: header_length (uint32, little-endian)
    - header_length bytes: JSON metadata (utf-8)
    - remaining bytes: x_data (float32[]) + y_data (float32[], if 2D)

    Metadata JSON: {plot_id, chart_type, axes, total, returned, sample_ids, colors}
    """
    import json

    result = analysis_service.get_plot_data_binary(db, plot_id, max_points=max_points)
    meta = result["meta"]
    meta["sample_ids"] = result["sample_ids"]
    meta["colors"] = result["colors"]

    header_bytes = json.dumps(meta).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))

    body = header_len + header_bytes + result["x_data"]
    if result["y_data"]:
        body += result["y_data"]

    return Response(content=body, media_type="application/octet-stream")


class BatchPlotDataRequest(BaseModel):
    plot_ids: list[str]
    max_points: int = 0


@router.post("/batch-plot-data")
def batch_plot_data(session_id: str, data: BatchPlotDataRequest, db: DbSession = Depends(get_db)):
    """Fetch data for multiple plots in a single round-trip."""
    return {"plots": analysis_service.batch_plot_data(db, data.plot_ids, max_points=data.max_points)}


@router.delete("/plots/{plot_id}", status_code=204)
def delete_plot(session_id: str, plot_id: str, db: DbSession = Depends(get_db)):
    analysis_service.delete_plot(db, plot_id)


# ── Gates ───────────────────────────────────────────────────────────────


@router.post("/plots/{plot_id}/gates", response_model=GateOut, status_code=201)
def create_gate(session_id: str, plot_id: str, data: GateCreate, db: DbSession = Depends(get_db)):
    return analysis_service.create_gate(
        db, session_id, plot_id,
        data.gate_type, data.definition, data.parameters,
        data.name, data.color, data.is_active,
        parent_gate_id=data.parent_gate_id,
    )


@router.get("/gates", response_model=list[GateOut])
def list_all_gates(session_id: str, db: DbSession = Depends(get_db)):
    """List all gates across all plots in this session."""
    return analysis_service.list_gates(db, session_id)


@router.get("/plots/{plot_id}/gates", response_model=list[GateOut])
def list_plot_gates(session_id: str, plot_id: str, db: DbSession = Depends(get_db)):
    return analysis_service.list_gates(db, session_id, plot_id=plot_id)


@router.patch("/gates/{gate_id}", response_model=GateOut)
def update_gate(session_id: str, gate_id: str, data: GateUpdate, db: DbSession = Depends(get_db)):
    return analysis_service.update_gate(db, gate_id, data.name, data.color, data.definition, data.is_active)


@router.delete("/gates/{gate_id}", status_code=204)
def delete_gate(session_id: str, gate_id: str, db: DbSession = Depends(get_db)):
    analysis_service.delete_gate(db, gate_id)


@router.get("/gates/{gate_id}/population", response_model=list[str])
def get_gate_population(session_id: str, gate_id: str, db: DbSession = Depends(get_db)):
    """Return sample IDs that fall within this gate."""
    return analysis_service.get_gate_population(db, gate_id)


# ── Active samples ──────────────────────────────────────────────────────


@router.get("/active-samples", response_model=list[str])
def get_active_samples(session_id: str, db: DbSession = Depends(get_db)):
    return analysis_service.get_active_sample_ids(db, session_id)


@router.post("/reset-gates")
def reset_gates(session_id: str, db: DbSession = Depends(get_db)):
    """Deactivate all gates and reactivate all samples."""
    count = analysis_service.reset_active_samples(db, session_id)
    return {"reactivated": count}


# ── Population tree ─────────────────────────────────────────────────────


@router.get("/population-tree")
def get_population_tree(session_id: str, db: DbSession = Depends(get_db)):
    """Return hierarchical gate tree for this session."""
    return analysis_service.get_population_tree(db, session_id)
