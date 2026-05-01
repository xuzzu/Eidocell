from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.workspace.segmentation import (
    RunSegmentationRequest,
    RunSegmentationPreviewRequest,
    SegmentationResult,
    SegmentationMethod,
)
from services.workspace import segmentation_service

router = APIRouter(prefix="/sessions/{session_id}", tags=["segmentation"])


@router.get("/segmentation/methods", response_model=list[SegmentationMethod])
def list_methods(session_id: str):
    """List available segmentation methods and their parameters."""
    return segmentation_service.list_available_methods()


@router.post("/segmentation/run", response_model=SegmentationResult)
def run_segmentation(
    session_id: str,
    data: RunSegmentationRequest,
    db: DbSession = Depends(get_db),
):
    """Run segmentation on all active samples (synchronous)."""
    return segmentation_service.run_segmentation(db, session_id, data.method, data.params)


@router.post("/segmentation/preview", response_model=SegmentationResult)
def run_segmentation_preview(
    session_id: str,
    data: RunSegmentationPreviewRequest,
    db: DbSession = Depends(get_db),
):
    """Run segmentation on specific samples (for live preview)."""
    return segmentation_service.run_segmentation_preview(db, session_id, data.method, data.params, data.sample_ids)


@router.post("/segmentation/run-async")
def run_segmentation_async(
    session_id: str,
    data: RunSegmentationRequest,
    db: DbSession = Depends(get_db),
):
    """Start segmentation as a background task. Returns task ID."""
    task_id = segmentation_service.run_segmentation_async(db, session_id, data.method, data.params)
    return {"task_id": task_id}


@router.get("/samples/{sample_id}/mask/attributes")
def get_mask_attributes(
    session_id: str, sample_id: str, db: DbSession = Depends(get_db)
):
    """Get computed mask attributes for a sample."""
    return segmentation_service.get_mask_attributes(db, sample_id)


@router.get("/samples/{sample_id}/mask/overlay")
def get_mask_overlay(
    session_id: str, sample_id: str, db: DbSession = Depends(get_db)
):
    """Serve the mask overlay image."""
    path = segmentation_service.get_mask_overlay_path(db, sample_id)
    return FileResponse(path, media_type="image/png")
