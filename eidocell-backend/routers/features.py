from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.features import (
    RunFeatureExtractionRequest,
    FeatureExtractionMethod,
    RunDimReductionRequest,
)
from services import feature_extraction_service, dimensionality_reduction_service

router = APIRouter(prefix="/sessions/{session_id}/features", tags=["features"])


# ── Feature extraction ────────────────────────────────────────────────


@router.get("/methods", response_model=list[FeatureExtractionMethod])
def list_extraction_methods(session_id: str):
    return feature_extraction_service.list_available_methods()


@router.post("/extract")
def run_feature_extraction(
    session_id: str,
    data: RunFeatureExtractionRequest,
    db: DbSession = Depends(get_db),
):
    """Submit feature extraction as a background task. Returns task_id."""
    task_id = feature_extraction_service.run_feature_extraction(
        db, session_id, data.method
    )
    return {"task_id": task_id}


# ── Dimensionality reduction ─────────────────────────────────────────


@router.get("/dim-reduction/methods")
def list_dim_reduction_methods(session_id: str):
    return dimensionality_reduction_service.list_available_methods()


@router.post("/dim-reduction/run")
def run_dim_reduction(
    session_id: str,
    data: RunDimReductionRequest,
    db: DbSession = Depends(get_db),
):
    """Submit dimensionality reduction as a background task. Returns task_id."""
    task_id = dimensionality_reduction_service.run_dimensionality_reduction(
        db, session_id, data.method, data.n_components, data.params
    )
    return {"task_id": task_id}
