from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.learning import (
    TrainModelRequest,
    TrainModelResult,
    RunInferenceRequest,
    InferenceResult,
    ApplyModelRequest,
    TrainedModelOut,
)
from services import learning_service

router = APIRouter(tags=["learning"])


# ── Session-scoped: train & infer ─────────────────────────────────────


@router.post(
    "/sessions/{session_id}/learning/train",
    response_model=TrainModelResult,
)
def train_model(
    session_id: str,
    data: TrainModelRequest,
    db: DbSession = Depends(get_db),
):
    return learning_service.train_model(
        db, session_id, data.name, data.classifier_type, data.feature_source, data.params
    )


@router.post(
    "/sessions/{session_id}/learning/infer",
    response_model=InferenceResult,
)
def run_inference(
    session_id: str,
    data: RunInferenceRequest,
    db: DbSession = Depends(get_db),
):
    return learning_service.run_inference(db, session_id, data.model_id)


@router.get(
    "/sessions/{session_id}/learning/models",
    response_model=list[TrainedModelOut],
)
def list_session_models(session_id: str, db: DbSession = Depends(get_db)):
    return learning_service.list_models(db, session_id=session_id)


# ── Global model registry ────────────────────────────────────────────


@router.get("/models/", response_model=list[TrainedModelOut])
def list_all_models(db: DbSession = Depends(get_db)):
    return learning_service.list_models(db)


@router.get("/models/{model_id}", response_model=TrainedModelOut)
def get_model(model_id: str, db: DbSession = Depends(get_db)):
    return learning_service.get_model(db, model_id)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str, db: DbSession = Depends(get_db)):
    learning_service.delete_model(db, model_id)


@router.post("/models/{model_id}/apply", response_model=InferenceResult)
def apply_model(
    model_id: str,
    data: ApplyModelRequest,
    db: DbSession = Depends(get_db),
):
    return learning_service.apply_model_cross_session(
        db, model_id, data.target_session_id
    )
