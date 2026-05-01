"""Dimensionality reduction service: project features into 2D/3D for visualization."""

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.processors.inference.dimensionality_reduction import (
    get_processor,
    DR_REGISTRY,
)
from core.utils import load_session_features, get_active_samples
from models.models import Session, Sample


def run_dimensionality_reduction(
    db: DbSession,
    session_id: str,
    method: str,
    n_components: int,
    params: dict,
) -> dict:
    try:
        processor = get_processor(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    samples = get_active_samples(db, session_id)
    if not samples:
        raise HTTPException(status_code=400, detail="No active samples")

    indices = [s.storage_index for s in samples]
    features = load_session_features(session.session_folder, indices)

    # Check for zero-variance features and remove them
    stds = np.std(features, axis=0)
    valid_cols = stds > 0
    if not np.any(valid_cols):
        raise HTTPException(
            status_code=400,
            detail="All features have zero variance. Cannot reduce dimensionality.",
        )
    features_clean = features[:, valid_cols]

    if features_clean.shape[0] < n_components:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {n_components} samples, got {features_clean.shape[0]}",
        )

    embeddings = processor.fit_transform(
        features_clean, n_components=n_components, **params
    )

    # Build response with sample info
    result_embeddings = []
    for i, s in enumerate(samples):
        entry = {
            "sample_id": s.id,
            "filename": s.filename,
            "class_id": s.class_id,
            "x": float(embeddings[i, 0]),
            "y": float(embeddings[i, 1]),
        }
        if n_components == 3:
            entry["z"] = float(embeddings[i, 2])
        result_embeddings.append(entry)

    return {
        "method": method,
        "n_components": n_components,
        "n_samples": len(samples),
        "embeddings": result_embeddings,
    }


def list_available_methods() -> list[dict]:
    return [{"id": k, "name": k.upper()} for k in DR_REGISTRY]
