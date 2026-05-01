"""Dimensionality reduction service: project features into 2D/3D for visualization.

Runs as a background task — UMAP/t-SNE on tens of thousands of points can be
multi-second to multi-minute work, so the request returns a task_id immediately
and clients poll via /tasks/{id}.
"""

import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.processors.errors import UnknownProcessorError
from core.processors.inference.dimensionality_reduction import (
    DR_REGISTRY,
    get_processor,
)
from core.task_manager import task_manager
from core.utils import load_session_features
from services._pipeline_utils import (
    clean_zero_variance,
    task_context,
    validate_session_and_active_samples,
)

logger = logging.getLogger("eidocell.dr")


def run_dimensionality_reduction(
    db: DbSession,
    session_id: str,
    method: str,
    n_components: int,
    params: dict,
) -> str:
    """Submit dimensionality reduction as a background task. Returns task_id."""
    try:
        get_processor(method)
    except UnknownProcessorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session, samples = validate_session_and_active_samples(db, session_id)
    if len(samples) < n_components:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {n_components} samples, got {len(samples)}",
        )

    sample_meta = [
        {
            "id": s.id,
            "filename": s.filename,
            "class_id": s.class_id,
            "storage_index": s.storage_index,
        }
        for s in samples
    ]

    return task_manager.submit(
        name=f"Dimensionality reduction ({method})",
        func=_background_dr,
        session_folder=session.session_folder,
        sample_meta=sample_meta,
        method=method,
        n_components=n_components,
        params=params or {},
    )


def _background_dr(
    *,
    session_folder: str,
    sample_meta: list[dict],
    method: str,
    n_components: int,
    params: dict,
    on_progress,
    is_cancelled=None,
):
    """Run dimensionality reduction in a background thread."""
    with task_context(on_progress, is_cancelled, total=4) as ctx:
        ctx.stage("Loading features...", advance=1)
        indices = [s["storage_index"] for s in sample_meta]
        features = load_session_features(session_folder, indices)

        ctx.stage("Cleaning features...", advance=1)
        features_clean = clean_zero_variance(features, raise_on_all_zero=True)

        ctx.stage(f"Fitting {method.upper()}...", advance=1)
        processor = get_processor(method)
        embeddings = processor.fit_transform(
            features_clean, n_components=n_components, **params
        )

        ctx.stage("Building response...", advance=1)
        result_embeddings = []
        for i, s in enumerate(sample_meta):
            entry = {
                "sample_id": s["id"],
                "filename": s["filename"],
                "class_id": s["class_id"],
                "x": float(embeddings[i, 0]),
                "y": float(embeddings[i, 1]),
            }
            if n_components == 3:
                entry["z"] = float(embeddings[i, 2])
            result_embeddings.append(entry)

    return {
        "method": method,
        "n_components": n_components,
        "n_samples": len(sample_meta),
        "embeddings": result_embeddings,
    }


def list_available_methods() -> list[dict]:
    return [{"id": k, "name": k.upper()} for k in DR_REGISTRY]
