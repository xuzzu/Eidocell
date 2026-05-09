"""Feature extraction service: build feature vectors from mask attributes or images.

Single async entrypoint — extraction always runs as a background task and is
streamed into the per-session LanceDB features table in batches so RSS stays
bounded and a cancel cleans up cleanly (Lance MVCC drops in-flight batches).
"""

import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.processors.errors import UnknownProcessorError
from core.processors.inference.feature_extraction import (
    FEATURE_EXTRACTION_REGISTRY,
    get_processor,
)
from core.task_manager import task_manager
from services._pipeline_utils import (
    extract_features_to_lance,
    preload_morphological_masks,
    snapshot_samples,
    validate_session_and_active_samples,
)

logger = logging.getLogger("eidocell.features")


def run_feature_extraction(
    db: DbSession, session_id: str, method: str
) -> str:
    """Submit feature extraction as a background task. Returns task_id."""
    try:
        processor = get_processor(method)
    except UnknownProcessorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session, samples = validate_session_and_active_samples(db, session_id)
    sample_data = snapshot_samples(samples)
    masks = preload_morphological_masks(session.id, sample_data, processor)

    return task_manager.submit(
        name=f"Feature extraction ({method})",
        func=_background_extract,
        method=method,
        sample_data=sample_data,
        session_id=session.id,
        masks=masks,
    )


def _background_extract(
    *,
    method: str,
    sample_data: list[dict],
    session_id: str,
    masks: dict,
    on_progress,
    is_cancelled=None,
):
    """Run feature extraction in a background thread."""
    processor = get_processor(method)
    feature_dim = processor.feature_dim()

    processed, skipped = extract_features_to_lance(
        sample_data=sample_data,
        processor=processor,
        masks=masks,
        session_id=session_id,
        method=method,
        feature_dim=feature_dim,
        on_progress=on_progress,
        is_cancelled=is_cancelled,
    )
    return {
        "processed": processed,
        "skipped": skipped,
        "total": len(sample_data),
        "feature_dim": feature_dim,
    }


def list_available_methods() -> list[dict]:
    return [
        {
            "id": method_id,
            "name": method_id.replace("_", " ").title(),
            "feature_dim": processor.feature_dim(),
        }
        for method_id, processor in FEATURE_EXTRACTION_REGISTRY.items()
    ]
