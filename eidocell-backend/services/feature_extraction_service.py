"""Feature extraction service: build feature vectors from mask attributes or images."""

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.processors.inference.feature_extraction import (
    get_processor,
    MorphologicalFeatureExtraction,
    FEATURE_EXTRACTION_REGISTRY,
)
from core.task_manager import task_manager
from core.utils import get_active_samples, npy_save, npy_load
from models.models import Session, Sample, Mask

logger = logging.getLogger("eidocell.features")


def run_feature_extraction(
    db: DbSession, session_id: str, method: str
) -> dict:
    """Run feature extraction synchronously (for morphological / small datasets)."""
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

    # Snapshot data to avoid thread/session issues
    sample_data = [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]

    # Pre-load mask attributes for morphological
    masks = {}
    if isinstance(processor, MorphologicalFeatureExtraction):
        sample_ids = [s["id"] for s in sample_data]
        masks = {
            m.sample_id: m.attributes
            for m in db.query(Mask).filter(Mask.sample_id.in_(sample_ids)).all()
            if m.attributes
        }

    feature_dim = processor.feature_dim()
    max_index = max(s["storage_index"] for s in sample_data) + 1

    features_dir = Path(session.session_folder) / "features"
    features_dir.mkdir(exist_ok=True)
    features_path = features_dir / "session_features.npy"

    all_features = _load_or_create_features(features_path, max_index, feature_dim)

    processed = 0
    skipped = 0

    for sd in sample_data:
        if isinstance(processor, MorphologicalFeatureExtraction):
            attrs = masks.get(sd["id"])
            if not attrs:
                skipped += 1
                continue
            features = processor.extract_from_attributes(attrs)
        else:
            image = cv2.imread(sd["path"], cv2.IMREAD_UNCHANGED)
            if image is None:
                skipped += 1
                continue
            try:
                features = processor.extract(image)
            except Exception:
                skipped += 1
                continue

        all_features[sd["storage_index"]] = features
        processed += 1

    npy_save(features_path, all_features)
    logger.info("Feature extraction complete: %d processed, %d skipped", processed, skipped)

    return {
        "processed": processed,
        "skipped": skipped,
        "total": len(sample_data),
        "feature_dim": feature_dim,
    }


def run_feature_extraction_async(
    db: DbSession, session_id: str, method: str
) -> str:
    """Start feature extraction as a background task. Returns task ID."""
    try:
        get_processor(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    samples = get_active_samples(db, session_id)
    if not samples:
        raise HTTPException(status_code=400, detail="No active samples")

    # Snapshot everything needed for the background thread
    sample_data = [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]
    session_folder = session.session_folder

    # For morphological, pre-load mask attributes
    masks = {}
    processor = get_processor(method)
    if isinstance(processor, MorphologicalFeatureExtraction):
        sample_ids = [s["id"] for s in sample_data]
        masks = {
            m.sample_id: m.attributes
            for m in db.query(Mask).filter(Mask.sample_id.in_(sample_ids)).all()
            if m.attributes
        }

    # Get the DB URL from the current session's bind
    db_url = str(db.get_bind().url)

    task_id = task_manager.submit(
        name=f"Feature extraction ({method})",
        func=_background_extract,
        method=method,
        sample_data=sample_data,
        session_folder=session_folder,
        masks=masks,
    )
    return task_id


def _background_extract(
    *,
    method: str,
    sample_data: list[dict],
    session_folder: str,
    masks: dict,
    on_progress,
    is_cancelled=None,
):
    """Run feature extraction in a background thread."""
    processor = get_processor(method)
    feature_dim = processor.feature_dim()
    max_index = max(s["storage_index"] for s in sample_data) + 1

    features_dir = Path(session_folder) / "features"
    features_dir.mkdir(exist_ok=True)
    features_path = features_dir / "session_features.npy"

    all_features = _load_or_create_features(features_path, max_index, feature_dim)

    total = len(sample_data)
    processed = 0
    skipped = 0

    on_progress(0, total, "Starting feature extraction...")

    for i, sd in enumerate(sample_data):
        if is_cancelled and is_cancelled():
            # The task manager will catch the exception raised by on_progress shortly, 
            # or we can break. we can also just let on_progress raise it! Wait, 
            # if we just call on_progress, it raises TaskCancelledException.
            pass

        if isinstance(processor, MorphologicalFeatureExtraction):
            attrs = masks.get(sd["id"])
            if not attrs:
                skipped += 1
            else:
                features = processor.extract_from_attributes(attrs)
                all_features[sd["storage_index"]] = features
                processed += 1
        else:
            image = cv2.imread(sd["path"], cv2.IMREAD_UNCHANGED)
            if image is None:
                skipped += 1
            else:
                try:
                    features = processor.extract(image)
                    all_features[sd["storage_index"]] = features
                    processed += 1
                except Exception:
                    skipped += 1

        on_progress(i + 1, total, f"Processed {i + 1}/{total}")

    npy_save(features_path, all_features)
    logger.info("Background feature extraction complete: %d processed, %d skipped", processed, skipped)

    return {
        "processed": processed,
        "skipped": skipped,
        "total": total,
        "feature_dim": feature_dim,
    }


def _load_or_create_features(
    features_path: Path, max_index: int, feature_dim: int
) -> np.ndarray:
    if features_path.exists():
        all_features = npy_load(features_path)
        if all_features.shape[0] < max_index or all_features.shape[1] != feature_dim:
            new_arr = np.zeros((max_index, feature_dim), dtype=np.float32)
            rows = min(all_features.shape[0], max_index)
            if all_features.shape[1] == feature_dim:
                new_arr[:rows] = all_features[:rows]
            all_features = new_arr
    else:
        all_features = np.zeros((max_index, feature_dim), dtype=np.float32)
    return all_features


def list_available_methods() -> list[dict]:
    results = []
    for method_id, processor in FEATURE_EXTRACTION_REGISTRY.items():
        results.append({
            "id": method_id,
            "name": method_id.replace("_", " ").title(),
            "feature_dim": processor.feature_dim(),
        })
    return results
