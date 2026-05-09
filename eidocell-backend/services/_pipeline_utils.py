"""Shared helpers for processor-driven services (FE / DR / clustering pipeline).

These exist to remove the duplication between feature_extraction_service,
dimensionality_reduction_service, and clustering_pipeline_service:

- Session/sample validation, snapshotting for thread handoff
- Mask attribute preloading for morphological extraction (from Lance)
- Streaming feature extraction into the per-session LanceDB features table
- Zero-variance column filtering
- Thread-local DB session contextmanager (worker side)
- Task progress contextmanager (worker side, replaces dead is_cancelled checks)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Iterator

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DbSession, sessionmaker

from core.image_io import read as read_image
from core.processors.errors import (
    ProcessorError,
    ProcessorInputError,
)
from core.processors.inference.feature_extraction import (
    FeatureExtractionProcessor,
    MorphologicalFeatureExtraction,
)
from core.storage import features as lance_features
from core.storage import mask_attrs as lance_mask_attrs
from core.task_manager import TaskCancelledException
from core.utils import get_active_samples
from models.models import Sample, SampleClass, Session

logger = logging.getLogger("eidocell.pipeline_utils")

# Default flush interval for the streaming extractor. Tunes the trade-off
# between Lance write throughput and how much in-flight work a cancel discards.
DEFAULT_FLUSH_EVERY = 256


# ── Validation / snapshotting ───────────────────────────────────────────


def validate_session_and_active_samples(
    db: DbSession, session_id: str
) -> tuple[Session, list[Sample]]:
    """Look up the session and active samples; raise HTTPException if either is missing."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    samples = get_active_samples(db, session_id)
    if not samples:
        raise HTTPException(status_code=400, detail="No active samples")
    return session, samples


def snapshot_samples(samples: list[Sample]) -> list[dict]:
    """Snapshot ORM rows into plain dicts so a worker thread can use them safely."""
    return [{"id": s.id, "path": s.path} for s in samples]


def preload_morphological_masks(
    session_id: str,
    sample_data: list[dict],
    processor: FeatureExtractionProcessor,
) -> dict[str, dict]:
    """If `processor` is morphological, eagerly load attrs from Lance keyed by sample id."""
    if not isinstance(processor, MorphologicalFeatureExtraction):
        return {}
    sample_ids = [s["id"] for s in sample_data]
    bulk = lance_mask_attrs.get_attrs_bulk(session_id, sample_ids)
    return {sid: attrs for sid, attrs in bulk.items() if attrs}


def features_cached_for(
    session_id: str,
    method: str,
    sample_data: list[dict],
) -> bool:
    """Return True if every sample has a feature row for this method."""
    sample_ids = [s["id"] for s in sample_data]
    if not sample_ids:
        return True
    have = lance_features.has_method(session_id, method, sample_ids)
    return have >= set(sample_ids)


# ── Streaming feature extraction ────────────────────────────────────────


def extract_features_to_lance(
    *,
    sample_data: list[dict],
    processor: FeatureExtractionProcessor,
    masks: dict[str, dict],
    session_id: str,
    method: str,
    feature_dim: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> tuple[int, int]:
    """Extract features for all samples and stream them into LanceDB.

    Buffers `flush_every` rows then upserts via merge_insert. On cancel/error
    the partial batch is dropped; Lance MVCC keeps already-committed rows.
    Returns (processed, skipped).
    """
    is_morphological = isinstance(processor, MorphologicalFeatureExtraction)
    total = len(sample_data)
    processed = 0
    skipped = 0
    buffer: list[dict] = []

    def _flush():
        nonlocal buffer
        if buffer:
            lance_features.upsert_features(session_id, method, buffer)
            buffer = []

    try:
        if on_progress:
            on_progress(0, total, "Starting feature extraction...")

        for i, sd in enumerate(sample_data):
            if is_cancelled and is_cancelled():
                raise TaskCancelledException("Feature extraction cancelled")

            try:
                if is_morphological:
                    features = processor.extract_from_attributes(masks.get(sd["id"]))
                else:
                    image, _ = read_image(sd["path"])
                    if image is None:
                        raise ProcessorInputError(f"failed to read image {sd['path']}")
                    features = processor.extract(image)
            except ProcessorError as e:
                logger.warning("sample %s skipped: %s", sd["id"], e)
                skipped += 1
            else:
                vec = np.asarray(features, dtype=np.float32).ravel()
                if vec.shape[0] != feature_dim:
                    logger.warning(
                        "sample %s: dim mismatch %s vs %s; skipping",
                        sd["id"], vec.shape[0], feature_dim,
                    )
                    skipped += 1
                else:
                    buffer.append({"sample_id": sd["id"], "vector": vec})
                    processed += 1

            if len(buffer) >= flush_every:
                _flush()
            if on_progress:
                on_progress(i + 1, total, f"Processed {i + 1}/{total}")

        _flush()
        logger.info(
            "Feature extraction complete: %d processed, %d skipped", processed, skipped
        )
        return processed, skipped
    except BaseException:
        # Drop unflushed work; whatever made it to Lance stays committed.
        buffer = []
        raise


# ── Misc utilities ──────────────────────────────────────────────────────


def clean_zero_variance(
    features: np.ndarray, *, raise_on_all_zero: bool = False
) -> np.ndarray:
    """Drop columns with zero variance. Optionally raise HTTP 400 if all are zero."""
    stds = np.std(features, axis=0)
    valid = stds > 0
    if not np.any(valid):
        if raise_on_all_zero:
            raise HTTPException(
                status_code=400,
                detail="All features have zero variance. Cannot proceed.",
            )
        return features
    return features[:, valid]


@contextmanager
def thread_db_session(db_url: str) -> Iterator[DbSession]:
    """Open a fresh DB session bound to a new engine for use in a worker thread."""
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


class _TaskContext:
    """Worker-side helper bundling progress updates and cooperative cancellation."""

    def __init__(
        self,
        on_progress: Callable[[int, int, str], None] | None,
        is_cancelled: Callable[[], bool] | None,
        total: int = 0,
    ) -> None:
        self._on_progress = on_progress
        self._is_cancelled = is_cancelled
        self.total = total
        self.current = 0

    def set_total(self, total: int) -> None:
        self.total = total

    def _check_cancel(self) -> None:
        if self._is_cancelled and self._is_cancelled():
            raise TaskCancelledException("Task was cancelled")

    def stage(self, message: str, *, advance: int = 0) -> None:
        self._check_cancel()
        if advance:
            self.current += advance
        if self._on_progress:
            self._on_progress(self.current, self.total or 1, message)

    def tick(self, current: int, message: str) -> None:
        self._check_cancel()
        self.current = current
        if self._on_progress:
            self._on_progress(current, self.total or 1, message)


@contextmanager
def task_context(
    on_progress: Callable[[int, int, str], None] | None,
    is_cancelled: Callable[[], bool] | None,
    *,
    total: int = 0,
) -> Iterator[_TaskContext]:
    yield _TaskContext(on_progress, is_cancelled, total=total)


def project_to_2d(features: np.ndarray) -> np.ndarray:
    if features.shape[1] == 2:
        return features
    if features.shape[1] == 1:
        return np.column_stack([features, np.zeros(len(features))])
    from sklearn.decomposition import PCA

    n_comp = min(2, features.shape[0], features.shape[1])
    return PCA(n_components=n_comp).fit_transform(features)


# ── Scope resolution ────────────────────────────────────────────────────


def resolve_scoped_samples(
    db: DbSession, session_id: str, scope
) -> tuple[Session, list[Sample]]:
    """Resolve a `ClusteringScope` to (session, samples).

    Always restricts to active samples. Raises 400 if the resolved set is empty.
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    mode = getattr(scope, "mode", "all") if scope is not None else "all"
    q = (
        db.query(Sample)
        .filter(Sample.session_id == session_id, Sample.is_active == True)  # noqa: E712
    )

    if mode == "unlabeled":
        uncat = (
            db.query(SampleClass)
            .filter(
                SampleClass.session_id == session_id,
                SampleClass.name == "Uncategorized",
            )
            .first()
        )
        if uncat is None:
            q = q.filter(Sample.class_id.is_(None))
        else:
            q = q.filter(
                (Sample.class_id.is_(None)) | (Sample.class_id == uncat.id)
            )
    elif mode == "class":
        class_id = getattr(scope, "class_id", None)
        if not class_id:
            raise HTTPException(status_code=400, detail="scope.class_id required for mode='class'")
        q = q.filter(Sample.class_id == class_id)
    elif mode == "samples":
        sample_ids = getattr(scope, "sample_ids", None) or []
        if not sample_ids:
            raise HTTPException(status_code=400, detail="scope.sample_ids required for mode='samples'")
        q = q.filter(Sample.id.in_(sample_ids))
    elif mode != "all":
        raise HTTPException(status_code=400, detail=f"Unknown scope mode: {mode}")

    samples = q.order_by(Sample.filename).all()
    if not samples:
        raise HTTPException(
            status_code=400,
            detail=f"No active samples match scope (mode={mode})",
        )
    return session, samples


# ── Cluster quality metric ──────────────────────────────────────────────


def compute_quality(
    features: np.ndarray, labels: np.ndarray, k: int
) -> float | None:
    """Mean Euclidean distance to centroid for cluster `k`.

    Lower = tighter. Returns None for clusters with fewer than 2 members.
    """
    rows = features[labels == k]
    if len(rows) < 2:
        return None
    centroid = rows.mean(axis=0)
    return float(np.linalg.norm(rows - centroid, axis=1).mean())


__all__ = [
    "validate_session_and_active_samples",
    "snapshot_samples",
    "preload_morphological_masks",
    "features_cached_for",
    "extract_features_to_lance",
    "clean_zero_variance",
    "thread_db_session",
    "task_context",
    "DEFAULT_FLUSH_EVERY",
    "project_to_2d",
    "resolve_scoped_samples",
    "compute_quality",
]
