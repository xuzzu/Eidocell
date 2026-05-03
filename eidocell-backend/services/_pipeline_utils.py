"""Shared helpers for processor-driven services (FE / DR / clustering pipeline).

These exist to remove the duplication between feature_extraction_service,
dimensionality_reduction_service, and clustering_pipeline_service:

- Session/sample validation, snapshotting for thread handoff
- Mask preloading for morphological extraction
- Streaming feature extraction into a memmap (bounds RSS, cleans up on cancel)
- Zero-variance column filtering
- Thread-local DB session contextmanager (worker side)
- Task progress contextmanager (worker side, replaces dead is_cancelled checks)
"""
from __future__ import annotations
from core.utils import npy_load

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DbSession, sessionmaker

from core.processors.errors import (
    ProcessorError,
    ProcessorInputError,
)
from core.processors.inference.feature_extraction import (
    FeatureExtractionProcessor,
    MorphologicalFeatureExtraction,
)
from core.task_manager import TaskCancelledException
from core.utils import get_active_samples
from models.models import Mask, Sample, SampleClass, Session

logger = logging.getLogger("eidocell.pipeline_utils")

# Default flush interval for the streaming extractor.
# Tunes the trade-off between disk I/O and RSS exposure on cancel.
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
    return [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]


def preload_morphological_masks(
    db: DbSession, sample_data: list[dict], processor: FeatureExtractionProcessor
) -> dict[str, dict]:
    """If `processor` is morphological, eagerly load mask attributes keyed by sample id."""
    if not isinstance(processor, MorphologicalFeatureExtraction):
        return {}
    sample_ids = [s["id"] for s in sample_data]
    return {
        m.sample_id: m.attributes
        for m in db.query(Mask).filter(Mask.sample_id.in_(sample_ids)).all()
        if m.attributes
    }

def features_cached_for(
    features_path: Path, sample_data: list[dict], feature_dim: int
) -> bool:
    """Return True if `features_path` has non-zero features for every active sample."""
    if not features_path.exists():
        return False
    try:
        existing = npy_load(features_path)
    except Exception:
        return False
    max_index = max(s["storage_index"] for s in sample_data) + 1
    if existing.ndim != 2 or existing.shape[0] < max_index or existing.shape[1] != feature_dim:
        return False
    indices = [s["storage_index"] for s in sample_data]
    rows = existing[indices]
    return bool(np.all(np.abs(rows).sum(axis=1) > 0))

# ── Streaming feature extraction ────────────────────────────────────────


def extract_features_into(
    *,
    sample_data: list[dict],
    processor: FeatureExtractionProcessor,
    masks: dict[str, dict],
    features_path: Path,
    feature_dim: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> tuple[int, int]:
    """Extract features for all samples, streaming through a memmap.

    Writes to `<features_path>.tmp` while running, flushing every `flush_every`
    samples to bound RSS. On success the temp file is atomically renamed to
    `features_path`. On cancel/error the temp file is unlinked and the previous
    `features_path` (if any) is left intact.

    Returns (processed, skipped). Reasons for skipping a sample (bad input,
    runtime failure, missing mask) are logged but never abort the run.
    """
    features_path.parent.mkdir(parents=True, exist_ok=True)
    max_index = max(s["storage_index"] for s in sample_data) + 1
    tmp_path = features_path.with_suffix(features_path.suffix + ".tmp")

    # Open a real .npy-format memmap (proper header) so the file is loadable by
    # np.load on success without any post-processing. Seed it with previous
    # features when compatible so unprocessed rows preserve their values.
    memmap = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(max_index, feature_dim),
    )
    _seed_memmap_from_existing(features_path, memmap, max_index, feature_dim)

    is_morphological = isinstance(processor, MorphologicalFeatureExtraction)
    total = len(sample_data)
    processed = 0
    skipped = 0

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
                    image = cv2.imread(sd["path"], cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ProcessorInputError(f"failed to read image {sd['path']}")
                    features = processor.extract(image)
            except ProcessorError as e:
                logger.warning("sample %s skipped: %s", sd["id"], e)
                skipped += 1
            else:
                memmap[sd["storage_index"]] = features
                processed += 1

            if (i + 1) % flush_every == 0:
                memmap.flush()
            if on_progress:
                on_progress(i + 1, total, f"Processed {i + 1}/{total}")

        memmap.flush()
        # Drop the memmap before renaming so Windows-style file locks (and
        # numpy's lazy file handles) release the temp path.
        del memmap
        os.replace(tmp_path, features_path)
        logger.info(
            "Feature extraction complete: %d processed, %d skipped", processed, skipped
        )
        return processed, skipped

    except BaseException:
        # Cancel, error, or process death: drop the memmap and unlink the temp
        # file so the previous features.npy remains the source of truth.
        try:
            del memmap
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("failed to clean up %s", tmp_path)
        raise


def _seed_memmap_from_existing(
    features_path: Path,
    memmap: np.ndarray,
    max_index: int,
    feature_dim: int,
) -> None:
    """Copy any compatible existing features into the freshly-allocated memmap."""
    if not features_path.exists():
        return
    try:
        existing = np.load(features_path, mmap_mode="r")
    except Exception:
        logger.warning("could not load existing features at %s; starting fresh", features_path)
        return
    if existing.ndim != 2 or existing.shape[1] != feature_dim:
        logger.warning(
            "existing features at %s have incompatible shape %s; starting fresh",
            features_path,
            existing.shape,
        )
        return
    rows = min(existing.shape[0], max_index)
    memmap[:rows] = existing[:rows]


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
    """Open a fresh DB session bound to a new engine for use in a worker thread.

    Always disposes the engine and closes the session, even on error. Use this
    instead of hand-rolling create_engine/sessionmaker in every background worker.
    """
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


class _TaskContext:
    """Worker-side helper bundling progress updates and cooperative cancellation.

    Replaces the pattern of manual `if is_cancelled(): pass` blocks scattered
    through workers — call ctx.tick() at each iteration and ctx.stage() between
    steps; both raise TaskCancelledException promptly if the task was cancelled.
    """

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
        """Mark progress for a coarse pipeline stage transition."""
        self._check_cancel()
        if advance:
            self.current += advance
        if self._on_progress:
            self._on_progress(self.current, self.total or 1, message)

    def tick(self, current: int, message: str) -> None:
        """Mark progress after processing a single item."""
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
        # Sessions auto-assign new samples to an "Uncategorized" class.
        # An "unlabeled" sample is one with class_id IS NULL or pointing to
        # that placeholder class.
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

    samples = q.order_by(Sample.storage_index).all()
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
    "extract_features_into",
    "clean_zero_variance",
    "thread_db_session",
    "task_context",
    "DEFAULT_FLUSH_EVERY",
    "project_to_2d",
    "resolve_scoped_samples",
    "compute_quality",
]
