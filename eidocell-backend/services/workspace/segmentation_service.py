"""Segmentation service: run segmentation, store masks, compute attributes."""

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.image_io import read as read_image
from core.processors.image_utils import array_to_uint8, channel_to_uint8, generate_mask_overlay
from core.processors.inference.mask_attributes import compute_mask_attributes
from core.processors.inference.segmentation import get_processor, list_methods
from core.storage import images as lance_images
from core.storage import mask_attrs as lance_mask_attrs
from core.storage import masks as mask_files
from core.task_manager import task_manager
from core.utils import get_active_samples
from models.models import Mask, Sample, Session
from services._pipeline_utils import thread_db_session

logger = logging.getLogger("eidocell.segmentation")


_OVERLAYS_SUBDIR = "previews/overlays"


def _overlay_path(session_folder: Path, sample_id: str, channel_index: int = 0) -> Path:
    return session_folder / _OVERLAYS_SUBDIR / f"{sample_id}_ch{int(channel_index)}.png"


def _load_channel_image(
    session_id: str,
    sample_id: str,
    src_path: Path,
    channel_index: int,
) -> np.ndarray | None:
    """Load the requested channel as a 2-D uint8 ndarray, preferring Lance.

    Normalises to uint8 because downstream OpenCV ops (GaussianBlur, etc.)
    reject CV_32S and other unusual dtypes that Lance happily round-trips.
    Falls back to the legacy file path on disk when no Lance row exists.
    """
    arr = lance_images.read_array(session_id, sample_id)
    if arr is not None:
        if arr.ndim == 2:
            if channel_index != 0:
                return None
            return array_to_uint8(arr)
        if arr.ndim == 3:
            n = arr.shape[2]
            if channel_index < 0 or channel_index >= n:
                return None
            return channel_to_uint8(arr, channel_index)
        return None
    # Legacy fallback — no Lance array. Use disk file.
    image, _channels = read_image(src_path)
    if image is None:
        return None
    if image.ndim == 2:
        if channel_index != 0:
            return None
        return array_to_uint8(image)
    if image.ndim == 3:
        n = image.shape[2]
        if channel_index < 0 or channel_index >= n:
            return None
        return channel_to_uint8(image, channel_index)
    return None


def _segment_one(
    src_path: Path,
    processor,
    params: dict,
    scale_factor: float,
    session_folder: Path,
    sample_id: str,
    channel_index: int = 0,
    session_id: str | None = None,
):
    """Read the chosen channel, segment, write mask + overlay to disk, return (attrs, ok)."""
    channel_img = None
    if session_id is not None:
        channel_img = _load_channel_image(session_id, sample_id, src_path, channel_index)
    if channel_img is None:
        # No Lance array and no session_id provided — try direct file read.
        image, _channels = read_image(src_path)
        if image is None:
            return None, False
        if image.ndim == 2 and channel_index == 0:
            channel_img = array_to_uint8(image)
        elif image.ndim == 3 and 0 <= channel_index < image.shape[2]:
            channel_img = channel_to_uint8(image, channel_index)
        if channel_img is None:
            return None, False

    mask_data = processor.segment(channel_img, **params)
    attrs = compute_mask_attributes(channel_img, mask_data, scale_factor)

    mask_files.write_mask(session_folder, sample_id, mask_data, channel_index=channel_index)

    overlay = generate_mask_overlay(channel_img, mask_data)
    overlay_path = _overlay_path(session_folder, sample_id, channel_index)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return attrs, True


def _persist_results(
    db: DbSession,
    session_id: str,
    method: str,
    new_mask_keys: list[tuple[str, int]],
    attr_rows: list[dict],
) -> None:
    """Insert/update Mask rows for (sample_id, channel_index) keys and upsert Lance attribute rows."""
    if new_mask_keys:
        for sid, ch in new_mask_keys:
            existing = db.query(Mask).filter(
                Mask.sample_id == sid, Mask.channel_index == ch,
            ).first()
            if existing is not None:
                existing.segmentation_method = method
            else:
                db.add(Mask(sample_id=sid, channel_index=ch, segmentation_method=method))
        db.commit()

    if attr_rows:
        lance_mask_attrs.upsert_attrs(session_id, attr_rows)


def _validate_channel_index(channel_index: int, session: Session) -> None:
    ch = int(channel_index or 0)
    n = max(1, int(session.channel_count or 1))
    if ch < 0 or ch >= n:
        raise HTTPException(
            status_code=400,
            detail=f"channel_index {ch} out of range (session has {n} channel(s))",
        )


def run_segmentation(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
    *,
    channel_index: int = 0,
) -> dict:
    """Run segmentation on all active samples in the session."""
    try:
        processor = get_processor(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _validate_channel_index(channel_index, session)

    samples = get_active_samples(db, session_id)
    if not samples:
        raise HTTPException(status_code=400, detail="No active samples")

    session_folder = Path(session.session_folder)
    sample_data = [{"id": s.id, "path": s.path} for s in samples]
    return _run_for_samples(
        db, session, processor, method, params, sample_data, session_folder,
        channel_index=channel_index,
    )


def run_segmentation_preview(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
    sample_ids: list[str],
    *,
    channel_index: int = 0,
) -> dict:
    """Run segmentation on specific samples (committed to disk + DB)."""
    try:
        processor = get_processor(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _validate_channel_index(channel_index, session)

    samples = db.query(Sample).filter(
        Sample.id.in_(sample_ids),
        Sample.session_id == session_id,
    ).all()
    if not samples:
        raise HTTPException(status_code=400, detail="No matching samples found")

    session_folder = Path(session.session_folder)
    sample_data = [{"id": s.id, "path": s.path} for s in samples]
    return _run_for_samples(
        db, session, processor, method, params, sample_data, session_folder,
        channel_index=channel_index,
    )


def _run_for_samples(
    db: DbSession,
    session: Session,
    processor,
    method: str,
    params: dict,
    sample_data: list[dict],
    session_folder: Path,
    channel_index: int = 0,
) -> dict:
    scale_factor = session.scale_factor
    processed = 0
    failed = 0
    attr_rows: list[dict] = []
    new_mask_keys: list[tuple[str, int]] = []

    for sd in sample_data:
        src_path = Path(sd["path"])
        try:
            attrs, ok = _segment_one(
                src_path, processor, params, scale_factor, session_folder, sd["id"], channel_index,
                session_id=session.id,
            )
            if not ok:
                failed += 1
                continue
            new_mask_keys.append((sd["id"], channel_index))
            attr_rows.append({"sample_id": sd["id"], "channel_index": channel_index, **(attrs or {})})
            processed += 1
        except Exception:
            logger.exception("segmentation failed for sample %s", sd["id"])
            failed += 1

    _persist_results(db, session.id, method, new_mask_keys, attr_rows)
    logger.info("Segmentation complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": len(sample_data)}


def stream_preview_overlays(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
    sample_ids: list[str],
    *,
    channel_index: int = 0,
):
    """Yield (sample_id, overlay_png_bytes, attributes) per sample.

    Used by the WebSocket preview endpoint. Does NOT write masks/overlays
    to disk or DB — preview is throwaway.
    """
    processor = get_processor(method)

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _validate_channel_index(channel_index, session)

    samples = db.query(Sample).filter(
        Sample.id.in_(sample_ids),
        Sample.session_id == session_id,
    ).all()
    samples_by_id = {s.id: s for s in samples}
    ordered = [samples_by_id[sid] for sid in sample_ids if sid in samples_by_id]

    scale_factor = session.scale_factor

    for s in ordered:
        try:
            channel_img = _load_channel_image(session_id, s.id, Path(s.path), channel_index)
            if channel_img is None:
                yield s.id, None, None
                continue
            mask_data = processor.segment(channel_img, **params)
            attrs = compute_mask_attributes(channel_img, mask_data, scale_factor)
            overlay = generate_mask_overlay(channel_img, mask_data)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                yield s.id, None, None
                continue
            yield s.id, bytes(buf), attrs
        except Exception:
            logger.exception("preview failed for sample %s", s.id)
            yield s.id, None, None


def run_segmentation_async(
    db: DbSession, session_id: str, method: str, params: dict,
    *,
    channel_index: int = 0,
) -> str:
    """Start segmentation as a background task. Returns task ID."""
    try:
        get_processor(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _validate_channel_index(channel_index, session)

    samples = get_active_samples(db, session_id)
    if not samples:
        raise HTTPException(status_code=400, detail="No active samples")

    sample_data = [{"id": s.id, "path": s.path} for s in samples]
    db_url = str(db.get_bind().url)

    return task_manager.submit(
        name=f"Segmentation ({method})",
        func=_background_segment,
        method=method,
        params=params,
        sample_data=sample_data,
        session_id=session.id,
        session_folder=session.session_folder,
        scale_factor=session.scale_factor,
        db_url=db_url,
        channel_index=channel_index,
    )


def _background_segment(
    *,
    method: str,
    params: dict,
    sample_data: list[dict],
    session_id: str,
    session_folder: str,
    scale_factor: float,
    db_url: str,
    on_progress,
    is_cancelled=None,
    channel_index: int = 0,
):
    """Run segmentation in a background thread."""
    processor = get_processor(method)
    session_folder = Path(session_folder)

    total = len(sample_data)
    processed = 0
    failed = 0
    attr_rows: list[dict] = []
    new_mask_keys: list[tuple[str, int]] = []

    on_progress(0, total, "Starting segmentation...")

    for i, sd in enumerate(sample_data):
        if is_cancelled and is_cancelled():
            pass  # TaskCancelledException raised by on_progress

        src_path = Path(sd["path"])

        try:
            attrs, ok = _segment_one(
                src_path, processor, params, scale_factor, session_folder, sd["id"], channel_index,
                session_id=session_id,
            )
            if not ok:
                failed += 1
            else:
                new_mask_keys.append((sd["id"], channel_index))
                attr_rows.append({"sample_id": sd["id"], "channel_index": channel_index, **(attrs or {})})
                processed += 1
        except Exception:
            logger.exception("segmentation failed for sample %s", sd["id"])
            failed += 1

        on_progress(i + 1, total, f"Processed {i + 1}/{total}")

    with thread_db_session(db_url) as db:
        _persist_results(db, session_id, method, new_mask_keys, attr_rows)

    logger.info("Background segmentation complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": total}


def get_mask_attributes(db: DbSession, sample_id: str, channel_index: int = 0) -> dict:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    attrs = lance_mask_attrs.get_attrs(sample.session_id, sample_id, channel_index=channel_index)
    if attrs is None:
        raise HTTPException(status_code=404, detail="No mask found for this sample")
    return attrs


def get_mask_overlay_path(db: DbSession, sample_id: str, channel_index: int = 0) -> Path:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    session = db.query(Session).filter(Session.id == sample.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    path = _overlay_path(Path(session.session_folder), sample_id, channel_index)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Mask overlay file not found on disk")
    return path


def list_available_methods() -> list[dict]:
    return list_methods()
