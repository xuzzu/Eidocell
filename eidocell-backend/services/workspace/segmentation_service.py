"""Segmentation service: run segmentation, store masks, compute attributes."""

import logging
from pathlib import Path

import cv2
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.image_io import read as read_image
from core.processors.image_utils import generate_mask_overlay
from core.processors.inference.mask_attributes import compute_mask_attributes
from core.processors.inference.segmentation import get_processor, list_methods
from core.storage import mask_attrs as lance_mask_attrs
from core.storage import masks as mask_files
from core.task_manager import task_manager
from core.utils import get_active_samples
from models.models import Mask, Sample, Session
from services._pipeline_utils import thread_db_session

logger = logging.getLogger("eidocell.segmentation")


_OVERLAYS_SUBDIR = "previews/overlays"


def _overlay_path(session_folder: Path, sample_id: str) -> Path:
    return session_folder / _OVERLAYS_SUBDIR / f"{sample_id}.png"


def _segment_one(
    src_path: Path,
    processor,
    params: dict,
    scale_factor: float,
    session_folder: Path,
    sample_id: str,
):
    """Read image, segment, write mask + overlay to disk, return (attrs, ok).

    Returns (attrs_or_None, True) on success, (None, False) on failure.
    """
    image, _channels = read_image(src_path)
    if image is None:
        return None, False
    mask_data = processor.segment(image, **params)
    attrs = compute_mask_attributes(image, mask_data, scale_factor)

    mask_files.write_mask(session_folder, sample_id, mask_data)

    overlay = generate_mask_overlay(image, mask_data)
    overlay_path = _overlay_path(session_folder, sample_id)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return attrs, True


def _persist_results(
    db: DbSession,
    session_id: str,
    method: str,
    new_mask_sample_ids: list[str],
    attr_rows: list[dict],
) -> None:
    """Insert Mask rows for new samples and upsert Lance attribute rows."""
    if new_mask_sample_ids:
        existing_ids = {
            sid for (sid,) in db.query(Mask.sample_id)
            .filter(Mask.sample_id.in_(new_mask_sample_ids)).all()
        }
        for sid in new_mask_sample_ids:
            if sid in existing_ids:
                db.query(Mask).filter(Mask.sample_id == sid).update(
                    {"segmentation_method": method}
                )
            else:
                db.add(Mask(sample_id=sid, segmentation_method=method))
        db.commit()

    if attr_rows:
        lance_mask_attrs.upsert_attrs(session_id, attr_rows)


def run_segmentation(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
) -> dict:
    """Run segmentation on all active samples in the session."""
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

    session_folder = Path(session.session_folder)
    sample_data = [{"id": s.id, "path": s.path} for s in samples]
    return _run_for_samples(
        db, session, processor, method, params, sample_data, session_folder
    )


def run_segmentation_preview(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
    sample_ids: list[str],
) -> dict:
    """Run segmentation on specific samples (committed to disk + DB)."""
    try:
        processor = get_processor(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    samples = db.query(Sample).filter(
        Sample.id.in_(sample_ids),
        Sample.session_id == session_id,
    ).all()
    if not samples:
        raise HTTPException(status_code=400, detail="No matching samples found")

    session_folder = Path(session.session_folder)
    sample_data = [{"id": s.id, "path": s.path} for s in samples]
    return _run_for_samples(
        db, session, processor, method, params, sample_data, session_folder
    )


def _run_for_samples(
    db: DbSession,
    session: Session,
    processor,
    method: str,
    params: dict,
    sample_data: list[dict],
    session_folder: Path,
) -> dict:
    scale_factor = session.scale_factor
    processed = 0
    failed = 0
    attr_rows: list[dict] = []
    new_mask_sample_ids: list[str] = []

    for sd in sample_data:
        src_path = Path(sd["path"])
        if not src_path.is_file():
            failed += 1
            continue
        try:
            attrs, ok = _segment_one(
                src_path, processor, params, scale_factor, session_folder, sd["id"]
            )
            if not ok:
                failed += 1
                continue
            new_mask_sample_ids.append(sd["id"])
            attr_rows.append({"sample_id": sd["id"], **(attrs or {})})
            processed += 1
        except Exception:
            logger.exception("segmentation failed for sample %s", sd["id"])
            failed += 1

    _persist_results(db, session.id, method, new_mask_sample_ids, attr_rows)
    logger.info("Segmentation complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": len(sample_data)}


def stream_preview_overlays(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
    sample_ids: list[str],
):
    """Yield (sample_id, overlay_png_bytes, attributes) per sample.

    Used by the WebSocket preview endpoint. Does NOT write masks/overlays
    to disk or DB — preview is throwaway.
    """
    processor = get_processor(method)

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    samples = db.query(Sample).filter(
        Sample.id.in_(sample_ids),
        Sample.session_id == session_id,
    ).all()
    samples_by_id = {s.id: s for s in samples}
    ordered = [samples_by_id[sid] for sid in sample_ids if sid in samples_by_id]

    scale_factor = session.scale_factor

    for s in ordered:
        src_path = Path(s.path)
        if not src_path.is_file():
            yield s.id, None, None
            continue
        try:
            image, _ = read_image(src_path)
            if image is None:
                yield s.id, None, None
                continue
            mask_data = processor.segment(image, **params)
            attrs = compute_mask_attributes(image, mask_data, scale_factor)
            overlay = generate_mask_overlay(image, mask_data)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                yield s.id, None, None
                continue
            yield s.id, bytes(buf), attrs
        except Exception:
            logger.exception("preview failed for sample %s", s.id)
            yield s.id, None, None


def run_segmentation_async(
    db: DbSession, session_id: str, method: str, params: dict
) -> str:
    """Start segmentation as a background task. Returns task ID."""
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
):
    """Run segmentation in a background thread."""
    processor = get_processor(method)
    session_folder = Path(session_folder)

    total = len(sample_data)
    processed = 0
    failed = 0
    attr_rows: list[dict] = []
    new_mask_sample_ids: list[str] = []

    on_progress(0, total, "Starting segmentation...")

    for i, sd in enumerate(sample_data):
        if is_cancelled and is_cancelled():
            pass  # TaskCancelledException raised by on_progress

        src_path = Path(sd["path"])
        if not src_path.is_file():
            failed += 1
            on_progress(i + 1, total, f"Processed {i + 1}/{total}")
            continue

        try:
            attrs, ok = _segment_one(
                src_path, processor, params, scale_factor, session_folder, sd["id"]
            )
            if not ok:
                failed += 1
            else:
                new_mask_sample_ids.append(sd["id"])
                attr_rows.append({"sample_id": sd["id"], **(attrs or {})})
                processed += 1
        except Exception:
            logger.exception("segmentation failed for sample %s", sd["id"])
            failed += 1

        on_progress(i + 1, total, f"Processed {i + 1}/{total}")

    with thread_db_session(db_url) as db:
        _persist_results(db, session_id, method, new_mask_sample_ids, attr_rows)

    logger.info("Background segmentation complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": total}


def get_mask_attributes(db: DbSession, sample_id: str) -> dict:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    attrs = lance_mask_attrs.get_attrs(sample.session_id, sample_id)
    if attrs is None:
        raise HTTPException(status_code=404, detail="No mask found for this sample")
    return attrs


def get_mask_overlay_path(db: DbSession, sample_id: str) -> Path:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    session = db.query(Session).filter(Session.id == sample.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    path = _overlay_path(Path(session.session_folder), sample_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Mask overlay file not found on disk")
    return path


def list_available_methods() -> list[dict]:
    return list_methods()
