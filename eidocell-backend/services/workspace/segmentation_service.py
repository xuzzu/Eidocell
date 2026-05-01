"""Segmentation service: run segmentation, store masks, compute attributes."""

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DbSession, sessionmaker

from core.processors.inference.segmentation import get_processor, list_methods
from core.processors.inference.mask_attributes import compute_mask_attributes
from core.processors.image_utils import generate_mask_overlay
from core.task_manager import task_manager
from core.utils import get_active_samples, npy_save, npy_load
from models.models import Session, Sample, Mask

logger = logging.getLogger("eidocell.segmentation")


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
    scale_factor = session.scale_factor
    overlay_dir = session_folder / "masked_images"
    overlay_dir.mkdir(exist_ok=True)

    # Snapshot sample data to avoid lazy-load issues during processing
    sample_data = [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]

    # Check which samples already have masks
    existing_mask_ids = {
        m.sample_id
        for m in db.query(Mask.sample_id).filter(
            Mask.sample_id.in_([s["id"] for s in sample_data])
        ).all()
    }

    # Build/load consolidated masks array
    masks_path = session_folder / "features" / "session_masks.npy"
    max_index = max(s["storage_index"] for s in sample_data) + 1
    if masks_path.exists():
        all_masks = npy_load(masks_path, allow_pickle=True)
        if len(all_masks) < max_index:
            new_arr = np.empty(max_index, dtype=object)
            new_arr[:len(all_masks)] = all_masks
            all_masks = new_arr
    else:
        all_masks = np.empty(max_index, dtype=object)

    processed = 0
    failed = 0
    mask_updates = []
    mask_creates = []

    for sd in sample_data:
        src_path = Path(sd["path"])
        if not src_path.is_file():
            failed += 1
            continue

        try:
            image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                failed += 1
                continue

            mask_data = processor.segment(image, **params)
            all_masks[sd["storage_index"]] = mask_data
            attrs = compute_mask_attributes(image, mask_data, scale_factor)

            overlay = generate_mask_overlay(image, mask_data)
            overlay_path = overlay_dir / f"{sd['id']}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if sd["id"] in existing_mask_ids:
                mask_updates.append((sd["id"], str(overlay_path), attrs))
            else:
                mask_creates.append((sd["id"], str(overlay_path), attrs))

            processed += 1
        except Exception:
            failed += 1
            continue

    # Batch DB writes
    for sample_id, overlay_path, attrs in mask_updates:
        db.query(Mask).filter(Mask.sample_id == sample_id).update(
            {"masked_image_path": overlay_path, "attributes": attrs}
        )
    for sample_id, overlay_path, attrs in mask_creates:
        db.add(Mask(sample_id=sample_id, masked_image_path=overlay_path, attributes=attrs))

    (session_folder / "features").mkdir(exist_ok=True)
    npy_save(masks_path, all_masks, allow_pickle=True)

    db.commit()
    logger.info("Segmentation complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": len(sample_data)}


def run_segmentation_preview(
    db: DbSession,
    session_id: str,
    method: str,
    params: dict,
    sample_ids: list[str],
) -> dict:
    """Run segmentation on specific samples (for live preview)."""
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
    scale_factor = session.scale_factor
    overlay_dir = session_folder / "masked_images"
    overlay_dir.mkdir(exist_ok=True)

    sample_data = [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]

    existing_mask_ids = {
        m.sample_id
        for m in db.query(Mask.sample_id).filter(
            Mask.sample_id.in_([s["id"] for s in sample_data])
        ).all()
    }

    # Build/load consolidated masks array
    masks_path = session_folder / "features" / "session_masks.npy"
    max_index = max(s["storage_index"] for s in sample_data) + 1
    if masks_path.exists():
        all_masks = npy_load(masks_path, allow_pickle=True)
        if len(all_masks) < max_index:
            new_arr = np.empty(max_index, dtype=object)
            new_arr[:len(all_masks)] = all_masks
            all_masks = new_arr
    else:
        all_masks = np.empty(max_index, dtype=object)

    processed = 0
    failed = 0
    mask_updates = []
    mask_creates = []

    for sd in sample_data:
        src_path = Path(sd["path"])
        if not src_path.is_file():
            failed += 1
            continue

        try:
            image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                failed += 1
                continue

            mask_data = processor.segment(image, **params)
            all_masks[sd["storage_index"]] = mask_data
            attrs = compute_mask_attributes(image, mask_data, scale_factor)

            overlay = generate_mask_overlay(image, mask_data)
            overlay_path = overlay_dir / f"{sd['id']}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if sd["id"] in existing_mask_ids:
                mask_updates.append((sd["id"], str(overlay_path), attrs))
            else:
                mask_creates.append((sd["id"], str(overlay_path), attrs))

            processed += 1
        except Exception:
            failed += 1
            continue

    # Batch DB writes
    for sample_id, overlay_path, attrs in mask_updates:
        db.query(Mask).filter(Mask.sample_id == sample_id).update(
            {"masked_image_path": overlay_path, "attributes": attrs}
        )
    for sample_id, overlay_path, attrs in mask_creates:
        db.add(Mask(sample_id=sample_id, masked_image_path=overlay_path, attributes=attrs))

    (session_folder / "features").mkdir(exist_ok=True)
    npy_save(masks_path, all_masks, allow_pickle=True)

    db.commit()
    logger.info("Segmentation preview complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": len(sample_data)}


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

    sample_data = [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]
    existing_mask_ids = {
        m.sample_id
        for m in db.query(Mask.sample_id).filter(
            Mask.sample_id.in_([s["id"] for s in sample_data])
        ).all()
    }

    db_url = str(db.get_bind().url)

    task_id = task_manager.submit(
        name=f"Segmentation ({method})",
        func=_background_segment,
        method=method,
        params=params,
        sample_data=sample_data,
        existing_mask_ids=existing_mask_ids,
        session_folder=session.session_folder,
        scale_factor=session.scale_factor,
        db_url=db_url,
    )
    return task_id


def _background_segment(
    *,
    method: str,
    params: dict,
    sample_data: list[dict],
    existing_mask_ids: set,
    session_folder: str,
    scale_factor: float,
    db_url: str,
    on_progress,
    is_cancelled=None,
):
    """Run segmentation in a background thread."""
    processor = get_processor(method)
    session_folder = Path(session_folder)
    overlay_dir = session_folder / "masked_images"
    overlay_dir.mkdir(exist_ok=True)

    masks_path = session_folder / "features" / "session_masks.npy"
    max_index = max(s["storage_index"] for s in sample_data) + 1
    if masks_path.exists():
        all_masks = npy_load(masks_path, allow_pickle=True)
        if len(all_masks) < max_index:
            new_arr = np.empty(max_index, dtype=object)
            new_arr[:len(all_masks)] = all_masks
            all_masks = new_arr
    else:
        all_masks = np.empty(max_index, dtype=object)

    total = len(sample_data)
    processed = 0
    failed = 0
    mask_updates = []
    mask_creates = []

    on_progress(0, total, "Starting segmentation...")

    for i, sd in enumerate(sample_data):
        if is_cancelled and is_cancelled():
            pass # TaskCancelledException will be raised by on_progress

        src_path = Path(sd["path"])
        if not src_path.is_file():
            failed += 1
            on_progress(i + 1, total, f"Processed {i + 1}/{total}")
            continue

        try:
            image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                failed += 1
                on_progress(i + 1, total, f"Processed {i + 1}/{total}")
                continue

            mask_data = processor.segment(image, **params)
            all_masks[sd["storage_index"]] = mask_data
            attrs = compute_mask_attributes(image, mask_data, scale_factor)

            overlay = generate_mask_overlay(image, mask_data)
            overlay_path = overlay_dir / f"{sd['id']}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if sd["id"] in existing_mask_ids:
                mask_updates.append((sd["id"], str(overlay_path), attrs))
            else:
                mask_creates.append((sd["id"], str(overlay_path), attrs))

            processed += 1
        except Exception:
            failed += 1

        on_progress(i + 1, total, f"Processed {i + 1}/{total}")

    # Write DB in background thread using its own session
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    DBSession = sessionmaker(bind=engine)
    db = DBSession()
    try:
        for sample_id, overlay_path, attrs in mask_updates:
            db.query(Mask).filter(Mask.sample_id == sample_id).update(
                {"masked_image_path": overlay_path, "attributes": attrs}
            )
        for sample_id, overlay_path, attrs in mask_creates:
            db.add(Mask(sample_id=sample_id, masked_image_path=overlay_path, attributes=attrs))

        (session_folder / "features").mkdir(exist_ok=True)
        npy_save(masks_path, all_masks, allow_pickle=True)
        db.commit()
    finally:
        db.close()
        engine.dispose()

    logger.info("Background segmentation complete: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "total": total}


def get_mask_attributes(db: DbSession, sample_id: str) -> dict:
    mask = db.query(Mask).filter(Mask.sample_id == sample_id).first()
    if not mask:
        raise HTTPException(status_code=404, detail="No mask found for this sample")
    return mask.attributes or {}


def get_mask_overlay_path(db: DbSession, sample_id: str) -> Path:
    mask = db.query(Mask).filter(Mask.sample_id == sample_id).first()
    if not mask or not mask.masked_image_path:
        raise HTTPException(status_code=404, detail="No mask overlay found for this sample")
    path = Path(mask.masked_image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Mask overlay file not found on disk")
    return path


def list_available_methods() -> list[dict]:
    return list_methods()
