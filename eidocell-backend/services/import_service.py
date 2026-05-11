"""Import orchestrator.

Two-phase import driven from one background task:

  Phase 1 (I/O):    walk source, group multi-channel files, parse CSV,
                    decode bytes, stage in ``import_staging_{sid}_{iid}``.
  Phase 2 (preproc): drain staging, apply pipeline, write final arrays to
                    ``images_{sid}``, create Sample SQL rows, write
                    ``preprocessing_metadata.json``.

Both phases stream through Lance with bounded memory. Cancellation is
checked at every flush point. Skipped/errored items are logged into the
``Import.errors`` JSON list rather than aborting the whole job.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.config import SUPPORTED_IMAGE_EXTENSIONS
from core.image_io import cif_rif, registry as image_registry
from core.image_io.csv_linker import lookup as csv_lookup, parse_csv as parse_csv_file
from core.image_io.errors import ContainerParseError, CsvLinkError, ImageReadError
from core.image_io.grouping import group_paths
from core.preprocessing import build_pipeline
from core.preprocessing.stats import compute_size_stats
from core.storage import images as image_store
from core.storage import import_staging
from core.storage import masks as mask_store
from core.storage import sample_attrs as attrs_store
from core.task_manager import TaskCancelledException, task_manager
from models.models import Import, Mask, Sample, SampleClass, Session
from schemas.imports import ImportCreate
from services._pipeline_utils import thread_db_session

logger = logging.getLogger("eidocell.import")

_FLUSH_EVERY = 512


# ── Public API ─────────────────────────────────────────────────────────


def run_import(db: DbSession, session_id: str, payload: ImportCreate) -> tuple[str, str]:
    """Validate inputs, persist Import row, submit background task.

    Returns ``(import_id, task_id)``.
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    src = Path(payload.source_path)
    if payload.source_kind == "folder":
        if not src.is_dir():
            raise HTTPException(status_code=400, detail=f"Folder not found: {src}")
    else:
        if not src.is_file():
            raise HTTPException(status_code=400, detail=f"File not found: {src}")

    if payload.csv_path:
        if not Path(payload.csv_path).is_file():
            raise HTTPException(status_code=400, detail=f"CSV not found: {payload.csv_path}")
        if not payload.csv_filename_col:
            raise HTTPException(status_code=400, detail="csv_filename_col required when csv_path is set")

    n_channels = payload.resolved_n_channels()
    channel_names = payload.channel_names

    # First import sets the session's channel topology. Subsequent imports must match.
    existing_sample_count = db.query(Sample).filter(Sample.session_id == session.id).count()
    if existing_sample_count == 0:
        session.channel_count = n_channels
        if channel_names is not None:
            if len(channel_names) != n_channels:
                raise HTTPException(
                    status_code=400,
                    detail=f"channel_names length ({len(channel_names)}) must equal n_channels ({n_channels})",
                )
            session.channel_names = channel_names
        elif session.channel_names is None:
            session.channel_names = [f"Channel {i + 1}" for i in range(n_channels)]
    else:
        if n_channels != session.channel_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This session was created with {session.channel_count} channel(s); "
                    f"new imports must match. Got n_channels={n_channels}."
                ),
            )

    record = Import(
        session_id=session.id,
        source_kind=payload.source_kind,
        source_path=str(src),
        csv_path=payload.csv_path,
        csv_filename_col=payload.csv_filename_col,
        channel_grouping=payload.channel_grouping,
        preprocessing_config=(payload.preprocessing.model_dump() if payload.preprocessing else None),
        status="pending",
        sample_count=0,
        skipped_count=0,
        errors=[],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    db_url = str(db.get_bind().url)
    task_id = task_manager.submit(
        name=f"Import ({payload.source_kind})",
        func=_background_import,
        import_id=record.id,
        session_id=session.id,
        session_folder=session.session_folder,
        source_kind=payload.source_kind,
        source_path=str(src),
        csv_path=payload.csv_path,
        csv_filename_col=payload.csv_filename_col,
        channel_grouping=payload.channel_grouping,
        n_channels=n_channels,
        preprocessing_config=(payload.preprocessing.model_dump() if payload.preprocessing else None),
        db_url=db_url,
    )
    record.task_id = task_id
    db.commit()
    return record.id, task_id


def list_imports(db: DbSession, session_id: str) -> list[Import]:
    return (
        db.query(Import)
        .filter(Import.session_id == session_id)
        .order_by(Import.created_at.desc())
        .all()
    )


def get_import(db: DbSession, session_id: str, import_id: str) -> Import:
    rec = (
        db.query(Import)
        .filter(Import.session_id == session_id, Import.id == import_id)
        .first()
    )
    if not rec:
        raise HTTPException(status_code=404, detail="Import not found")
    return rec


def cancel_import(db: DbSession, session_id: str, import_id: str) -> Import:
    rec = get_import(db, session_id, import_id)
    if rec.status in ("completed", "failed", "cancelled"):
        return rec
    if rec.task_id:
        task_manager.cancel_task(rec.task_id)
    return rec


# ── Worker (runs on a thread, no Request DB session in scope) ──────────


def _background_import(
    *,
    import_id: str,
    session_id: str,
    session_folder: str,
    source_kind: str,
    source_path: str,
    csv_path: str | None,
    csv_filename_col: str | None,
    channel_grouping: bool,
    n_channels: int,
    preprocessing_config: dict | None,
    db_url: str,
    on_progress,
    is_cancelled,
):
    errors: list[dict] = []
    skipped = 0
    processed = 0

    def _set_status(status: str, *, increment_skipped: int = 0, sample_count: int | None = None, persist_errors: bool = False):
        with thread_db_session(db_url) as db:
            rec = db.query(Import).filter(Import.id == import_id).first()
            if rec is None:
                return
            rec.status = status
            if increment_skipped:
                rec.skipped_count = (rec.skipped_count or 0) + increment_skipped
            if sample_count is not None:
                rec.sample_count = sample_count
            if persist_errors:
                rec.errors = list(errors)
            db.commit()

    try:
        _set_status("loading")

        # ── Optional CSV parsing ────────────────────────────────────────
        csv_index: dict[str, dict] = {}
        if csv_path and csv_filename_col:
            try:
                csv_index = parse_csv_file(csv_path, filename_col=csv_filename_col)
            except CsvLinkError as e:
                errors.append({"path": csv_path, "reason": f"CSV: {e}"})
                _set_status("failed", persist_errors=True)
                return {"processed": 0, "skipped": 0, "error": str(e)}

        # ── Phase 1: I/O staging ────────────────────────────────────────
        on_progress(0, 1, "Phase 1: scanning source")
        staged = 0
        if source_kind == "folder":
            staged, skipped, errors = _stage_folder(
                session_id=session_id,
                import_id=import_id,
                folder=Path(source_path),
                channel_grouping=channel_grouping,
                n_channels=n_channels,
                csv_index=csv_index,
                errors=errors,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
        else:  # cif / rif
            try:
                staged, skipped, errors = _stage_container(
                    session_id=session_id,
                    import_id=import_id,
                    file_path=Path(source_path),
                    n_channels=n_channels,
                    csv_index=csv_index,
                    errors=errors,
                    on_progress=on_progress,
                    is_cancelled=is_cancelled,
                )
            except ContainerParseError as e:
                errors.append({"path": source_path, "reason": f"container: {e}"})
                _set_status("failed", persist_errors=True)
                return {"processed": 0, "skipped": 0, "error": str(e)}

        if staged == 0:
            errors.append({"path": source_path, "reason": "no readable images found"})
            _set_status("failed", persist_errors=True)
            import_staging.drop(session_id, import_id)
            return {"processed": 0, "skipped": skipped}

        # Persist mid-flight skipped count + errors so the UI can preview them.
        _set_status("loading", persist_errors=True)

        # ── Phase 2: finalize (preprocess or 1:1 copy) ─────────────────
        if preprocessing_config:
            _set_status("preprocessing")
            processed = _run_preprocessing(
                session_id=session_id,
                import_id=import_id,
                session_folder=session_folder,
                preprocessing_config=preprocessing_config,
                db_url=db_url,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
        else:
            processed = _finalize_without_preprocessing(
                session_id=session_id,
                import_id=import_id,
                db_url=db_url,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )

        # Cleanup + mark done.
        import_staging.drop(session_id, import_id)
        _set_status("completed", sample_count=processed, persist_errors=True)
        on_progress(processed, processed, f"Imported {processed} samples")

        # Kick off the eager preview pre-generation job. The session-open UI
        # blocks on this completing.
        try:
            from services.session_service import submit_pregenerate_previews
            previews_task_id = submit_pregenerate_previews(session_id=session_id, db_url=db_url)
            with thread_db_session(db_url) as db:
                rec = db.query(Import).filter(Import.id == import_id).first()
                if rec is not None:
                    rec.previews_task_id = previews_task_id
                    db.commit()
        except Exception:
            logger.exception("failed to submit preview pregen task for import %s", import_id)

        return {"processed": processed, "skipped": skipped}

    except TaskCancelledException:
        import_staging.drop(session_id, import_id)
        _set_status("cancelled", persist_errors=True)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("import %s failed", import_id)
        errors.append({"path": source_path, "reason": str(e)})
        import_staging.drop(session_id, import_id)
        _set_status("failed", persist_errors=True)
        raise


# ── Phase 1 helpers ────────────────────────────────────────────────────


def _stage_folder(
    *,
    session_id: str,
    import_id: str,
    folder: Path,
    channel_grouping: bool,
    n_channels: int,
    csv_index: dict[str, dict],
    errors: list[dict],
    on_progress,
    is_cancelled,
) -> tuple[int, int, list[dict]]:
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    groups = group_paths(files, detect_channels=channel_grouping)
    total = len(groups)
    if total == 0:
        return 0, 0, errors

    skipped = 0
    staged = 0
    buffer: list[dict] = []
    for i, g in enumerate(groups):
        if is_cancelled and is_cancelled():
            raise TaskCancelledException("Import cancelled")

        try:
            if n_channels > 1 and len(g["paths"]) != n_channels:
                raise ImageReadError(
                    f"expected {n_channels} channels for group '{g['group_stem']}', got {len(g['paths'])}"
                )
            arrays: list[np.ndarray] = []
            for p in g["paths"]:
                img, _meta = image_registry.read(p)
                if img is None:
                    raise ImageReadError(f"reader returned None for {p.name}")
                arrays.append(_normalise_2d(img))
            stacked = _stack_channels(arrays)
            attrs = csv_lookup(csv_index, g["paths"][0].name) if csv_index else {}
            buffer.append({
                "tmp_id": uuid.uuid4().hex[:12],
                "group_stem": g["group_stem"],
                "source_paths": [str(p) for p in g["paths"]],
                "channel_meta": {
                    "order": [str(idx) for idx in g["channel_indices"]],
                    "names": [p.name for p in g["paths"]],
                },
                "array": stacked,
                "parsed_attrs": attrs,
            })
            staged += 1
        except Exception as e:  # noqa: BLE001
            skipped += 1
            errors.append({"path": str(g["paths"][0]), "reason": str(e)})

        if len(buffer) >= _FLUSH_EVERY:
            import_staging.upsert_rows(session_id, import_id, buffer)
            buffer = []
        on_progress(i + 1, total, f"Phase 1: staged {i + 1}/{total}")

    if buffer:
        import_staging.upsert_rows(session_id, import_id, buffer)
    return staged, skipped, errors


def _stage_container(
    *,
    session_id: str,
    import_id: str,
    file_path: Path,
    n_channels: int,
    csv_index: dict[str, dict],
    errors: list[dict],
    on_progress,
    is_cancelled,
) -> tuple[int, int, list[dict]]:
    total = cif_rif.count_objects(file_path)
    if total == 0:
        return 0, 0, errors

    skipped = 0
    staged = 0
    buffer: list[dict] = []
    iterator = cif_rif.iter_objects(file_path, n_channels=n_channels)
    for i, rec in enumerate(iterator):
        if is_cancelled and is_cancelled():
            raise TaskCancelledException("Import cancelled")
        try:
            channels = rec["channels"]
            stacked = _stack_channels([_normalise_2d(c) for c in channels])
            # The IDEAS mask is stored as a single 2-D plane that spans the same
            # crop window as the multi-channel image. Channels share the same
            # cell silhouette, so we collapse the per-channel mask tiles back to
            # one HW plane (the union across channels) before persisting.
            ideas_mask = _flatten_mask(rec.get("mask_channels"), rec.get("mask"))
            stem = f"{file_path.stem}_obj{rec['object_index']:06d}"
            attrs = csv_lookup(csv_index, stem) if csv_index else {}
            buffer.append({
                "tmp_id": uuid.uuid4().hex[:12],
                "group_stem": stem,
                "source_paths": [f"{file_path.name}#obj={rec['object_index']}"],
                "channel_meta": {
                    "order": [str(c) for c in range(len(channels))],
                    "names": [f"ch{c+1}" for c in range(len(channels))],
                    "compression": rec["compression"],
                    "object_index": rec["object_index"],
                },
                "array": stacked,
                "mask": ideas_mask,
                "parsed_attrs": attrs,
            })
            staged += 1
        except Exception as e:  # noqa: BLE001
            skipped += 1
            errors.append({"path": f"{file_path.name}#obj={rec['object_index']}", "reason": str(e)})

        if len(buffer) >= _FLUSH_EVERY:
            import_staging.upsert_rows(session_id, import_id, buffer)
            buffer = []
        on_progress(i + 1, total, f"Phase 1: decoded {i + 1}/{total}")

    if buffer:
        import_staging.upsert_rows(session_id, import_id, buffer)
    return staged, skipped, errors


def _normalise_2d(img: np.ndarray) -> np.ndarray:
    """Drop alpha channel if present and squeeze trivial dims."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return img


def _flatten_mask(mask_channels: list[np.ndarray] | None, full_mask: np.ndarray | None) -> np.ndarray | None:
    """Collapse a multi-channel IDEAS mask to one HW plane (binary union)."""
    if mask_channels:
        if len(mask_channels) == 1:
            return (mask_channels[0] != 0).astype(np.uint8)
        h, w = mask_channels[0].shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        for m in mask_channels:
            out |= (m != 0).astype(np.uint8)
        return out
    if full_mask is not None:
        return (full_mask != 0).astype(np.uint8)
    return None


def _transform_mask_to_match(
    mask: np.ndarray,
    target_shape: tuple[int, int] | None,
    *,
    resize: bool,
) -> np.ndarray:
    """Apply spatial-only transforms so a mask lines up with its preprocessed
    image. Uses nearest-neighbour resize and constant-zero padding to preserve
    binary values; normalization/compensation/alignment are not applied.
    """
    if target_shape is None:
        return mask
    import cv2

    th, tw = target_shape
    h, w = mask.shape[:2]
    if (h, w) == (th, tw):
        return mask
    if resize:
        return cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    # Pad to target (centred, zero fill)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    top = pad_h // 2
    left = pad_w // 2
    out = np.zeros((th, tw), dtype=mask.dtype)
    out[top : top + h, left : left + w] = mask
    return out


def _stack_channels(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack a list of channel arrays into HWC (or HW for singletons).

    All inputs must share a common HW; this raises ImageReadError if shapes
    disagree, which the caller treats as a per-sample skip.
    """
    if len(arrays) == 1:
        return arrays[0]
    h, w = arrays[0].shape[:2]
    for a in arrays[1:]:
        if a.shape[:2] != (h, w):
            raise ImageReadError(
                f"channel shape mismatch: expected {(h, w)}, got {a.shape[:2]}"
            )
    flat = []
    for a in arrays:
        if a.ndim == 2:
            flat.append(a)
        elif a.ndim == 3:
            for c in range(a.shape[2]):
                flat.append(a[:, :, c])
        else:
            raise ImageReadError(f"unsupported ndim {a.ndim}")
    return np.stack(flat, axis=-1)


# ── Phase 2 helpers ────────────────────────────────────────────────────


def _finalize_without_preprocessing(
    *,
    session_id: str,
    import_id: str,
    db_url: str,
    on_progress,
    is_cancelled,
) -> int:
    """Copy staging → images table 1:1 and create Sample rows."""
    total = import_staging.count(session_id, import_id)
    if total == 0:
        return 0

    with thread_db_session(db_url) as db:
        uncategorized_id = _ensure_uncategorized(db, session_id)

    image_buffer: list[dict] = []
    sample_buffer: list[Sample] = []
    attrs_buffer: list[dict] = []
    mask_buffer: list[dict] = []
    processed = 0

    for i, row in enumerate(import_staging.iter_rows(session_id, import_id)):
        if is_cancelled and is_cancelled():
            raise TaskCancelledException("Import cancelled")
        sample_id = uuid.uuid4().hex[:12]
        arr = row["array"]
        h, w, c = arr.shape if arr.ndim == 3 else (*arr.shape, 1)
        meta = dict(row["channel_meta"] or {})
        meta.update({
            "shape": [int(h), int(w), int(c)],
            "raw_paths": row["source_paths"],
            "import_id": import_id,
            "preprocessing": None,
        })
        image_buffer.append({"sample_id": sample_id, "array": arr, "source": "raw"})
        sample_buffer.append(Sample(
            id=sample_id,
            session_id=session_id,
            filename=row["source_paths"][0] if row["source_paths"] else row["group_stem"],
            path=row["source_paths"][0] if row["source_paths"] else "",
            class_id=uncategorized_id,
            channels=meta,
        ))
        if row["parsed_attrs"]:
            attrs_buffer.append({"sample_id": sample_id, **row["parsed_attrs"]})
        if row.get("mask") is not None:
            mask_buffer.append({"sample_id": sample_id, "mask": row["mask"]})
        processed += 1

        if len(image_buffer) >= _FLUSH_EVERY:
            _flush_finalize(session_id, db_url, image_buffer, sample_buffer, attrs_buffer, mask_buffer)
            image_buffer.clear()
            sample_buffer.clear()
            attrs_buffer.clear()
            mask_buffer.clear()
        on_progress(i + 1, total, f"Phase 2: finalising {i + 1}/{total}")

    if image_buffer:
        _flush_finalize(session_id, db_url, image_buffer, sample_buffer, attrs_buffer, mask_buffer)
    return processed


def _run_preprocessing(
    *,
    session_id: str,
    import_id: str,
    session_folder: str,
    preprocessing_config: dict,
    db_url: str,
    on_progress,
    is_cancelled,
) -> int:
    summary = compute_size_stats(session_id, import_id)
    pipeline = build_pipeline(preprocessing_config, shape_summary=summary)

    total = summary.get("count", 0)
    if total == 0:
        return 0

    with thread_db_session(db_url) as db:
        uncategorized_id = _ensure_uncategorized(db, session_id)

    image_buffer: list[dict] = []
    sample_buffer: list[Sample] = []
    attrs_buffer: list[dict] = []
    mask_buffer: list[dict] = []
    processed = 0
    pipeline_log: list[dict] = []

    mask_resize = bool(preprocessing_config.get("resize")) if preprocessing_config else False

    for i, row in enumerate(import_staging.iter_rows(session_id, import_id)):
        if is_cancelled and is_cancelled():
            raise TaskCancelledException("Import cancelled")
        sample_id = uuid.uuid4().hex[:12]
        try:
            out, step_meta = pipeline.apply(row["array"])
        except Exception as e:  # noqa: BLE001
            logger.warning("sample %s preprocessing failed: %s", row["tmp_id"], e)
            continue
        h, w, c = out.shape if out.ndim == 3 else (*out.shape, 1)
        meta = dict(row["channel_meta"] or {})
        meta.update({
            "shape": [int(h), int(w), int(c)],
            "raw_paths": row["source_paths"],
            "import_id": import_id,
            "preprocessing": {
                "pipeline_id": pipeline.pipeline_id,
                "config": preprocessing_config,
            },
        })
        image_buffer.append({"sample_id": sample_id, "array": out, "source": "preprocessed"})
        sample_buffer.append(Sample(
            id=sample_id,
            session_id=session_id,
            filename=row["source_paths"][0] if row["source_paths"] else row["group_stem"],
            path=row["source_paths"][0] if row["source_paths"] else "",
            class_id=uncategorized_id,
            channels=meta,
        ))
        if row["parsed_attrs"]:
            attrs_buffer.append({"sample_id": sample_id, **row["parsed_attrs"]})
        if row.get("mask") is not None:
            mask_aligned = _transform_mask_to_match(
                row["mask"], pipeline.target_shape, resize=mask_resize
            )
            mask_buffer.append({"sample_id": sample_id, "mask": mask_aligned})
        pipeline_log.append({"sample_id": sample_id, **step_meta})
        processed += 1

        if len(image_buffer) >= _FLUSH_EVERY:
            _flush_finalize(session_id, db_url, image_buffer, sample_buffer, attrs_buffer, mask_buffer)
            image_buffer.clear()
            sample_buffer.clear()
            attrs_buffer.clear()
            mask_buffer.clear()
        on_progress(i + 1, total, f"Phase 2: preprocessing {i + 1}/{total}")

    if image_buffer:
        _flush_finalize(session_id, db_url, image_buffer, sample_buffer, attrs_buffer, mask_buffer)

    _write_metadata_doc(
        session_folder,
        import_id=import_id,
        pipeline_id=pipeline.pipeline_id,
        config=preprocessing_config,
        shape_summary=summary,
        target_shape=list(pipeline.target_shape) if pipeline.target_shape else None,
        steps=[s.name for s in pipeline.steps],
        per_sample=pipeline_log,
    )
    return processed


def _flush_finalize(
    session_id: str,
    db_url: str,
    image_buffer: list[dict],
    sample_buffer: list[Sample],
    attrs_buffer: list[dict],
    mask_buffer: list[dict] | None = None,
) -> None:
    if image_buffer:
        image_store.upsert_arrays(session_id, image_buffer)
    if attrs_buffer:
        attrs_store.upsert_attrs(session_id, attrs_buffer)
    if sample_buffer:
        with thread_db_session(db_url) as db:
            for s in sample_buffer:
                db.add(s)
            db.commit()

        # Pre-generate thumbnails
        session_folder = ""
        with thread_db_session(db_url) as db:
            sess = db.query(Session).filter(Session.id == session_id).first()
            if sess:
                session_folder = sess.session_folder

        if session_folder:
            from core.processors.image_utils import render_array_to_thumbnail
            thumb_dir = Path(session_folder) / "previews" / "thumbnails"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            for item in image_buffer:
                arr = item["array"]
                sample_id = item["sample_id"]
                thumb_path = thumb_dir / f"{sample_id}.jpg"
                render_array_to_thumbnail(arr, thumb_path)

            if mask_buffer:
                with thread_db_session(db_url) as db:
                    for item in mask_buffer:
                        mask_store.write_mask(
                            session_folder, item["sample_id"], item["mask"], channel_index=0,
                        )
                        db.add(Mask(
                            sample_id=item["sample_id"],
                            channel_index=0,
                            segmentation_method="ideas_import",
                        ))
                    db.commit()

def preview_preprocessing(db: DbSession, session_id: str, payload: "PreviewRequest") -> str:
    from core.preprocessing import build_pipeline
    import base64
    import cv2

    src = Path(payload.source_path)
    if not src.exists():
        raise Exception("Source path does not exist")
        
    img = None
    if payload.source_kind == "folder":
        if src.is_file():
            files = [src]
        else:
            files = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        if not files:
            raise Exception("No valid images found")
        img_arr, _ = image_registry.read(files[0])
        img = _normalise_2d(img_arr)
    else:
        # CIF/RIF
        iterator = cif_rif.iter_objects(src, n_channels=payload.resolved_n_channels())
        first_obj = next(iterator, None)
        if not first_obj:
            raise Exception("No objects found in container")
        img = _stack_channels([_normalise_2d(c) for c in first_obj["channels"]])

    pipeline = build_pipeline(payload.preprocessing.model_dump(), shape_summary=None)
    out, _ = pipeline.apply(img)
    
    # Render to base64
    # We can reuse the rendering logic from processor
    # But for a quick preview, let's just use PIL or cv2 directly
    
    # We have to handle multi-channel properly. The easiest is to use the existing `render_array_to_full` to a temp file, but that's messy.
    # Let's normalize it to 8-bit for preview
    if out.dtype != np.uint8:
        out_min = out.min()
        out_max = out.max()
        if out_max > out_min:
            out = ((out - out_min) / (out_max - out_min) * 255).astype(np.uint8)
        else:
            out = np.zeros_like(out, dtype=np.uint8)
            
    # For multi-channel, show a simple horizontal stack or let frontend handle it
    # We will just horizontally concatenate channels if > 3
    h, w = out.shape[:2]
    c = out.shape[2] if out.ndim == 3 else 1
    
    if c == 1:
        display = out
    elif c == 3:
        display = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    else:
        channels = [out[:, :, i] for i in range(c)]
        display = np.hstack(channels)
        
    _, buffer = cv2.imencode('.png', display)
    return base64.b64encode(buffer).decode('utf-8')

def _ensure_uncategorized(db: DbSession, session_id: str) -> str:
    cls = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == session_id, SampleClass.name == "Uncategorized")
        .first()
    )
    if cls is None:
        cls = SampleClass(session_id=session_id, name="Uncategorized", color="#808080")
        db.add(cls)
        db.commit()
        db.refresh(cls)
    return cls.id


def _write_metadata_doc(
    session_folder: str,
    *,
    import_id: str,
    pipeline_id: str,
    config: dict,
    shape_summary: dict,
    target_shape: list[int] | None,
    steps: list[str],
    per_sample: list[dict],
) -> None:
    folder = Path(session_folder) / "imports"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{import_id}.json"
    payload = {
        "import_id": import_id,
        "pipeline_id": pipeline_id,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "shape_summary": shape_summary,
        "target_shape": target_shape,
        "steps": steps,
        "per_sample_count": len(per_sample),
        # Per-sample log can be huge; persist counts only by default.
        # Set EIDOCELL_IMPORT_FULL_LOG to embed every sample's step trace.
        "per_sample": per_sample if _full_log() else [],
    }
    out.write_text(json.dumps(payload, indent=2))


def _full_log() -> bool:
    import os
    return os.environ.get("EIDOCELL_IMPORT_FULL_LOG") == "1"
