import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession
from sqlalchemy import func

from core.config import SESSIONS_DIR
from core.processors.image_utils import (
    render_array_to_thumbnail,
    render_channel_to_thumbnail,
)
from core.storage import images as lance_images
from core.storage import lance as lance_store
from core.storage import mask_attrs
from core.task_manager import TaskCancelledException, task_manager
from models.models import Import, Session, Sample, SampleClass
from schemas.sessions import SessionCreate, SessionUpdate
from services._pipeline_utils import thread_db_session

logger = logging.getLogger("eidocell.session")


# Subdirectories created under each session folder.
_SESSION_SUBDIRS = (
    "masks",
    "previews",
    "previews/thumbnails",
    "previews/overlays",
    "previews/collages",
    "previews/full",  # full-resolution PNGs rendered from Lance arrays
    "imports",  # preprocessing_metadata JSON files land here
)


def _sanitize_folder_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


def list_sessions(db: DbSession) -> list[dict]:
    sessions = db.query(Session).order_by(Session.last_opened_at.desc()).all()
    results = []
    for s in sessions:
        count = db.query(func.count(Sample.id)).filter(Sample.session_id == s.id).scalar()
        results.append({
            "id": s.id,
            "name": s.name,
            "images_directory": s.images_directory,
            "created_at": s.created_at,
            "last_opened_at": s.last_opened_at,
            "sample_count": count,
        })
    return results


def create_session(db: DbSession, data: SessionCreate) -> Session:
    """Create an empty session. Samples are added later via the import flow."""
    session = Session(
        name=data.name,
        images_directory=None,
        session_folder="",  # will be set after we have the id
    )
    db.add(session)
    db.flush()  # generate id

    # Create session folder + standard subdirs
    folder_name = f"{_sanitize_folder_name(data.name)}_{session.id}"
    session_folder = SESSIONS_DIR / folder_name
    session_folder.mkdir(parents=True, exist_ok=True)
    for sub in _SESSION_SUBDIRS:
        (session_folder / sub).mkdir(parents=True, exist_ok=True)
    session.session_folder = str(session_folder)

    # Pre-create the per-session mask_attrs Lance table (fixed schema).
    # The features and images tables are created lazily on first write.
    mask_attrs._ensure_table(session.id)  # noqa: SLF001 - intentional internal use

    # Create default "Uncategorized" class
    uncategorized = SampleClass(
        session_id=session.id,
        name="Uncategorized",
        color="#808080",
    )
    db.add(uncategorized)

    db.commit()
    db.refresh(session)
    return session


def get_session(db: DbSession, session_id: str) -> Session:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update last opened timestamp
    session.last_opened_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session


def update_session(db: DbSession, session_id: str, data: SessionUpdate) -> Session:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if data.name is not None:
        session.name = data.name
    if data.scale_factor is not None:
        session.scale_factor = data.scale_factor
    if data.scale_units is not None:
        session.scale_units = data.scale_units
    if data.channel_names is not None:
        if len(data.channel_names) != session.channel_count:
            raise HTTPException(
                status_code=400,
                detail=f"channel_names length ({len(data.channel_names)}) must match channel_count ({session.channel_count})",
            )
        session.channel_names = data.channel_names

    db.commit()
    db.refresh(session)
    return session


def delete_session(db: DbSession, session_id: str) -> None:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Drop Lance tables before the session row goes away so we can still
    # reference session_id for the table names.
    lance_store.drop_session_tables(session_id)

    # Remove session folder from disk (masks + previews live here)
    session_folder = Path(session.session_folder)
    if session_folder.exists():
        shutil.rmtree(session_folder)

    db.delete(session)
    db.commit()


def get_session_sample_count(db: DbSession, session_id: str) -> int:
    return db.query(func.count(Sample.id)).filter(Sample.session_id == session_id).scalar()


# ── Preview pregeneration ────────────────────────────────────────────────


_THUMB_BATCH = 32


def pregenerate_previews(
    *,
    session_id: str,
    db_url: str,
    on_progress,
    is_cancelled=None,
) -> dict:
    """Eagerly render per-channel + combined thumbnails for all samples in a session.

    Idempotent: skips files that already exist. Designed as a TaskManager target.
    Blocks the session-open UI via the /preview-status endpoint until done.
    """
    with thread_db_session(db_url) as db:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session is None:
            return {"generated": 0, "missing_session": True}
        session_folder = Path(session.session_folder)
        channel_count = max(1, int(session.channel_count or 1))
        sample_ids = [
            sid for (sid,) in db.query(Sample.id).filter(Sample.session_id == session_id).all()
        ]

    thumb_dir = session_folder / "previews" / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    total = len(sample_ids)
    if total == 0:
        on_progress(0, 0, "No samples to pre-generate")
        return {"generated": 0, "total": 0}

    on_progress(0, total, f"Pre-generating previews for {total} samples...")
    generated = 0
    failed = 0

    for i, sid in enumerate(sample_ids):
        if is_cancelled and is_cancelled():
            raise TaskCancelledException("Pregenerate previews cancelled")

        combined = thumb_dir / f"{sid}.jpg"
        per_channel_targets = [
            (ch, thumb_dir / f"{sid}_ch{ch}.jpg") for ch in range(channel_count)
        ]
        need_combined = not combined.exists()
        need_per_channel = [t for (_, t) in per_channel_targets if not t.exists()]

        if not need_combined and not need_per_channel:
            if (i + 1) % _THUMB_BATCH == 0 or i + 1 == total:
                on_progress(i + 1, total, f"Pre-generated {i + 1}/{total}")
            continue

        try:
            arr = lance_images.read_array(session_id, sid)
            if arr is None:
                failed += 1
                continue
            if need_combined:
                render_array_to_thumbnail(arr, combined)
            for ch, target in per_channel_targets:
                if target.exists():
                    continue
                try:
                    render_channel_to_thumbnail(arr, ch, target)
                except IndexError:
                    pass
            generated += 1
        except Exception:
            logger.exception("preview pregen failed for sample %s", sid)
            failed += 1

        if (i + 1) % _THUMB_BATCH == 0 or i + 1 == total:
            on_progress(i + 1, total, f"Pre-generated {i + 1}/{total}")

    return {"generated": generated, "failed": failed, "total": total}


def submit_pregenerate_previews(
    *,
    session_id: str,
    db_url: str,
) -> str:
    """Submit the pregen job to the task manager and return the task id."""
    return task_manager.submit(
        name="Pregenerate previews",
        func=pregenerate_previews,
        session_id=session_id,
        db_url=db_url,
    )


def preview_status(db: DbSession, session_id: str) -> dict:
    """Return preview readiness summary for the session-open UI.

    A session is "ready" when every import has completed AND every pregen task
    has finished. If any import or pregen is still running, returns progress.
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    imports = (
        db.query(Import)
        .filter(Import.session_id == session_id)
        .order_by(Import.created_at.desc())
        .all()
    )

    if not imports:
        return {"ready": True, "progress": 100.0, "message": "No imports", "phase": "ready"}

    # Find the latest non-terminal import; otherwise the most recent terminal one.
    pending_import = next(
        (imp for imp in imports if imp.status in ("pending", "loading", "preprocessing")),
        None,
    )
    if pending_import is not None:
        task = task_manager.get_task(pending_import.task_id) if pending_import.task_id else None
        info = task.to_dict() if task else None
        pct = float(info.get("percentage") or 0.0) if info else 0.0
        msg = (info.get("message") if info else None) or pending_import.status
        return {
            "ready": False,
            "progress": pct,
            "message": msg,
            "phase": "importing",
        }

    failed = [imp for imp in imports if imp.status in ("failed", "cancelled")]
    if failed and all(imp.status in ("failed", "cancelled") for imp in imports):
        return {
            "ready": False,
            "progress": 0.0,
            "message": f"Import {failed[0].status}",
            "phase": "failed",
        }

    # All completed imports — check pregen tasks.
    pregen_tasks = [
        imp.previews_task_id for imp in imports
        if imp.status == "completed" and imp.previews_task_id
    ]
    if not pregen_tasks:
        return {"ready": True, "progress": 100.0, "message": "Ready", "phase": "ready"}

    total_pct = 0.0
    running_msg = "Pre-generating previews..."
    all_done = True
    for tid in pregen_tasks:
        task = task_manager.get_task(tid)
        if task is None:
            continue
        info = task.to_dict()
        status = info.get("status")
        pct = float(info.get("percentage") or 0.0)
        total_pct += pct
        if status in ("pending", "running"):
            all_done = False
            running_msg = info.get("message") or running_msg
        elif status == "failed":
            return {
                "ready": False,
                "progress": 0.0,
                "message": info.get("error") or "Preview generation failed",
                "phase": "failed",
            }
    avg = total_pct / max(1, len(pregen_tasks))
    if all_done:
        return {"ready": True, "progress": 100.0, "message": "Ready", "phase": "ready"}
    return {"ready": False, "progress": avg, "message": running_msg, "phase": "pregenerating"}
