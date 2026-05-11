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

    Reads all required arrays in a single batched Lance scan keyed by missing
    sample_id, instead of one ``where=`` query per sample.
    """
    import numpy as np

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

    # Decide which sample_ids actually need work (idempotency).
    def _missing(sid: str) -> bool:
        if not (thumb_dir / f"{sid}.jpg").exists():
            return True
        for ch in range(channel_count):
            if not (thumb_dir / f"{sid}_ch{ch}.jpg").exists():
                return True
        return False

    work_ids = [sid for sid in sample_ids if _missing(sid)]
    skipped = total - len(work_ids)
    if not work_ids:
        on_progress(total, total, f"Pre-generated 0/{total} (already cached)")
        return {"generated": 0, "failed": 0, "total": total}

    on_progress(0, total, f"Pre-generating previews for {len(work_ids)} samples...")

    # One Lance scan per chunk of sample_ids — avoids per-sample table open +
    # SQL filter and uses the new pa.binary() decode path (~1500× faster).
    table_name = lance_store.images_table_name(session_id)
    try:
        table = lance_store.open_or_create_table(table_name, lance_images.schema(),
                                                 create_if_missing=False)
    except FileNotFoundError:
        return {"generated": 0, "failed": len(work_ids), "total": total}

    generated = 0
    failed = 0
    CHUNK = 256
    progressed = skipped
    for chunk_start in range(0, len(work_ids), CHUNK):
        if is_cancelled and is_cancelled():
            raise TaskCancelledException("Pregenerate previews cancelled")
        chunk = work_ids[chunk_start : chunk_start + CHUNK]
        ids_literal = ",".join(f"'{sid}'" for sid in chunk)
        where = (f"sample_id IN ({ids_literal}) "
                 f"AND channel_set = '{lance_images.DEFAULT_CHANNEL_SET}'")
        arrow = table.search().where(where).to_arrow()

        # Build a dict so we render in input order (cancellation-friendly).
        by_id: dict[str, tuple[bytes, int, int, int, str]] = {}
        for i in range(arrow.num_rows):
            sid = arrow.column("sample_id")[i].as_py()
            by_id[sid] = (
                arrow.column("data")[i].as_py(),         # bytes (pa.binary())
                arrow.column("height")[i].as_py(),
                arrow.column("width")[i].as_py(),
                arrow.column("n_channels")[i].as_py(),
                arrow.column("dtype")[i].as_py(),
            )

        for sid in chunk:
            if is_cancelled and is_cancelled():
                raise TaskCancelledException("Pregenerate previews cancelled")
            row = by_id.get(sid)
            if row is None:
                failed += 1
                progressed += 1
                continue
            data, h, w, c, dtype = row
            try:
                arr = np.frombuffer(data, dtype=dtype).copy()
                arr = arr.reshape((h, w)) if c == 1 else arr.reshape((h, w, c))
                combined = thumb_dir / f"{sid}.jpg"
                if not combined.exists():
                    render_array_to_thumbnail(arr, combined)
                for ch in range(channel_count):
                    target = thumb_dir / f"{sid}_ch{ch}.jpg"
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
            progressed += 1
            if progressed % _THUMB_BATCH == 0 or progressed == total:
                on_progress(progressed, total, f"Pre-generated {progressed}/{total}")

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
