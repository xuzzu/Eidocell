import shutil
from pathlib import Path
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession
from sqlalchemy import func

from core.config import SESSIONS_DIR, SUPPORTED_IMAGE_EXTENSIONS
from models.models import Session, Sample, SampleClass
from schemas.sessions import SessionCreate, SessionUpdate


def _sanitize_folder_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


def _scan_images(directory: Path) -> list[str]:
    """Return sorted list of image filenames in a flat directory."""
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
    return sorted(
        f.name
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


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
    images_dir = Path(data.images_directory)
    image_files = _scan_images(images_dir)
    if not image_files:
        raise HTTPException(status_code=400, detail="No supported images found in directory")

    session = Session(
        name=data.name,
        images_directory=str(images_dir),
        session_folder="",  # will be set after we have the id
    )
    db.add(session)
    db.flush()  # generate id

    # Create session folder
    folder_name = f"{_sanitize_folder_name(data.name)}_{session.id}"
    session_folder = SESSIONS_DIR / folder_name
    session_folder.mkdir(parents=True, exist_ok=True)
    for sub in ("features", "masks", "masked_images", "thumbnails"):
        (session_folder / sub).mkdir(exist_ok=True)
    session.session_folder = str(session_folder)

    # Create default "Uncategorized" class
    uncategorized = SampleClass(
        session_id=session.id,
        name="Uncategorized",
        color="#808080",
    )
    db.add(uncategorized)
    db.flush()

    # Create sample records
    for idx, filename in enumerate(image_files):
        sample = Sample(
            session_id=session.id,
            filename=filename,
            path=str(images_dir / filename),
            storage_index=idx,
            class_id=uncategorized.id,
        )
        db.add(sample)

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

    db.commit()
    db.refresh(session)
    return session


def delete_session(db: DbSession, session_id: str) -> None:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove session folder from disk
    session_folder = Path(session.session_folder)
    if session_folder.exists():
        shutil.rmtree(session_folder)

    db.delete(session)
    db.commit()


def get_session_sample_count(db: DbSession, session_id: str) -> int:
    return db.query(func.count(Sample.id)).filter(Sample.session_id == session_id).scalar()
