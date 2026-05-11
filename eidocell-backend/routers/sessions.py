from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.sessions import SessionCreate, SessionUpdate, SessionOut, SessionListItem
from services import session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/", response_model=list[SessionListItem])
def list_sessions(db: DbSession = Depends(get_db)):
    return session_service.list_sessions(db)


@router.post("/", response_model=SessionOut, status_code=201)
def create_session(data: SessionCreate, db: DbSession = Depends(get_db)):
    session = session_service.create_session(db, data)
    count = session_service.get_session_sample_count(db, session.id)
    return _session_to_out(session, count)


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: DbSession = Depends(get_db)):
    session = session_service.get_session(db, session_id)
    count = session_service.get_session_sample_count(db, session.id)
    return _session_to_out(session, count)


@router.get("/{session_id}/preview-status")
def preview_status(session_id: str, db: DbSession = Depends(get_db)):
    """Return whether all previews are pregenerated for the session.

    Used by the session-open UI to block until the import + pregen pipeline
    finishes. Shape: ``{ready: bool, progress: float, message: str, phase: str}``.
    """
    return session_service.preview_status(db, session_id)


@router.patch("/{session_id}", response_model=SessionOut)
def update_session(session_id: str, data: SessionUpdate, db: DbSession = Depends(get_db)):
    session = session_service.update_session(db, session_id, data)
    count = session_service.get_session_sample_count(db, session.id)
    return _session_to_out(session, count)


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, db: DbSession = Depends(get_db)):
    session_service.delete_session(db, session_id)


def _session_to_out(session, sample_count: int) -> dict:
    return {
        "id": session.id,
        "name": session.name,
        "images_directory": session.images_directory or None,
        "session_folder": session.session_folder,
        "created_at": session.created_at,
        "last_opened_at": session.last_opened_at,
        "scale_factor": session.scale_factor,
        "scale_units": session.scale_units,
        "channel_count": session.channel_count,
        "channel_names": session.channel_names,
        "sample_count": sample_count,
    }
