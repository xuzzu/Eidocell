from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.workspace.classes import ClassSummary, ClassStatistics
from schemas.workspace.gallery import SamplePage
from services.workspace import classes_service

router = APIRouter(prefix="/sessions/{session_id}/classes", tags=["classes"])


@router.get("/{class_id}/summary", response_model=ClassSummary)
def get_class_summary(
    session_id: str, class_id: str, db: DbSession = Depends(get_db)
):
    return classes_service.get_class_summary(db, class_id)


@router.get("/{class_id}/statistics", response_model=ClassStatistics)
def get_class_statistics(
    session_id: str, class_id: str, db: DbSession = Depends(get_db)
):
    return classes_service.get_class_statistics(db, class_id)


@router.get("/{class_id}/samples", response_model=SamplePage)
def list_class_samples(
    session_id: str,
    class_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: DbSession = Depends(get_db),
):
    items, total = classes_service.list_class_samples(db, class_id, offset, limit)
    return SamplePage(items=items, total=total, offset=offset, limit=limit)


@router.get("/{class_id}/preview")
def get_class_preview(
    session_id: str, class_id: str, db: DbSession = Depends(get_db)
):
    """Serve a collage preview image for this class."""
    path = classes_service.get_or_generate_collage(db, class_id)
    return FileResponse(path, media_type="image/jpeg")
