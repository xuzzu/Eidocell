from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.export import ExportRequest, ExportResult
from services import export_service

router = APIRouter(prefix="/sessions/{session_id}/export", tags=["export"])


@router.post("/", response_model=ExportResult)
def export_session(
    session_id: str,
    data: ExportRequest,
    db: DbSession = Depends(get_db),
):
    return export_service.export_session(
        db,
        session_id,
        data.output_directory,
        data.include_classes,
        data.include_clusters,
        data.include_masks,
        data.include_csv,
    )
