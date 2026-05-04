from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.workspace.similarity import (
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from services.workspace import similarity_service

router = APIRouter(prefix="/sessions/{session_id}/similarity", tags=["similarity"])


@router.post("/search", response_model=SimilaritySearchResponse)
def search_similar(
    session_id: str,
    data: SimilaritySearchRequest,
    db: DbSession = Depends(get_db),
):
    return similarity_service.search(db, session_id, data)
