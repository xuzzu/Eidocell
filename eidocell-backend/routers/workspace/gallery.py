from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.workspace.gallery import (
    SampleOut,
    SamplePage,
    SampleListParams,
    ClassCreate,
    ClassUpdate,
    ClassOut,
    BulkClassAssignment,
)
from services.workspace import gallery_service
from services import session_service

router = APIRouter(prefix="/sessions/{session_id}", tags=["gallery"])


# ── Samples ──────────────────────────────────────────────────────────────


@router.post("/samples/list", response_model=SamplePage)
def list_samples(
    session_id: str,
    params: SampleListParams,
    db: DbSession = Depends(get_db),
):
    """List samples with filtering, sorting, and pagination.
    Uses POST because filter conditions can be complex."""
    session_service.get_session(db, session_id)  # validates session exists
    items, total = gallery_service.list_samples(db, session_id, params)
    return SamplePage(
        items=items,
        total=total,
        offset=params.offset,
        limit=params.limit,
    )


@router.get("/samples/{sample_id}", response_model=SampleOut)
def get_sample(session_id: str, sample_id: str, db: DbSession = Depends(get_db)):
    return gallery_service.get_sample(db, sample_id)


@router.get("/samples/{sample_id}/image")
def get_sample_image(
    session_id: str,
    sample_id: str,
    channel: int | None = None,
    db: DbSession = Depends(get_db),
):
    """Serve the original full-resolution image, optionally a single channel."""
    path = gallery_service.get_image_path(db, sample_id, channel=channel)
    return FileResponse(path)


@router.get("/samples/{sample_id}/thumbnail")
def get_sample_thumbnail(
    session_id: str,
    sample_id: str,
    db: DbSession = Depends(get_db),
):
    """Serve a JPEG thumbnail, generating it on first request."""
    session = session_service.get_session(db, session_id)
    path = gallery_service.get_thumbnail_path(db, sample_id, session.session_folder)
    return FileResponse(path, media_type="image/jpeg")


@router.get("/samples/{sample_id}/channel/{channel}/thumbnail")
def get_sample_channel_thumbnail(
    session_id: str,
    sample_id: str,
    channel: int,
    db: DbSession = Depends(get_db),
):
    """Serve a JPEG thumbnail of a single channel of a multi-channel sample."""
    session = session_service.get_session(db, session_id)
    path = gallery_service.get_channel_thumbnail_path(
        db, sample_id, session.session_folder, channel
    )
    return FileResponse(path, media_type="image/jpeg")


@router.get("/sortable-attributes")
def list_sortable_attributes(session_id: str, db: DbSession = Depends(get_db)):
    """List mask attribute / channel pairs available for sorting.

    Each item: ``{name, channel_index, channel_name, value}`` — ``value`` is the
    raw sort_by token (e.g. ``attr:area:ch:0``).
    """
    return gallery_service.list_sortable_attributes(db, session_id)


# ── Classes ──────────────────────────────────────────────────────────────


@router.get("/classes", response_model=list[ClassOut])
def list_classes(session_id: str, db: DbSession = Depends(get_db)):
    return gallery_service.list_classes(db, session_id)


@router.post("/classes", response_model=ClassOut, status_code=201)
def create_class(session_id: str, data: ClassCreate, db: DbSession = Depends(get_db)):
    return gallery_service.create_class(db, session_id, data)


@router.patch("/classes/{class_id}", response_model=ClassOut)
def update_class(session_id: str, class_id: str, data: ClassUpdate, db: DbSession = Depends(get_db)):
    return gallery_service.update_class(db, class_id, data)


@router.delete("/classes/{class_id}", status_code=204)
def delete_class(session_id: str, class_id: str, db: DbSession = Depends(get_db)):
    gallery_service.delete_class(db, class_id)


# ── Bulk operations ─────────────────────────────────────────────────────


@router.post("/samples/assign-class")
def assign_samples_to_class(
    session_id: str,
    data: BulkClassAssignment,
    db: DbSession = Depends(get_db),
):
    updated = gallery_service.assign_samples_to_class(db, data)
    return {"updated": updated}
