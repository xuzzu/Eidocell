from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.imports import ImportCreate, ImportOut, ImportSubmitOut, PreviewRequest
from services import import_service

router = APIRouter(prefix="/sessions/{session_id}/imports", tags=["imports"])


@router.post("/preview", response_model=dict)
def preview_preprocessing(session_id: str, payload: PreviewRequest, db: DbSession = Depends(get_db)):
    try:
        b64_img = import_service.preview_preprocessing(db, session_id, payload)
        return {"image_data": f"data:image/png;base64,{b64_img}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/", response_model=ImportSubmitOut, status_code=202)
def submit_import(session_id: str, payload: ImportCreate, db: DbSession = Depends(get_db)):
    import_id, task_id = import_service.run_import(db, session_id, payload)
    return {"import_id": import_id, "task_id": task_id}


@router.get("/", response_model=list[ImportOut])
def list_imports(session_id: str, db: DbSession = Depends(get_db)):
    return import_service.list_imports(db, session_id)


@router.get("/{import_id}", response_model=ImportOut)
def get_import(session_id: str, import_id: str, db: DbSession = Depends(get_db)):
    return import_service.get_import(db, session_id, import_id)


@router.delete("/{import_id}", response_model=ImportOut)
def cancel_import(session_id: str, import_id: str, db: DbSession = Depends(get_db)):
    return import_service.cancel_import(db, session_id, import_id)
