from fastapi import APIRouter

from schemas.settings import AppSettings, AppSettingsUpdate
from services import settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/", response_model=AppSettings)
def get_settings():
    return settings_service.get_settings()


@router.patch("/", response_model=AppSettings)
def update_settings(data: AppSettingsUpdate):
    return settings_service.update_settings(data.model_dump(exclude_unset=True))


@router.post("/reset", response_model=AppSettings)
def reset_settings():
    return settings_service.reset_settings()
