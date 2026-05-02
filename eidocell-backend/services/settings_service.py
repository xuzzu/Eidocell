"""Settings service: app-level settings persisted to JSON."""

import json

from core.config import EIDOCELL_HOME
from schemas.settings import AppSettings

SETTINGS_PATH = EIDOCELL_HOME / "settings.json"


def get_settings() -> dict:
    settings = _load()
    return settings.model_dump()


def update_settings(updates: dict) -> dict:
    settings = _load()
    for key, value in updates.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    _save(settings)
    return settings.model_dump()


def reset_settings() -> dict:
    settings = AppSettings()
    _save(settings)
    return settings.model_dump()


def _load() -> AppSettings:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            return AppSettings(**data)
        except Exception:
            return AppSettings()
    return AppSettings()


def _save(settings: AppSettings) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings.model_dump(), indent=2))
