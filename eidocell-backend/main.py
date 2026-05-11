import logging
import os
import shutil

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import (
    DATABASE_PATH,
    EIDOCELL_HOME,
    LANCE_DIR,
    SESSIONS_DIR,
    STORAGE_VERSION,
    STORAGE_VERSION_FILE,
)
from db.session import engine, Base
from routers import sessions, features, export, settings, tasks, learning, notifications, imports as imports_router
from routers.workspace import gallery, classes, clusters, segmentation, analysis, similarity

logger = logging.getLogger("eidocell.storage")


def _check_storage_version() -> None:
    """Wipe ~/.eidocell when the on-disk version doesn't match STORAGE_VERSION.

    Per the v2 storage refactor, schema migrations are not maintained — bumping
    STORAGE_VERSION drops the SQLite DB, every Lance table, and every session
    folder. The user has explicitly opted into this trade-off.
    """
    try:
        current = int(STORAGE_VERSION_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        current = 0

    if current == STORAGE_VERSION:
        return

    logger.warning(
        "Storage version mismatch (on-disk=%s, expected=%s). Wiping ~/.eidocell.",
        current, STORAGE_VERSION,
    )
    shutil.rmtree(LANCE_DIR, ignore_errors=True)
    shutil.rmtree(SESSIONS_DIR, ignore_errors=True)
    DATABASE_PATH.unlink(missing_ok=True)

    EIDOCELL_HOME.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    LANCE_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_VERSION_FILE.write_text(str(STORAGE_VERSION))


_check_storage_version()
Base.metadata.create_all(bind=engine)


app = FastAPI(title="EidoCell", version="0.1.0")

_default_origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "app://.",
    "file://",
]
_cors_origins = os.environ.get("EIDOCELL_CORS_ORIGINS")
allowed_origins = _cors_origins.split(",") if _cors_origins else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(gallery.router)
app.include_router(classes.router)
app.include_router(clusters.router)
app.include_router(segmentation.router)
app.include_router(analysis.router)
app.include_router(similarity.router)
app.include_router(features.router)
app.include_router(export.router)
app.include_router(settings.router)
app.include_router(tasks.router)
app.include_router(learning.router)
app.include_router(notifications.router)
app.include_router(imports_router.router)



@app.get("/health")
def health():
    return {"status": "ok"}
