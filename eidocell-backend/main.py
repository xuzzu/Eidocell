import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from db.session import engine, Base
from routers import sessions, features, export, settings, tasks, learning, notifications
from routers.workspace import gallery, classes, clusters, segmentation, analysis

logger = logging.getLogger("eidocell.migrations")

# Create all tables
Base.metadata.create_all(bind=engine)


# Lightweight schema migrations (no Alembic)
def _run_migrations():
    """Add columns that were added to models after initial table creation."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        if "gates" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("gates")}
            if "parent_gate_id" not in columns:
                logger.info("Migration: adding gates.parent_gate_id column")
                conn.execute(text(
                    "ALTER TABLE gates ADD COLUMN parent_gate_id VARCHAR REFERENCES gates(id)"
                ))

        if "clusters" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("clusters")}
            if "quality_score" not in columns:
                logger.info("Migration: adding clusters.quality_score column")
                conn.execute(text("ALTER TABLE clusters ADD COLUMN quality_score FLOAT"))
            if "feature_method" not in columns:
                logger.info("Migration: adding clusters.feature_method column")
                conn.execute(text("ALTER TABLE clusters ADD COLUMN feature_method VARCHAR"))


_run_migrations()

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
app.include_router(features.router)
app.include_router(export.router)
app.include_router(settings.router)
app.include_router(tasks.router)
app.include_router(learning.router)
app.include_router(notifications.router)



@app.get("/health")
def health():
    return {"status": "ok"}
