import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from db.session import engine, Base
from routers import sessions, features, export, settings, tasks, learning, notifications
from routers.workspace import gallery, classes, clusters, segmentation, analysis, similarity

logger = logging.getLogger("eidocell.migrations")

# Lightweight schema migrations (no Alembic)
def _run_migrations():
    """Apply schema changes that create_all cannot handle alone."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        # Gating refactor: rebuild the gates table with the new schema (nullable
        # plot_id, operator, source_gate_ids). Wipe existing gate rows — the new
        # selection-based semantics doesn't carry over the old per-gate is_active
        # union flags meaningfully.
        if "gates" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("gates")}
            if "operator" not in columns or "source_gate_ids" not in columns:
                logger.info("Migration: rebuilding gates table for boolean-gate refactor")
                conn.execute(text("DROP TABLE gates"))

        if "clusters" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("clusters")}
            if "quality_score" not in columns:
                logger.info("Migration: adding clusters.quality_score column")
                conn.execute(text("ALTER TABLE clusters ADD COLUMN quality_score FLOAT"))
            if "feature_method" not in columns:
                logger.info("Migration: adding clusters.feature_method column")
                conn.execute(text("ALTER TABLE clusters ADD COLUMN feature_method VARCHAR"))

        if "sessions" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("sessions")}
            if "selected_gate_id" not in columns:
                logger.info("Migration: adding sessions.selected_gate_id column")
                conn.execute(text(
                    "ALTER TABLE sessions ADD COLUMN selected_gate_id VARCHAR "
                    "REFERENCES gates(id) ON DELETE SET NULL"
                ))

        if "plots" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("plots")}
            if "parent_gate_id" not in columns:
                logger.info("Migration: adding plots.parent_gate_id column")
                conn.execute(text(
                    "ALTER TABLE plots ADD COLUMN parent_gate_id VARCHAR "
                    "REFERENCES gates(id) ON DELETE SET NULL"
                ))


_run_migrations()

# Create all tables (recreates the gates table after the rebuild above).
Base.metadata.create_all(bind=engine)


def _post_migration_fixups():
    """One-shot data fixups that need the new schema in place."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        if "samples" in inspector.get_table_names():
            # After the gating wipe, no session has a selected population anymore —
            # so every sample should be considered active again.
            conn.execute(text("UPDATE samples SET is_active = 1 WHERE is_active = 0"))
        if "sessions" in inspector.get_table_names():
            conn.execute(text("UPDATE sessions SET selected_gate_id = NULL"))


_post_migration_fixups()

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



@app.get("/health")
def health():
    return {"status": "ok"}
