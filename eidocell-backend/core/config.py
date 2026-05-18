import logging
import os
from pathlib import Path

# Bumped on storage-layer breaking changes; see main.py wipe logic.
STORAGE_VERSION = 10

# Root directory for all eidocell data
EIDOCELL_HOME = Path(os.environ.get("EIDOCELL_HOME", Path.home() / ".eidocell"))
SESSIONS_DIR = EIDOCELL_HOME / "sessions"
LANCE_DIR = EIDOCELL_HOME / "lance"
DATABASE_PATH = EIDOCELL_HOME / "eidocell.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
STORAGE_VERSION_FILE = EIDOCELL_HOME / "storage_version"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

MODELS_DIR = EIDOCELL_HOME / "models"

# Ensure directories exist
EIDOCELL_HOME.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
LANCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Logging ────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("eidocell")
    if logger.handlers:
        return  # already configured
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)


setup_logging(os.environ.get("EIDOCELL_LOG_LEVEL", "INFO"))
