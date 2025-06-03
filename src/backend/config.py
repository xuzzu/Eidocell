# backend/config.py
import json
import logging
import os
import sys
from pathlib import Path

_IS_BUNDLED = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

if _IS_BUNDLED:
    _RUNTIME_PROJECT_ROOT = Path(sys._MEIPASS)
else:
    _RUNTIME_PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = (
    _RUNTIME_PROJECT_ROOT / "src" if not _IS_BUNDLED else _RUNTIME_PROJECT_ROOT / "src"
)


def resource_path(relative_path_from_project_root: str) -> str:
    if _IS_BUNDLED:
        return str(_RUNTIME_PROJECT_ROOT / relative_path_from_project_root)
    else:
        return str(_RUNTIME_PROJECT_ROOT / relative_path_from_project_root)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# UI Configs
WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 900
LIGHT_THEME_QSS_PATH = Path(
    resource_path(os.path.join("src", "UI", "resource", "light", "demo.qss"))
)
DARK_THEME_QSS_PATH = Path(
    resource_path(os.path.join("src", "UI", "resource", "dark", "demo.qss"))
)
FOLDER_CLOSE_ICON_PATH = ":/qfluentwidgets/images/folder_list_dialog/Close_{c}.png"
FOLDER_ADD_ICON_PATH = ":/qfluentwidgets/images/folder_list_dialog/Add_{c}.png"
APP_ICON_PATH = Path(resource_path(os.path.join("src", "UI", "resource", "icon.png")))
CLASS_FRAME_PATH = Path(
    resource_path(os.path.join("src", "UI", "resource", "class_frame.png"))
)


# Card Dimensions (These remain unchanged)
GALLERY_CARD_WIDTH = 128
GALLERY_CARD_HEIGHT = 136
GALLERY_CARD_IMAGE_HEIGHT = 78

CLASS_CARD_WIDTH = 128
CLASS_CARD_HEIGHT = 136

CLUSTERS_CARD_WIDTH = 180
CLUSTERS_CARD_HEIGHT = 198
CLUSTERS_CARD_IMAGE_HEIGHT = 145

ANALYSIS_CARD_WIDTH = 440
ANALYSIS_CARD_HEIGHT = 340
ANALYSIS_CARD_IMAGE_HEIGHT = 302

# Analysis View Configs (These remain unchanged)
DEFAULT_HISTOGRAM_BIN_COUNT = 10
HISTOGRAM_BIN_COUNT_RANGE = (5, 5000)

# Chart Fonts (These remain unchanged)
CHART_TITLE_FONT = dict(family="Arial", size=20, weight="bold")
CHART_AXIS_TITLE_FONT = dict(family="Arial", size=18, weight="bold")
CHART_DEFAULT_FONT = dict(family="Arial", size=14)

# Backend Configs
# Default Models
DEFAULT_MODELS_DICT = {
    "DINOv2": {
        "path": resource_path(
            os.path.join("src", "backend", "resources", "dinov2_weights.pth")
        ),
        "dimension": 384,
        "type": "pytorch",
        "architecture": "dinov2_vits14",
    },
    "MobileNetV3": {
        "path": resource_path(
            os.path.join("src", "backend", "resources", "mobilenetv3_small.pt")
        ),
        "dimension": 576,
        "type": "pytorch",
        "architecture": "mobilenetv3s",
    },
    # Add other ONNX default models here if they are bundled, e.g.:
    # "MobileNetV2_Segmentation_ONNX": {
    #     "path": resource_path(os.path.join("src", "backend", "resources", "mobilenetv2_segmentation.onnx")),
    #     "dimension": 0, # Or appropriate output if it's a feature extractor
    #     "type": "onnx",
    #     "architecture": "mobilenetv2_seg" # Example
    # },
}

CLUSTERING_N_ITER = 300
CLUSTERING_N_REDO = 10
CLUSTERING_DEFAULT_N_CLUSTERS = 10
# SEGMENTATION_MODEL_PATH = Path(
#     resource_path(
#         os.path.join("src", "backend", "resources", "mobilenetv2_segmentation.onnx")
#     )
# )

# Data Manager and Presenter Configs (These remain unchanged)
IMAGES_PER_PREVIEW = 25
COLLAGE_RES_SCALE = 0.3
SAMPLE_RES_SCALE = 0.5

DEFAULT_SETTINGS = {
    "theme": "light",
    "selected_model": "DINOv2",  # Renamed from 'model'
    "provider": "CPUExecutionProvider",
    "thumbnail_quality": 75,
    "images_per_collage": IMAGES_PER_PREVIEW,
    "collage_res_scale": COLLAGE_RES_SCALE,
    "custom_models": [],  # New field for custom models
}

if _IS_BUNDLED:
    _USER_DATA_ROOT = Path(sys.executable).parent
    SETTINGS_FILE = _USER_DATA_ROOT / "settings.json"
    SESSIONS_INDEX_FILE = _USER_DATA_ROOT / "sessions_index.json"
else:
    # Development paths
    SETTINGS_FILE = _RUNTIME_PROJECT_ROOT.parent / "settings.json"
    SESSIONS_INDEX_FILE = _RUNTIME_PROJECT_ROOT.parent / "sessions_index.json"


def load_settings():
    """Loads settings from a file, merging defaults and custom models."""
    try:
        # Ensure the directory for settings.json exists, especially for bundled app
        os.makedirs(SETTINGS_FILE.parent, exist_ok=True)
        with open(SETTINGS_FILE, "r") as f:
            settings_from_file = json.load(f)
        settings = DEFAULT_SETTINGS.copy()
        settings.update(settings_from_file)
        if not isinstance(settings.get("custom_models"), list):
            settings["custom_models"] = []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(
            f"Settings file not found or invalid ({SETTINGS_FILE}, Error: {e}). Using default settings."
        )
        settings = DEFAULT_SETTINGS.copy()
        save_settings(settings)
    return settings


def save_settings(settings):
    """Saves settings to a file."""
    try:
        os.makedirs(SETTINGS_FILE.parent, exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
    except IOError as e:
        logging.error(f"Failed to save settings to {SETTINGS_FILE}: {e}")


def get_available_models() -> dict:
    """
    Returns a dictionary of all available models (default + custom).
    Keys are model names, values are dicts with 'path' and 'dimension'.
    """
    settings = load_settings()
    available_models = (
        DEFAULT_MODELS_DICT.copy()
    )  # Default models paths are already resource_path'd

    for custom_model in settings.get("custom_models", []):
        name = custom_model.get("name")
        path_str = custom_model.get("path")  # User provides this path
        dimension = custom_model.get("dimension")
        model_type = custom_model.get(
            "type", "onnx"
        )  # Default to onnx if not specified
        architecture = custom_model.get("architecture")

        if (
            name and path_str and isinstance(dimension, int) and dimension >= 0
        ):  # Allow 0 dim for non-feature models
            if name in available_models:
                logging.warning(
                    f"Custom model name '{name}' clashes with a default model or another custom model. Using current custom model entry."
                )

            if os.path.exists(path_str):
                available_models[name] = {
                    "path": path_str,
                    "dimension": dimension,
                    "type": model_type,
                    "architecture": architecture,
                }
            else:
                logging.warning(
                    f"Custom model '{name}' path not found: {path_str}. Skipping."
                )
        else:
            logging.warning(f"Invalid custom model entry skipped: {custom_model}")

    return available_models
