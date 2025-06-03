# backend/data_manager.py
import csv
import json
import logging
import os
import random
import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from sklearn.preprocessing import StandardScaler
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor

from backend.config import get_available_models
from backend.database_manager import DatabaseManager
from backend.objects.cluster import Cluster
from backend.objects.mask import Mask
from backend.objects.sample import Sample
from backend.objects.sample_class import SampleClass
from backend.objects.session import Session
from backend.processor import Processor
from backend.pytorch_processor import PyTorchFeatureExtractor
from backend.utils.file_utils import atomic_write
from UI.navigation_interface.workspace.views.analysis.chart_configurations.parameter_holders import (
    HistogramParameters,
    ScatterParameters,
)
from UI.navigation_interface.workspace.views.analysis.plot_widget.gate import (
    BaseGate,
    IntervalGate,
    PolygonGate,
    RectangularGate,
    _random_bright_colour,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

FEATURES_FILENAME = "session_features.npy"
MASKS_FILENAME = "session_masks.npy"
# sample_order.json is replaced by samples.storage_index in DB


class DataManager(QObject):
    """
    Manages and stores image data, masks, features, clusters, and image classes.
    Interacts with the database via DatabaseManager.
    Features and masks are stored in consolidated .npy files.
    """

    samples_ready = Signal(dict)  # current self.samples
    class_added = Signal(str)  # class_id
    class_updated = Signal(str)  # class_id
    class_deleted = Signal(str)  # class_id
    images_loaded = Signal()

    feature_extraction_progress = Signal(int)  # Percentage of features extracted
    features_extracted_all = Signal()

    clustering_performed = Signal()  # After clustering and DB updates
    cluster_split = Signal(str, list)  # old_cluster_id, new_cluster_ids
    cluster_merged = Signal(str, list)  # new_cluster_id, old_cluster_ids

    plot_added_to_dm = Signal(str, object)  # plot_id, plot_configuration_obj
    plot_deleted_from_dm = Signal(str)  # plot_id
    gate_added_to_dm = Signal(str, object)  # plot_id (of parent plot), gate_obj
    gate_updated_in_dm = Signal(str, object)  # plot_id (of parent plot), gate_obj
    gate_deleted_from_dm = Signal(str, str)  # plot_id (of parent plot), gate_id
    active_samples_updated = Signal(dict)

    mask_created_in_db = Signal(
        object
    )  # mask_id (after DB entry and Mask object creation)
    mask_deleted_from_db = Signal(
        str
    )  # mask_id (after DB update and Mask object removal)

    features_invalidated = Signal()  # When model changes, features are wiped
    clusters_invalidated = Signal()  # When features or model change

    session_data_loaded = (
        Signal()
    )  # After all initial loading (samples, features, masks, classes, clusters)
    export_finished = Signal(bool, str)  # success_status, message_or_path

    # Internal state for consolidated data
    _all_features_array: Optional[np.ndarray] = None
    _all_masks_array: Optional[np.ndarray] = (
        None  # Assuming common mask dimensions for now
    )
    _sample_id_to_storage_index: Dict[str, int] = {}
    _storage_index_to_sample_id: Dict[int, str] = {}
    _is_features_dirty: bool = (
        False  # Flag to indicate if _all_features_array needs saving
    )
    _is_masks_dirty: bool = False  # Flag to indicate if _all_masks_array needs saving
    _common_mask_height: Optional[int] = None
    _common_mask_width: Optional[int] = None
    _current_features_model_name: Optional[str] = None
    _current_features_dimension: Optional[int] = None

    classifier_training_progress = Signal(
        str, int, float
    )  # classifier_name, epoch, metric
    classifier_training_finished = Signal(
        str, bool, str
    )  # classifier_name, success, message
    classification_run_finished = Signal(
        str, bool, str
    )  # classifier_name, success, message

    all_gate_populations_recalculated = Signal(list)
    all_mask_attributes_recalculated = Signal()

    def __init__(self, session: Session, settings: dict, segmentation_instance) -> None:
        super().__init__()
        self.session: Session = session
        self.settings = settings
        self.samples: Dict[str, Sample] = {}
        self.active_samples: Dict[str, Sample] = (
            {}
        )  # DERIVED: Samples passing global gates
        self.clusters: Dict[str, Cluster] = {}
        self.classes: Dict[str, SampleClass] = {}
        self.masks: Dict[str, Mask] = (
            {}
        )  # Stores Mask objects, Mask.mask_data will be slice from _all_masks_array
        self.plot_configurations: Dict[str, Any] = (
            {}
        )  # Stores PlotConfiguration-like objects (or dicts)
        self.gates: Dict[str, Any] = (
            {}
        )  # Stores BaseGate derivative instances (RectangularGate, etc.)
        self.active_global_gate_ids: Set[str] = (
            set()
        )  # IDs of gates for global AND filter
        self.pending_db_operations: List[Tuple[str, tuple]] = (
            []
        )  # For DB metadata, not file paths

        self.available_models = get_available_models()
        selected_model = self.settings.get(
            "selected_model",
            list(self.available_models.keys())[0] if self.available_models else "",
        )
        provider = self.settings.get("provider", "CPUExecutionProvider")
        self.segmentation_instance = segmentation_instance

        self.processor: Optional[Processor] = None
        if selected_model:
            try:
                self.processor = Processor(
                    model_name=selected_model,
                    available_models=self.available_models,
                    execution_provider=provider,
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                logging.critical(
                    f"Failed to initialize Processor for {selected_model}: {e}. Feature extraction disabled.",
                    exc_info=True,
                )
        else:
            logging.warning(
                "No feature extraction model selected or available. Feature extraction disabled."
            )

        db_path = os.path.join(self.session.metadata_directory, "metadata.db")
        self.db_manager = DatabaseManager(db_path)
        self.db_manager.connect()  # Ensures tables are created if they don't exist

        # Initialize internal arrays and maps
        self._all_features_array = None
        self._all_masks_array: Optional[np.ndarray] = None
        self._sample_id_to_storage_index = {}
        self._storage_index_to_sample_id = {}
        self._is_features_dirty = False
        self._is_masks_dirty = False

    # Slots

    @Slot()
    def _on_load_session_requested(self):
        logging.debug("DataManager received request: load_session")
        self.load_session_data()

    @Slot(str)
    def _on_load_images_from_folder_requested(self, folder_path: str):
        logging.debug(
            f"DataManager received request: load_images_from_folder({folder_path})"
        )
        self.load_images_from_folder(folder_path)

    @Slot()
    def _on_request_samples_requested(self):
        logging.debug("DataManager received request: request_samples")
        self.samples_ready.emit(self.samples.copy())

    @Slot()
    def _on_extract_all_features_requested(self):
        logging.debug("DataManager received request: extract_all_features")
        self._extract_all_features_internal()

    @Slot(str)
    def _on_extract_feature_requested(self, image_id: str):
        logging.warning(
            f"DataManager received request: extract_feature({image_id}). This will trigger re-extraction for all samples."
        )
        self._extract_all_features_internal()

    @Slot(str)
    def _on_delete_features_requested(self, image_id: str):
        logging.warning(
            f"DataManager received request: delete_features({image_id}). This is not fully supported with consolidated storage without re-write."
        )
        if (
            image_id in self._sample_id_to_storage_index
            and self._all_features_array is not None
        ):
            idx = self._sample_id_to_storage_index[image_id]
            if 0 <= idx < self._all_features_array.shape[0]:
                self._all_features_array[idx] = 0
                self._is_features_dirty = True
                self._persist_consolidated_data()  # Persist immediately
                if image_id in self.samples:
                    self.samples[image_id].features = None
                logging.info(f"Features for {image_id} zeroed out. File needs re-save.")
                self.features_invalidated.emit()
            else:
                logging.error(
                    f"Invalid storage index {idx} for sample {image_id} during feature deletion."
                )
        else:
            logging.warning(
                f"No features to delete for sample {image_id} or features not loaded."
            )

    @Slot(float, str)  # New slot
    def _on_set_session_scale_requested(self, scale_factor: float, units: str):
        if self.session:
            self.session.scale_factor = scale_factor
            self.session.scale_units = units
            logging.info(
                f"Session scale updated: {scale_factor} {units}/pixel for session {self.session.id}"
            )
            # Persist the session_info.json
            try:
                atomic_write(self.session.session_info_path, self.session.to_dict())
                self._recalculate_mask_attributes()
                logging.info(
                    f"Persisted updated session_info.json for session {self.session.id} with new scale."
                )
            except Exception as e:
                logging.error(
                    f"Failed to save session_info.json after scale update: {e}",
                    exc_info=True,
                )
        else:
            logging.error("Cannot set session scale: No active session in DataManager.")

    @Slot(int, int, int, bool)
    def _on_perform_clustering_requested(self, ui_params):
        logging.debug(f"Starting clustering with parameters: {ui_params}")
        self.perform_clustering(ui_params)

    @Slot(str, int, int, int)
    def _on_split_cluster_requested(
        self, cluster_id: str, n_clusters: int, n_iter: int, n_redo: int
    ):
        logging.debug(
            f"DataManager received request: split_cluster({cluster_id}, K={n_clusters})"
        )
        self.split_cluster(cluster_id, n_clusters, n_iter, n_redo)

    @Slot(list)
    def _on_merge_clusters_requested(self, cluster_ids: list):
        logging.debug(f"DataManager received request: merge_clusters({cluster_ids})")
        self.merge_clusters(cluster_ids)

    @Slot()
    def _on_clear_clusters_requested(self):
        logging.debug("DataManager received request: clear_clusters")
        self.clear_clusters()  # This method will need to be updated

    @Slot(str)
    def _on_delete_cluster_requested(self, cluster_id: str):
        logging.debug(f"DataManager received request: delete_cluster({cluster_id})")
        self.delete_cluster(cluster_id)  # This method will need to be updated

    @Slot(str, str)
    def _on_create_class_requested(self, name: str, color: str):
        logging.debug(f"DataManager received request: create_class({name})")
        self.create_class(name, color)

    @Slot(str, str)
    def _on_rename_class_requested(self, class_id: str, new_name: str):
        logging.debug(
            f"DataManager received request: rename_class({class_id} to '{new_name}')"
        )
        self.rename_class(class_id, new_name)

    @Slot(list, str)
    def _on_assign_images_to_class_requested(self, image_ids: list, new_class_id: str):
        logging.debug(
            f"DataManager received request: assign_images_to_class(to={new_class_id})"
        )
        self.assign_images_to_class(image_ids, new_class_id)

    @Slot(str)
    def _on_delete_class_requested(self, class_id: str):
        logging.debug(f"DataManager received request: delete_class({class_id}")
        self.delete_class(class_id)

    @Slot(str)
    def _on_delete_mask_requested(self, mask_id: str):
        logging.debug(f"DataManager received request: delete_mask({mask_id})")
        self.delete_mask(mask_id)

    @Slot(dict)
    def _on_update_processor_requested(self, settings: dict):
        logging.debug("DataManager received request: update_processor")
        self.update_processor(settings)

    @Slot(dict)
    def _on_export_data_requested(self, params: dict):
        logging.debug("DataManager received request: export_data")
        self.export_data(params)

    @Slot(object)
    def _on_flush_db_requested(self, callback: Optional[callable] = None):
        logging.debug(
            "DataManager received request: flush_db (pending DB ops and consolidated data)"
        )
        self._flush_pending_db_operations(
            lambda: self._persist_consolidated_data(callback)
        )

    def update_active_samples(self):
        """
        Updates self.active_samples based on self.samples and self.active_global_gate_ids.
        If multiple global gates are active, samples that fall into ANY of them will be included (OR logic).
        Emits active_samples_updated signal.
        """
        logging.debug(
            f"DataManager: Updating active_samples. Global gates (OR logic): {self.active_global_gate_ids}"
        )

        if not self.active_global_gate_ids:
            # No global gates, so all *originally active* samples are "active_samples"
            self.active_samples = {
                s_id: s_obj for s_id, s_obj in self.samples.items() if s_obj.is_active
            }
            logging.info(
                f"DataManager: Global filters cleared. Active samples count: {len(self.active_samples)}"
            )
        else:
            # Start with an empty set for the union of samples passing any active global gate
            union_of_passing_sample_ids: Set[str] = set()

            for gate_id_str in self.active_global_gate_ids:
                gate_obj = self.gates.get(gate_id_str)
                if not gate_obj:
                    logging.warning(
                        f"Global filter: Gate ID {gate_id_str} not found. Skipping this filter."
                    )
                    continue

                # gate_obj.ids_in_gate should be populated by recalculate_gate_population
                sample_ids_in_this_gate = set(gate_obj.ids_in_gate)
                union_of_passing_sample_ids.update(sample_ids_in_this_gate)

            self.active_samples = {
                s_id: self.samples[s_id]
                for s_id in union_of_passing_sample_ids
                if s_id in self.samples and self.samples[s_id].is_active
            }
            logging.info(
                f"DataManager: Global OR filters applied. Active samples count: {len(self.active_samples)}"
            )

        self.clusters_invalidated.emit()
        self.active_samples_updated.emit(self.active_samples.copy())

    def _recalculate_mask_attributes(self):
        """
        Recalculates attributes for all existing masks using the current session scale factor.
        This should be called after the session's scale_factor has been updated.
        """
        if not self.session or self.session.scale_factor is None:
            logging.info(
                "Recalculation of mask attributes skipped: No session or no scale factor set."
            )
            return

        if not self.segmentation_instance:
            logging.error(
                "Cannot recalculate mask attributes: SegmentationModel instance not available in DataManager."
            )
            return

        logging.info(
            f"Starting recalculation of all mask attributes with scale: {self.session.scale_factor} {self.session.scale_units or ''}"
        )
        updated_masks_count = 0

        for sample_id, sample in self.samples.items():
            if not sample.is_active or not sample.mask_id:
                continue

            mask_obj = self.masks.get(sample.mask_id)
            if not mask_obj:
                logging.warning(
                    f"Mask {sample.mask_id} for sample {sample_id} not found in self.masks during recalculation."
                )
                continue

            # Ensure mask_data (binary numpy array) is available
            if mask_obj.mask_data is None:
                if (
                    self._all_masks_array is not None
                    and sample.storage_index is not None
                    and 0 <= sample.storage_index < self._all_masks_array.shape[0]
                ):
                    mask_obj.mask_data = self._all_masks_array[sample.storage_index]
                else:
                    logging.warning(
                        f"Binary mask data not available for mask {mask_obj.id} (sample {sample_id}). Cannot recalculate attributes."
                    )
                    continue

            if not sample.path or not os.path.exists(sample.path):
                logging.warning(
                    f"Image path for sample {sample_id} is invalid. Cannot recalculate intensity attributes for mask {mask_obj.id}."
                )
                # We can still calculate geometric properties if intensity is not critical or handled by get_object_properties
                # For now, let get_object_properties handle it.

            try:
                new_attributes = self.segmentation_instance.get_object_properties(
                    image_path=sample.path,
                    mask=mask_obj.mask_data,
                    scale_factor=self.session.scale_factor,
                )
                mask_obj.attributes = new_attributes  # Update in-memory Mask object

                # Queue DB update for these attributes
                # save_mask_attributes deletes old and inserts new, so it's an update.
                self.add_pending_db_operation(
                    "DELETE FROM mask_attributes WHERE mask_id = ?", (mask_obj.id,)
                )
                for name, value in new_attributes.items():
                    value_str = str(value)
                    self.add_pending_db_operation(
                        "INSERT INTO mask_attributes (mask_id, attribute_name, attribute_value) VALUES (?, ?, ?)",
                        (mask_obj.id, name, value_str),
                    )
                updated_masks_count += 1
            except Exception as e:
                logging.error(
                    f"Error recalculating attributes for mask {mask_obj.id} (sample {sample_id}): {e}",
                    exc_info=True,
                )

        if updated_masks_count > 0:
            self._flush_pending_db_operations(
                callback=lambda: self.all_mask_attributes_recalculated.emit()
            )
            logging.info(
                f"Successfully recalculated and queued updates for attributes of {updated_masks_count} masks."
            )
        else:
            logging.info(
                "No mask attributes were updated during recalculation (or no masks to update)."
            )
            # Still emit if logic implies state might have changed for UI
            self.all_mask_attributes_recalculated.emit()

    @Slot(
        str, str, str, str, bool
    )  # image_id, attributes_json, mask_path (now irrelevant), masked_image_preview_path, update_db
    def _on_create_mask_requested(
        self,
        image_id: str,
        attributes_json: str,
        mask_file_path_ignored: str,
        masked_image_preview_path: str,
        update_db: bool,
    ):
        logging.debug(
            f"DataManager received request: create_mask for {image_id} (update_db={update_db})"
        )
        attributes_dict = {}
        try:
            if attributes_json:
                attributes_dict = json.loads(attributes_json)
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to decode attributes JSON for mask of image {image_id}: {e}. JSON: '{attributes_json}'"
            )

        if image_id not in self.samples:
            logging.error(f"Cannot create mask for non-existent sample ID: {image_id}")
            return

        sample = self.samples[image_id]
        if sample.mask_id and sample.mask_id in self.masks:  # Existing mask
            old_mask_id = sample.mask_id
            # We are replacing the mask. The old mask's data in _all_masks_array is now orphaned.
            # The Mask object for old_mask_id will be removed.
            self.db_manager.delete_mask(old_mask_id)  # Removes from DB
            del self.masks[old_mask_id]
            logging.info(
                f"Removed old mask {old_mask_id} for sample {image_id} before creating new one."
            )

        new_mask_id = str(uuid.uuid4())

        mask_obj = Mask(
            id=new_mask_id,
            image_id=image_id,
            attributes=attributes_dict,
            masked_image_path=masked_image_preview_path,
            mask_data=None,  # Will be populated from _all_masks_array by storage_index
        )
        self.masks[new_mask_id] = mask_obj

        if update_db:
            self.db_manager.create_mask(
                new_mask_id, image_id, masked_image_preview_path
            )
            self.db_manager.save_mask_attributes(new_mask_id, attributes_dict)
            # Sample.mask_id is updated by db_manager.create_mask
            sample.mask_id = new_mask_id
            self.mask_created_in_db.emit(mask_obj)

        # Flag that masks data might need saving (if a new mask array was actually added to a buffer)
        # self._is_masks_dirty = True
        # Actual saving of masks.npy will be handled by a persist method.

    #  Core Data Loading and Persistence

    def add_pending_db_operation(self, query: str, params: Tuple = ()) -> None:
        """
        Queues a database operation (SQL query and parameters) to be executed later
        by _flush_pending_db_operations.

        Args:
            query (str): The SQL query string.
            params (tuple, optional): Parameters for the SQL query. Defaults to ().
        """
        self.pending_db_operations.append((query, params))

    def load_session_data(self) -> None:
        """Loads all session data: samples, features, masks, classes, clusters, plots, gates."""
        logging.info(f"Loading session data for {self.session.id}...")
        db_samples_count = len(self.db_manager.get_all_samples(include_inactive=True))

        if (
            db_samples_count == 0
            and self.session.images_directory
            and os.path.isdir(self.session.images_directory)
        ):
            logging.info(
                f"Session {self.session.id} appears new. Initial image import from: {self.session.images_directory}"
            )
            self.load_images_from_folder(self.session.images_directory)
        else:
            logging.info(
                f"Session {self.session.id} has existing sample data. Loading normally."
            )
            self._load_samples_and_mappings()

        self._load_consolidated_features()
        self._load_consolidated_masks()
        self.load_classes()
        self.load_clusters()
        self._load_plots_and_gates()

        if self.gates:
            logging.info("Recalculating populations for all loaded gates...")
            self.recalculate_all_gate_populations()

        self.update_active_samples()

        self.images_loaded.emit()
        self.session_data_loaded.emit()
        QTimer.singleShot(0, lambda: self.samples_ready.emit(self.samples.copy()))
        logging.info("Session data loading complete.")

    def _load_plots_and_gates(self):
        """Loads plot configurations and gate objects from the database."""
        self.plot_configurations.clear()
        self.gates.clear()
        self.active_global_gate_ids.clear()  # Clear this set before loading
        logging.info("Loading plots and gates from database...")

        # Load Plots (no change here, this part was fine)
        db_plots = self.db_manager.get_all_plots()
        for plot_data_dict in db_plots:
            plot_id = plot_data_dict["id"]
            try:
                params_json_str = plot_data_dict["parameters_json"]
                params_dict_from_json = json.loads(params_json_str)
                chart_type = plot_data_dict["chart_type"]
                parameters_obj_instance = None
                if chart_type == "histogram":
                    parameters_obj_instance = HistogramParameters(
                        **params_dict_from_json
                    )
                elif chart_type == "scatter":
                    parameters_obj_instance = ScatterParameters(**params_dict_from_json)
                else:
                    parameters_obj_instance = params_dict_from_json
                plot_config_for_memory = {
                    "id": plot_id,
                    "name": plot_data_dict["name"],
                    "chart_type": chart_type,
                    "parameters_obj": parameters_obj_instance,
                    "created_at": plot_data_dict["created_at"],
                }
                self.plot_configurations[plot_id] = plot_config_for_memory
            except Exception as e:
                logging.error(
                    f"Error loading plot {plot_id} from DB: {e}", exc_info=True
                )
        logging.info(f"Loaded {len(self.plot_configurations)} plot configurations.")

        db_gates = self.db_manager.get_all_gates()
        for gate_data_dict in db_gates:
            gate_id_str = gate_data_dict["id"]  # This is a string from DB
            plot_id_for_gate = gate_data_dict["plot_id"]

            if plot_id_for_gate not in self.plot_configurations:
                logging.warning(
                    f"Gate {gate_id_str} references non-existent plot {plot_id_for_gate}. Skipping gate load."
                )
                continue
            try:
                gate_type = gate_data_dict["gate_type"]
                definition_dict = json.loads(gate_data_dict["definition_json"])
                params_tuple = tuple(
                    json.loads(gate_data_dict["parameters_tuple_json"])
                )
                color = (
                    QColor(gate_data_dict["color"])
                    if gate_data_dict["color"]
                    else _random_bright_colour()
                )
                name = gate_data_dict["name"]

                gate_obj: Optional[BaseGate] = None
                if gate_type == "rectangular":
                    gate_obj = RectangularGate(
                        x=definition_dict["x"],
                        y=definition_dict["y"],
                        width=definition_dict["width"],
                        height=definition_dict["height"],
                        name=name,
                        color=color,
                        parameters=params_tuple,
                    )
                elif gate_type == "polygon":
                    vertices_qpoints = [
                        QPointF(v[0], v[1]) for v in definition_dict["vertices"]
                    ]
                    gate_obj = PolygonGate(
                        vertices_qpoints,
                        name=name,
                        color=color,
                        parameters=params_tuple,
                    )
                elif gate_type == "interval":
                    gate_obj = IntervalGate(
                        min_val=definition_dict["min_val"],
                        max_val=definition_dict["max_val"],
                        name=name,
                        color=color,
                        parameter_name=params_tuple[0] if params_tuple else None,
                    )

                if gate_obj:
                    gate_obj.id = uuid.UUID(
                        gate_id_str
                    )  # Convert string from DB to UUID object for consistency
                    gate_obj.plot_id = plot_id_for_gate
                    self.gates[gate_id_str] = (
                        gate_obj  # Store with string ID as key (consistent with DB)
                    )

                    self.active_global_gate_ids.add(gate_id_str)
                else:
                    logging.warning(
                        f"Unsupported gate_type '{gate_type}' for gate {gate_id_str}."
                    )
            except Exception as e:
                logging.error(
                    f"Error loading gate {gate_id_str} from DB: {e}", exc_info=True
                )
        logging.info(
            f"Loaded {len(self.gates)} gates. Active global gates set: {self.active_global_gate_ids}"
        )

    def recalculate_gate_population(self, gate: Any):  # gate is BaseGate derivative
        if not self.samples:
            gate.ids_in_gate = []
            gate.event_count = 0
            return

        gate.ids_in_gate = []
        num_params_for_gate = len(gate.parameters)

        for sample_id, sample in self.samples.items():
            if not sample.is_active:
                continue

            if not sample.mask_id or sample.mask_id not in self.masks:
                if num_params_for_gate > 0 and any(
                    p not in vars(sample) for p in gate.parameters
                ):  # Simple check
                    continue

            mask_attributes = (
                self.masks[sample.mask_id].attributes
                if sample.mask_id and sample.mask_id in self.masks
                else {}
            )

            try:
                param_values_for_gate = []
                all_params_found = True
                for param_name in gate.parameters:
                    if param_name in mask_attributes:
                        param_values_for_gate.append(
                            float(mask_attributes[param_name])
                        )  # Ensure float for comparison
                    elif hasattr(
                        sample, param_name.lower()
                    ):  # Check direct sample attributes (e.g. 'id')
                        param_values_for_gate.append(
                            getattr(sample, param_name.lower())
                        )
                    else:
                        all_params_found = False
                        break

                if all_params_found and param_values_for_gate:  # Ensure we got values
                    if gate.is_point_inside(*param_values_for_gate):
                        gate.ids_in_gate.append(sample.id)
            except (ValueError, TypeError) as e:
                pass  # Skip sample if attribute conversion fails
            except Exception as e_generic:
                logging.error(
                    f"Generic error processing sample {sample_id} for gate {gate.id}: {e_generic}",
                    exc_info=True,
                )

        gate.event_count = len(gate.ids_in_gate)

    def recalculate_all_gate_populations(self):
        if not self.samples:  # No samples, clear all gate populations
            for gate in self.gates.values():
                gate.ids_in_gate = []
                gate.event_count = 0
            if self.gates:
                self.all_gate_populations_recalculated.emit(list(self.gates.values()))
            return

        logging.info(
            f"DataManager: Recalculating populations for all {len(self.gates)} gates..."
        )
        for gate_obj in self.gates.values():
            self.recalculate_gate_population(gate_obj)

        if self.gates:
            self.all_gate_populations_recalculated.emit(list(self.gates.values()))
        logging.info("DataManager: Finished recalculating all gate populations.")

    def _load_samples_and_mappings(
        self,
    ):  # REVISED to include loading sample.cluster_ids
        """Loads sample metadata from DB, populates index mappings, and sample's cluster_ids."""
        self.samples.clear()
        self._sample_id_to_storage_index.clear()
        self._storage_index_to_sample_id.clear()

        db_samples_data = self.db_manager.get_all_samples(include_inactive=False)

        temp_samples_for_sorting = []
        for sample_data_dict in db_samples_data:
            sample_obj = Sample.from_dict(
                sample_data_dict
            )  # from_dict populates basic fields

            # Populate sample.cluster_ids from the DB (samples_clusters table)
            cluster_ids_for_this_sample = self.db_manager.get_clusters_for_sample(
                sample_obj.id
            )
            sample_obj.cluster_ids = set(
                cluster_ids_for_this_sample
            )  # Ensure it's a set

            if sample_obj.storage_index is None or sample_obj.storage_index < 0:
                logging.warning(
                    f"Sample {sample_obj.id} has invalid storage_index ({sample_obj.storage_index})."
                )
            temp_samples_for_sorting.append(sample_obj)

        temp_samples_for_sorting.sort(
            key=lambda s: (
                s.storage_index if s.storage_index is not None else float("inf")
            )
        )

        for sample_obj_sorted in temp_samples_for_sorting:
            self.samples[sample_obj_sorted.id] = sample_obj_sorted
            if (
                sample_obj_sorted.storage_index is not None
                and sample_obj_sorted.storage_index >= 0
            ):
                self._sample_id_to_storage_index[sample_obj_sorted.id] = (
                    sample_obj_sorted.storage_index
                )
                self._storage_index_to_sample_id[sample_obj_sorted.storage_index] = (
                    sample_obj_sorted.id
                )

        logging.info(
            f"Loaded {len(self.samples)} active samples from DB, built index maps, and populated sample cluster IDs."
        )

    @Slot()
    def load_clusters(self) -> None:
        """Loads clusters from the database and populates in-memory structures."""
        self.clusters.clear()
        clusters_data_from_db = self.db_manager.get_all_clusters()

        for cluster_entry_dict in clusters_data_from_db:
            cluster_obj = Cluster.from_dict(cluster_entry_dict)
            self.clusters[cluster_obj.id] = cluster_obj

        for sample_obj in self.samples.values():
            for (
                cluster_id_for_sample
            ) in (
                sample_obj.cluster_ids
            ):  # sample.cluster_ids is populated during sample loading
                if cluster_id_for_sample in self.clusters:
                    self.clusters[cluster_id_for_sample].add_image(
                        sample_obj
                    )  # add_image also updates sample's cluster_ids (redundant here but safe)
                else:
                    logging.warning(
                        f"Sample {sample_obj.id} refers to cluster_id {cluster_id_for_sample} not found in loaded clusters."
                    )

        logging.info(
            f"Successfully loaded and linked {len(self.clusters)} clusters with their samples."
        )

    @Slot()  # Make it a slot if it needs to be callable via invokeMethod or direct signal
    def load_classes(self) -> None:
        """Loads classes from the database and populates in-memory structures."""
        self.classes.clear()  # Clear existing in-memory classes
        classes_data_from_db = self.db_manager.get_all_classes()

        for class_entry_dict in classes_data_from_db:
            # Create SampleClass object from DB data
            # The SampleClass.from_dict method should NOT try to load samples itself.
            # We will link samples to classes after all samples and classes are loaded.
            image_class_obj = SampleClass.from_dict(class_entry_dict)
            self.classes[image_class_obj.id] = image_class_obj
            # Logging for each class loaded can be verbose, consider summarizing.
            # logging.debug(f"Loaded class metadata: {image_class_obj.name} (ID: {image_class_obj.id})")

        # After all Sample objects are loaded (in _load_samples_and_mappings)
        # and all SampleClass objects are loaded (above),
        # now link them based on sample.class_id.
        for sample_obj in self.samples.values():
            if sample_obj.class_id and sample_obj.class_id in self.classes:
                self.classes[sample_obj.class_id].add_image(
                    sample_obj
                )  # add_image also updates sample.class_id

        logging.info(
            f"Successfully loaded and linked {len(self.classes)} classes with their samples."
        )

    def _load_consolidated_features(self):
        """Loads features from the consolidated features.npy file."""
        features_path = os.path.join(self.session.features_directory, FEATURES_FILENAME)
        if not os.path.exists(features_path):
            logging.warning(
                f"Consolidated features file not found: {features_path}. No features loaded."
            )
            self._all_features_array = None
            # Ensure all samples have None features if file is missing
            for sample in self.samples.values():
                sample.features = None
            return

        try:
            self._all_features_array = np.load(features_path)
            logging.info(
                f"Loaded consolidated features from {features_path}, shape: {self._all_features_array.shape}"
            )

            # Assign features to Sample objects
            for sample_id, sample in self.samples.items():
                if sample_id in self._sample_id_to_storage_index:
                    idx = self._sample_id_to_storage_index[sample_id]
                    if 0 <= idx < self._all_features_array.shape[0]:
                        sample.features = self._all_features_array[idx]
                    else:
                        logging.warning(
                            f"Storage index {idx} for sample {sample_id} out of bounds for features array (shape {self._all_features_array.shape})."
                        )
                        sample.features = None
                else:
                    sample.features = None  # No valid storage index for this sample
        except Exception as e:
            logging.error(
                f"Error loading consolidated features from {features_path}: {e}",
                exc_info=True,
            )
            self._all_features_array = None
            for sample in self.samples.values():
                sample.features = None

    def _load_consolidated_masks(self):
        """Loads masks from the consolidated masks.npy file."""
        masks_path = os.path.join(self.session.masks_directory, MASKS_FILENAME)
        if not os.path.exists(masks_path):
            logging.warning(
                f"Consolidated masks file not found: {masks_path}. No masks loaded into Mask objects."
            )
            self._all_masks_array = None
            for (
                mask
            ) in self.masks.values():  # Ensure existing Mask objects have no stale data
                mask.mask_data = None
            return

        try:
            # MODIFICATION: Load with allow_pickle=True for dtype=object array
            self._all_masks_array = np.load(masks_path, allow_pickle=True)
            logging.info(
                f"Loaded consolidated masks (dtype=object) from {masks_path}, shape: {self._all_masks_array.shape if self._all_masks_array is not None else 'None'}"
            )

            # REMOVED: Logic for setting _common_mask_height and _common_mask_width

            self._load_masks_metadata()  # Populates self.masks from DB

            # Assign mask_data to Mask objects
            for sample_id, sample in self.samples.items():
                if sample.mask_id and sample.mask_id in self.masks:
                    mask_obj = self.masks[sample.mask_id]
                    if (
                        sample.storage_index is not None
                        and self._all_masks_array is not None
                        and 0 <= sample.storage_index < len(self._all_masks_array)
                    ):

                        # MODIFICATION: Directly assign the object from the array
                        # It's expected to be a NumPy array (the actual mask) or None
                        loaded_mask_data = self._all_masks_array[sample.storage_index]
                        if (
                            isinstance(loaded_mask_data, np.ndarray)
                            or loaded_mask_data is None
                        ):
                            mask_obj.mask_data = loaded_mask_data
                        else:
                            logging.warning(
                                f"Unexpected data type in _all_masks_array at index {sample.storage_index} for sample {sample_id}. Expected ndarray or None, got {type(loaded_mask_data)}. Setting mask_data to None."
                            )
                            mask_obj.mask_data = None
                    else:
                        logging.warning(
                            f"Storage index {sample.storage_index} for sample {sample_id} (mask {sample.mask_id}) out of bounds or _all_masks_array is None."
                        )
                        mask_obj.mask_data = None
        except Exception as e:
            logging.error(
                f"Error loading consolidated masks from {masks_path}: {e}",
                exc_info=True,
            )
            self._all_masks_array = None
            for mask in self.masks.values():
                mask.mask_data = None

    @Slot(
        str, object, object, str
    )  # image_id, raw_mask_numpy_array, attributes_dict, masked_preview_path
    def _on_process_new_mask_requested(
        self,
        image_id: str,
        raw_mask_np: Any,  # Will be np.ndarray
        attributes: Dict[str, Any],
        masked_preview_path: str,
    ):
        """
        Slot to handle the request_process_new_mask signal from DataManagerInterface.
        This is called when SegmentationPresenter has new raw mask data to be processed and stored.
        """
        logging.debug(
            f"DataManager._on_process_new_mask_requested for image_id: {image_id}"
        )

        if not isinstance(raw_mask_np, np.ndarray):
            logging.error(
                f"Invalid mask data type received for {image_id}: {type(raw_mask_np)}. Expected np.ndarray."
            )
            return
        if not isinstance(attributes, dict):
            logging.error(
                f"Invalid attributes type received for {image_id}: {type(attributes)}. Expected dict."
            )
            # Potentially proceed with empty attributes or default
            attributes = {}

        try:
            mask_obj = self.add_raw_mask_data_for_sample(
                image_id=image_id,
                mask_array=raw_mask_np,
                attributes=attributes,
                masked_image_preview_path=masked_preview_path,
            )

            if mask_obj:
                logging.debug(
                    f"Successfully processed and added new mask data for {image_id}, mask_id: {mask_obj.id}"
                )
            else:
                logging.error(
                    f"Failed to process and add new mask data for {image_id} in add_raw_mask_data_for_sample."
                )
        except Exception as e:
            logging.error(
                f"Exception in _on_process_new_mask_requested for {image_id}: {e}",
                exc_info=True,
            )

    def _load_masks_metadata(self):
        """Loads Mask object metadata from the database into self.masks."""
        self.masks.clear()
        # This assumes DB stores mask_id, image_id, masked_image_preview_path, and attributes
        for sample in self.samples.values():
            if sample.mask_id:
                mask_db_data = self.db_manager.get_mask(sample.mask_id)
                if mask_db_data:
                    mask_obj = Mask.from_dict(mask_db_data)
                    mask_obj.attributes = self.db_manager.get_mask_attributes(
                        mask_obj.id
                    )
                    self.masks[mask_obj.id] = mask_obj
                else:
                    logging.warning(
                        f"Sample {sample.id} has mask_id {sample.mask_id} but no corresponding entry in masks table."
                    )
        logging.info(f"Loaded metadata for {len(self.masks)} masks from DB.")

    def _persist_consolidated_data(self, final_callback: Optional[callable] = None):
        """Saves dirty feature or mask arrays to their .npy files."""
        # This method should be called after DB operations are flushed, so storage_indices in DB are up-to-date.

        if self._is_features_dirty and self._all_features_array is not None:
            features_path = os.path.join(
                self.session.features_directory, FEATURES_FILENAME
            )
            try:
                np.save(features_path, self._all_features_array)
                self._is_features_dirty = False
                logging.info(f"Consolidated features saved to {features_path}")
            except Exception as e:
                logging.error(
                    f"Failed to save consolidated features: {e}", exc_info=True
                )

        if self._is_masks_dirty and self._all_masks_array is not None:
            masks_path = os.path.join(self.session.masks_directory, MASKS_FILENAME)
            try:
                np.save(masks_path, self._all_masks_array, allow_pickle=True)
                self._is_masks_dirty = False
                logging.info(f"Consolidated masks (dtype=object) saved to {masks_path}")
            except Exception as e:
                logging.error(
                    f"Failed to save consolidated masks (dtype=object): {e}",
                    exc_info=True,
                )

        if final_callback:
            final_callback()

    def _flush_pending_db_operations(self, callback: Optional[callable] = None) -> None:
        """Executes all pending DB operations in one transaction."""
        if not self.pending_db_operations:
            if callback:
                (
                    QTimer.singleShot(0, callback)
                    if isinstance(self.thread(), QThread)
                    and self.thread() != QThread.currentThread()
                    else callback()
                )
            return

        # Current implementation uses direct execution in DataManager's thread
        conn = self.db_manager.conn
        num_ops = len(self.pending_db_operations)
        pending_ops_copy = list(self.pending_db_operations)  # Work on a copy
        self.pending_db_operations.clear()  # Clear original list immediately

        try:
            conn.execute(
                "BEGIN IMMEDIATE TRANSACTION"
            )  # Use IMMEDIATE for exclusive access during commit
            for query, params in pending_ops_copy:
                conn.execute(query, params)
            conn.commit()
            logging.info(f"Successfully flushed {num_ops} DB operations.")
        except sqlite3.Error as e:  # More specific exception
            logging.error(
                f"SQLite error flushing {num_ops} DB operations: {e}. Query that might have failed (last one): {pending_ops_copy[-1] if pending_ops_copy else 'N/A'}",
                exc_info=True,
            )
            try:
                conn.rollback()
                logging.info("DB transaction rolled back.")
            except Exception as e_rb:
                logging.error(f"Critical error during rollback: {e_rb}", exc_info=True)
        except Exception as e_generic:
            logging.error(
                f"Generic error flushing {num_ops} DB operations: {e_generic}",
                exc_info=True,
            )
            try:
                conn.rollback()
                logging.info("DB transaction rolled back due to generic error.")
            except:
                pass  # Ignore rollback error if main error was not DB related
        finally:
            if callback:
                (
                    QTimer.singleShot(0, callback)
                    if isinstance(self.thread(), QThread)
                    and self.thread() != QThread.currentThread()
                    else callback()
                )

    #  Feature Management
    def _extract_all_features_internal(self):
        """
        Extracts features for all active samples.
        If the current processor/model is different from what features were last extracted with,
        or if features are missing, it re-extracts for all.
        """
        if not self.processor:
            logging.error(
                "DataManager: Cannot extract features - Processor not initialized or failed to update."
            )
            self.features_extracted_all.emit()
            return
        if not self.samples:
            logging.info("DataManager: No samples loaded to extract features for.")
            self.features_extracted_all.emit()
            return

        active_samples_for_extraction: List[Sample] = [
            s for s in self.samples.values() if s.is_active
        ]
        if not active_samples_for_extraction:
            logging.info("DataManager: No active samples for feature extraction.")
            self.feature_extraction_progress.emit(100)
            self.features_extracted_all.emit()
            return

        #  Determine if full re-extraction is needed
        needs_full_reextraction = False
        if self._all_features_array is None:
            needs_full_reextraction = True
            logging.info(
                "DataManager: No existing feature array. Full feature extraction required."
            )
        elif self._current_features_model_name != self.processor.model_name:
            needs_full_reextraction = True
            logging.info(
                f"DataManager: DL Model mismatch. Current features for '{self._current_features_model_name}', "
                f"processor is '{self.processor.model_name}'. Full re-extraction required."
            )
        elif self._current_features_dimension != self.processor.feature_dim:
            needs_full_reextraction = True
            logging.info(
                f"DataManager: Feature dimension mismatch. Current features dim {self._current_features_dimension}, "
                f"processor expects {self.processor.feature_dim}. Full re-extraction required."
            )

        if needs_full_reextraction:
            logging.info(
                f"DataManager: Performing full feature re-extraction for {len(active_samples_for_extraction)} samples using model '{self.processor.model_name}'."
            )
            # Clear old feature-related states
            self._all_features_array = None  # Will be re-initialized
            self._current_features_model_name = None
            self._current_features_dimension = None
            for sample in self.samples.values():
                sample.features = None
        else:
            logging.info(
                "DataManager: Feature array and model appear consistent. No full re-extraction triggered by model change."
            )
            self.feature_extraction_progress.emit(100)
            self.features_extracted_all.emit()
            return

        #  1. Prepare list of samples and ensure storage_index is assigned (or re-assigned if full re-extraction)
        active_samples_for_extraction.sort(
            key=lambda s: (s.storage_index is None, s.storage_index, s.path)
        )

        self._sample_id_to_storage_index.clear()
        self._storage_index_to_sample_id.clear()
        id_index_map_for_db_update = {}
        db_updates_pending = False

        for idx, sample in enumerate(active_samples_for_extraction):
            if (
                sample.storage_index != idx or needs_full_reextraction
            ):  # Force update if full re-extraction
                db_updates_pending = True
            sample.storage_index = idx
            self._sample_id_to_storage_index[sample.id] = idx
            self._storage_index_to_sample_id[idx] = sample.id
            id_index_map_for_db_update[sample.id] = idx

        if db_updates_pending:
            self.db_manager.update_samples_storage_indices(id_index_map_for_db_update)
            logging.info(
                f"DataManager: (Re)Assigned and saved storage_indices for {len(id_index_map_for_db_update)} samples."
            )

        num_total_samples_to_process = len(active_samples_for_extraction)
        if not self.processor.feature_dim:
            logging.error(
                "DataManager: Feature dimension unknown. Cannot initialize features array."
            )
            self.features_extracted_all.emit()
            return

        self._all_features_array = np.zeros(
            (num_total_samples_to_process, self.processor.feature_dim), dtype=np.float32
        )
        logging.info(
            f"DataManager: Initialized _all_features_array to shape: {self._all_features_array.shape}"
        )

        extracted_successfully_count = 0
        processed_sample_count = 0
        for sample in active_samples_for_extraction:
            current_storage_idx = sample.storage_index
            try:
                feature_vector = self.processor.extract_features(sample.path)
                if feature_vector is not None:
                    if feature_vector.shape[0] != self.processor.feature_dim:
                        logging.error(
                            f"DataManager: Extracted feature dim mismatch for {sample.id}. Expected {self.processor.feature_dim}, got {feature_vector.shape[0]}. Assigning zeros."
                        )
                        self._all_features_array[current_storage_idx] = 0
                    else:
                        self.samples[sample.id].features = feature_vector
                        self._all_features_array[current_storage_idx] = feature_vector
                        extracted_successfully_count += 1
                else:
                    logging.warning(
                        f"DataManager: Feature extraction failed for sample {sample.id}. Assigning zeros."
                    )
                    self._all_features_array[current_storage_idx] = 0
            except Exception as e:
                logging.error(
                    f"DataManager: Error extracting features for {sample.id}: {e}",
                    exc_info=True,
                )
                if 0 <= current_storage_idx < self._all_features_array.shape[0]:
                    self._all_features_array[current_storage_idx] = 0
            processed_sample_count += 1
            progress_percentage = int(
                (processed_sample_count / num_total_samples_to_process) * 100
            )
            self.feature_extraction_progress.emit(progress_percentage)

        logging.info(
            f"DataManager: Feature extraction loop completed. Successfully extracted: {extracted_successfully_count}/{num_total_samples_to_process}."
        )

        if extracted_successfully_count > 0 or db_updates_pending:
            self._is_features_dirty = True
            self._current_features_model_name = (
                self.processor.model_name
            )  # Update tracker
            self._current_features_dimension = (
                self.processor.feature_dim
            )  # Update tracker
            if (
                processed_sample_count == num_total_samples_to_process
                and progress_percentage < 100
            ):
                self.feature_extraction_progress.emit(100)
            self._persist_consolidated_data(
                final_callback=lambda: self.features_extracted_all.emit()
            )
        else:
            logging.info("DataManager: No new features extracted (or indices changed).")
            if num_total_samples_to_process > 0 and progress_percentage < 100:
                self.feature_extraction_progress.emit(100)
            self.features_extracted_all.emit()

    @Slot()
    def clear_features_and_clusters(self):
        """Clears features (disk and memory) and all clusters."""
        logging.info("Clearing all features and clusters...")

        # Clear Features from memory
        if self._all_features_array is not None:
            self._all_features_array = None
        for sample in self.samples.values():
            sample.features = None
            sample.storage_index = None

        # Delete features.npy file
        features_path = os.path.join(self.session.features_directory, FEATURES_FILENAME)
        if os.path.exists(features_path):
            try:
                os.remove(features_path)
                logging.info(f"Deleted {features_path}")
            except OSError as e:
                logging.error(f"Error deleting feature file {features_path}: {e}")
        self._is_features_dirty = False

        all_sample_ids = list(self.samples.keys())
        if all_sample_ids:
            # This is a batch update, might be slow for many samples.
            self._sample_id_to_storage_index.clear()
            self._storage_index_to_sample_id.clear()

        self.features_invalidated.emit()
        logging.info("In-memory features and feature file cleared.")

        self.clear_clusters()
        logging.info("Features and clusters cleared.")

    @Slot(dict)
    def update_processor(self, new_settings: dict):
        """
        Updates the processor instance based on new settings.
        If the DL model name changes, it emits features_invalidated.
        It does NOT clear features directly anymore.
        """
        if not self.processor:  # First time initialization or previous failure
            old_model_name = None
        else:
            old_model_name = self.processor.model_name

        new_model_name = new_settings.get("selected_model")
        new_provider = new_settings.get("provider")

        if not new_model_name or not new_provider:
            logging.error(
                "DataManager: Cannot update processor - Invalid settings (model or provider missing)."
            )
            return

        # Update internal settings reference (used by various parts of DM)
        self.settings["selected_model"] = new_model_name
        self.settings["provider"] = new_provider
        # Note: self.available_models is usually loaded in __init__. If it can change, reload here.
        # self.available_models = get_available_models()

        if (
            self.processor
            and new_model_name == old_model_name
            and new_provider == self.processor.execution_provider
        ):
            logging.info("DataManager: Processor settings unchanged. No update needed.")
            return

        logging.info(
            f"DataManager: Updating processor. Old model: {old_model_name}, New model: {new_model_name}, Provider: {new_provider}"
        )
        try:
            self.processor = Processor(
                model_name=new_model_name,
                available_models=self.available_models,  # Assuming this is up-to-date
                execution_provider=new_provider,
            )
            logging.info(
                f"DataManager: Processor updated successfully to model '{new_model_name}'."
            )

            if old_model_name != new_model_name and old_model_name is not None:
                logging.warning(
                    f"DataManager: DL Model changed from '{old_model_name}' to '{new_model_name}'. "
                    f"Emitting features_invalidated. Features will be re-extracted on next clustering request if needed."
                )
                # Don't clear features here. Just notify.
                # The _extract_all_features_internal method will check consistency.
                self.features_invalidated.emit()
                # Clusters are implicitly invalid if features are.
                self.clusters_invalidated.emit()  # Presenter will clear UI

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logging.error(
                f"DataManager: Failed to update Processor to {new_model_name}: {e}. Feature extraction might fail.",
                exc_info=True,
            )
            self.processor = None  # Set to None to indicate an unusable state

    #  Mask Management
    # This is a simplified `create_mask` that only handles the metadata part.
    # The actual mask data (NumPy array) should be added to `_all_masks_array`
    # and then persisted via `_persist_consolidated_data`.
    # The `SegmentationThread` will need to emit the raw mask data.

    def add_raw_mask_data_for_sample(
        self,
        image_id: str,
        mask_array: np.ndarray,  # This is now variable-sized
        attributes: Dict[str, Any],
        masked_image_preview_path: Optional[str],
    ) -> Optional[Mask]:
        """
        Receives raw mask data for a sample, updates/creates Mask object and its metadata in DB (queued),
        adds pixel data (as object) to _all_masks_array, and flags for .npy persistence.
        """
        if image_id not in self.samples:
            logging.error(
                f"DataManager: Cannot add mask data for non-existent sample ID: {image_id}"
            )
            return None
        sample = self.samples[image_id]
        if sample.storage_index is None:
            logging.error(
                f"DataManager: Sample {image_id} has no storage_index. Cannot assign mask data."
            )
            return None
        if (
            mask_array is None
            or not isinstance(mask_array, np.ndarray)
            or mask_array.ndim < 2
        ):
            logging.error(
                f"DataManager: Invalid or empty mask_array received for {image_id}. Cannot process mask."
            )
            return None

        num_total_samples_with_indices = len(self._storage_index_to_sample_id)
        # Fallback if map is empty but we have a valid index (e.g., first item)
        if num_total_samples_with_indices == 0 and sample.storage_index is not None:
            num_total_samples_with_indices = sample.storage_index + 1

        if (
            self._all_masks_array is None
            or self._all_masks_array.shape[0] < num_total_samples_with_indices
        ):
            new_size = num_total_samples_with_indices
            logging.info(
                f"DataManager: Initializing/Resizing _all_masks_array (dtype=object). Current size: {len(self._all_masks_array) if self._all_masks_array is not None else 'None'}, New size: {new_size}"
            )

            new_array = np.empty(new_size, dtype=object)
            if self._all_masks_array is not None:
                current_len = len(self._all_masks_array)
                new_array[:current_len] = self._all_masks_array  # Copy old data
            self._all_masks_array = new_array
            self._is_masks_dirty = True

        storage_idx = sample.storage_index
        if self._all_masks_array is not None and 0 <= storage_idx < len(
            self._all_masks_array
        ):
            self._all_masks_array[storage_idx] = mask_array
            self._is_masks_dirty = True
        else:
            logging.error(
                f"DataManager: Invalid storage index {storage_idx} or _all_masks_array not properly initialized for sample {image_id}."
            )
            return None

        mask_obj: Optional[Mask] = None
        if sample.mask_id and sample.mask_id in self.masks:  # Update existing mask
            mask_obj = self.masks[sample.mask_id]
            mask_obj.attributes = attributes
            mask_obj.masked_image_path = masked_image_preview_path
            mask_obj.mask_data = mask_array  # Update in-memory Mask object's data
        else:  # Create new mask
            new_mask_id = str(uuid.uuid4())
            mask_obj = Mask(
                id=new_mask_id,
                image_id=image_id,
                attributes=attributes,
                masked_image_path=masked_image_preview_path,
                mask_data=mask_array,
            )
            self.masks[new_mask_id] = mask_obj
            old_mask_id_if_replacing = sample.mask_id
            sample.mask_id = new_mask_id
            self.add_pending_db_operation(  # Insert into masks table
                "INSERT INTO masks (id, image_id, masked_image_path) VALUES (?, ?, ?)",
                (new_mask_id, image_id, masked_image_preview_path),
            )
            self.add_pending_db_operation(  # Update samples.mask_id
                "UPDATE samples SET mask_id = ? WHERE id = ?", (new_mask_id, image_id)
            )
            if old_mask_id_if_replacing:
                self.add_pending_db_operation(
                    "DELETE FROM mask_attributes WHERE mask_id = ?",
                    (old_mask_id_if_replacing,),
                )
                self.add_pending_db_operation(
                    "DELETE FROM masks WHERE id = ?", (old_mask_id_if_replacing,)
                )
            for name_attr, value_attr in attributes.items():
                value_str = str(value_attr)
                self.add_pending_db_operation(
                    "INSERT INTO mask_attributes (mask_id, attribute_name, attribute_value) VALUES (?, ?, ?)",
                    (new_mask_id, name_attr, value_str),
                )

        if mask_obj:
            self.mask_created_in_db.emit(mask_obj)

        return mask_obj

    def delete_mask(self, mask_id_or_image_id: str, by_image_id=False):
        """
        Deletes a mask. If by_image_id, finds mask_id first.
        This means nullifying its entry in DB and self.masks.
        The data in _all_masks_array becomes orphaned until next compaction/rewrite.
        """
        mask_to_delete_id = None
        image_id_of_mask = None

        if by_image_id:
            image_id_of_mask = mask_id_or_image_id
            if image_id_of_mask in self.samples:
                mask_to_delete_id = self.samples[image_id_of_mask].mask_id
            else:
                logging.warning(
                    f"Cannot delete mask by image_id: Sample {image_id_of_mask} not found."
                )
                return
        else:
            mask_to_delete_id = mask_id_or_image_id
            mask_obj = self.masks.get(mask_to_delete_id)
            if mask_obj:
                image_id_of_mask = mask_obj.image_id
            else:
                logging.warning(f"Mask ID {mask_to_delete_id} not found in self.masks.")
                # Try to find image_id from DB if mask object not in memory
                db_mask_info = self.db_manager.get_mask(mask_to_delete_id)
                if db_mask_info:
                    image_id_of_mask = db_mask_info.get("image_id")

        if not mask_to_delete_id:
            logging.warning(
                f"No mask found to delete for input: {mask_id_or_image_id} (by_image_id={by_image_id})"
            )
            return

        self.db_manager.delete_mask(mask_to_delete_id)

        if mask_to_delete_id in self.masks:
            del self.masks[mask_to_delete_id]

        if image_id_of_mask and image_id_of_mask in self.samples:
            self.samples[image_id_of_mask].mask_id = None
            if (
                self.samples[image_id_of_mask].storage_index is not None
                and self._all_masks_array is not None
            ):
                idx = self.samples[image_id_of_mask].storage_index
                if 0 <= idx < self._all_masks_array.shape[0]:
                    self._all_masks_array[idx] = None
                    self._is_masks_dirty = True
                    self._persist_consolidated_data()

        logging.info(
            f"Mask {mask_to_delete_id} (for image {image_id_of_mask or 'unknown'}) deleted from DB and memory."
        )
        self.mask_deleted_from_db.emit(mask_to_delete_id)

    def get_mask(self, mask_id: str) -> Optional[Mask]:
        return self.masks.get(mask_id)

    # Gates and plots

    @Slot(
        str, str, str, object
    )  # plot_id, name, chart_type_key, parameters_obj (now a dict)
    def _on_create_plot_requested(
        self,
        plot_id: str,
        name: str,
        chart_type_key: str,
        parameters_dict_received: object,
    ):
        # parameters_dict_received is now always expected to be a dict
        logging.debug(
            f"DataManager: Plot creation requested - ID: {plot_id}, Name: {name}, Type: {chart_type_key}, Params Type: {type(parameters_dict_received)}"
        )
        try:
            if not isinstance(parameters_dict_received, dict):
                logging.error(
                    f"DataManager: Expected a dictionary for parameters_obj, got {type(parameters_dict_received)}. Plot {plot_id} cannot be created."
                )
                return

            params_dict_for_json = parameters_dict_received
            parameters_json = json.dumps(params_dict_for_json)
            created_at_iso = datetime.now().isoformat()
            self.db_manager.create_plot(
                plot_id, name, chart_type_key, parameters_json, created_at_iso
            )

            parameters_obj_instance = None
            if chart_type_key == "histogram":
                try:
                    parameters_obj_instance = HistogramParameters(
                        **params_dict_for_json
                    )
                except TypeError as te:
                    logging.error(
                        f"DM: Mismatch creating HistogramParameters for plot {plot_id}: {te}. Params: {params_dict_for_json}"
                    )
                    return
            elif chart_type_key == "scatter":
                try:
                    parameters_obj_instance = ScatterParameters(**params_dict_for_json)
                except TypeError as te:
                    logging.error(
                        f"DM: Mismatch creating ScatterParameters for plot {plot_id}: {te}. Params: {params_dict_for_json}"
                    )
                    return
            else:
                logging.error(
                    f"DM: Unknown chart_type '{chart_type_key}' for plot {plot_id}. Cannot create in-memory object."
                )
                return

            plot_config_for_memory = {
                "id": plot_id,
                "name": name,
                "chart_type": chart_type_key,
                "parameters_obj": parameters_obj_instance,  # Store the RECONSTRUCTED dataclass object
                "created_at": created_at_iso,
            }
            self.plot_configurations[plot_id] = plot_config_for_memory
            # The plot_added_to_dm signal will emit this dict, which contains the dataclass instance.
            # AnalysisPresenter.on_plot_added_to_dm receives this and passes it to AnalysisViewWidget.
            self.plot_added_to_dm.emit(plot_id, plot_config_for_memory)
            logging.info(f"Plot {plot_id} ({name}) created and added to DataManager.")

        except Exception as e:
            logging.error(
                f"Error creating plot {plot_id} in DataManager: {e}", exc_info=True
            )

    @Slot(str)  # plot_id
    def _on_delete_plot_requested(self, plot_id: str):
        logging.debug(f"DataManager: Plot deletion requested - ID: {plot_id}")

        associated_gate_ids_in_memory = [
            gid for gid, gate_obj in self.gates.items() if gate_obj.plot_id == plot_id
        ]

        try:
            self.db_manager.delete_plot(plot_id)  # DB cascade will handle gates in DB

            for gid in associated_gate_ids_in_memory:
                if gid in self.gates:  # Should always be true
                    del self.gates[gid]
                    logging.debug(
                        f"Removed gate {gid} from memory due to plot {plot_id} deletion."
                    )

            if plot_id in self.plot_configurations:
                del self.plot_configurations[plot_id]

            self.plot_deleted_from_dm.emit(plot_id)
            self.update_active_samples()
            logging.info(
                f"Plot {plot_id} and its associated gates deleted from DataManager and DB."
            )
        except Exception as e:
            logging.error(
                f"Error deleting plot {plot_id} from DB, in-memory state preserved: {e}",
                exc_info=True,
            )

    @Slot(str, object)  # plot_id, gate_obj (BaseGate derivative instance)
    def _on_create_gate_requested(
        self, plot_id: str, gate_obj: Any
    ):  # gate_obj is BaseGate
        logging.debug(
            f"DataManager: Gate creation requested for plot {plot_id}, Gate ID: {gate_obj.id}, Name: {gate_obj.name}"
        )
        try:
            if (
                not hasattr(gate_obj, "id")
                or not hasattr(gate_obj, "name")
                or not hasattr(gate_obj, "parameters")
                or not hasattr(gate_obj, "color")
            ):
                logging.error(
                    f"Received invalid gate_obj for creation on plot {plot_id}."
                )
                return

            gate_obj.plot_id = plot_id  # Ensure plot_id is set on the gate object

            # Serialize definition and parameters for DB
            definition_json = ""
            if (
                hasattr(gate_obj, "rect") and gate_obj.rect is not None
            ):  # RectangularGate
                r = gate_obj.rect
                definition_json = json.dumps(
                    {"x": r.x(), "y": r.y(), "width": r.width(), "height": r.height()}
                )
                gate_type = "rectangular"
            elif (
                hasattr(gate_obj, "polygon") and gate_obj.polygon is not None
            ):  # PolygonGate
                vertices = [(p.x(), p.y()) for p in gate_obj.polygon]
                definition_json = json.dumps({"vertices": vertices})
                gate_type = "polygon"
            elif hasattr(gate_obj, "min_val") and hasattr(
                gate_obj, "max_val"
            ):  # IntervalGate
                definition_json = json.dumps(
                    {"min_val": gate_obj.min_val, "max_val": gate_obj.max_val}
                )
                gate_type = "interval"
            else:
                logging.error(
                    f"Unknown gate object type for serialization: {type(gate_obj)}"
                )
                return

            parameters_tuple_json = json.dumps(
                gate_obj.parameters
            )  # Should be a tuple of strings
            color_hex = (
                gate_obj.color.name() if hasattr(gate_obj.color, "name") else "#FFFFFF"
            )
            gate_id = str(gate_obj.id)
            self.db_manager.create_gate(
                gate_id,
                plot_id,
                gate_obj.name,
                gate_type,
                definition_json,
                color_hex,
                parameters_tuple_json,
            )

            self.gates[str(gate_obj.id)] = gate_obj  # Store the live object
            self.active_global_gate_ids.add(gate_id)  # Add to active gates set
            self.recalculate_gate_population(gate_obj)  # Calculate initial population
            self.update_active_samples()
            self.gate_added_to_dm.emit(plot_id, gate_obj)

            logging.info(
                f"Gate {gate_obj.id} ({gate_obj.name}) created for plot {plot_id} and added to DataManager."
            )

        except Exception as e:
            logging.error(
                f"Error creating gate {getattr(gate_obj, 'id', 'N/A')} for plot {plot_id}: {e}",
                exc_info=True,
            )

    @Slot(object)  # gate_obj (updated BaseGate instance)
    def _on_update_gate_requested(self, gate_obj: Any):
        gate_id_str = str(gate_obj.id)
        logging.debug(f"DataManager: Gate update requested - ID: {gate_id_str}")
        try:
            if gate_id_str not in self.gates:
                logging.error(
                    f"Cannot update gate {gate_id_str}: Not found in DataManager."
                )
                return

            # Serialize definition and parameters for DB
            definition_json, parameters_tuple_json = None, None
            if hasattr(gate_obj, "rect") and gate_obj.rect is not None:
                r = gate_obj.rect
                definition_json = json.dumps(
                    {"x": r.x(), "y": r.y(), "width": r.width(), "height": r.height()}
                )

            if (
                gate_obj.parameters
            ):  # If parameters can change (unlikely for a defined gate)
                parameters_tuple_json = json.dumps(gate_obj.parameters)

            color_hex = (
                gate_obj.color.name() if hasattr(gate_obj.color, "name") else None
            )

            self.db_manager.update_gate(
                gate_id_str,
                name=gate_obj.name,
                definition_json=definition_json,
                color=color_hex,
                parameters_tuple_json=parameters_tuple_json,
            )

            self.gates[gate_id_str] = gate_obj  # Update in-memory object
            self.recalculate_gate_population(
                gate_obj
            )  # Population might change if definition changed

            self.gate_updated_in_dm.emit(gate_obj.plot_id, gate_obj)
            logging.info(f"Gate {gate_id_str} updated in DataManager.")
        except Exception as e:
            logging.error(f"Error updating gate {gate_id_str}: {e}", exc_info=True)

    @Slot(str)  # gate_id (string)
    def _on_delete_gate_requested(self, gate_id_str: str):
        logging.debug(f"DataManager: Gate deletion requested - ID: {gate_id_str}")

        gate_to_delete_obj = self.gates.get(gate_id_str)  # Get from memory
        plot_id_of_gate = None
        if gate_to_delete_obj:
            plot_id_of_gate = gate_to_delete_obj.plot_id
        else:
            db_gate_info = self.db_manager.get_gate(gate_id_str)
            if db_gate_info:
                plot_id_of_gate = db_gate_info.get("plot_id")
            logging.warning(
                f"DataManager: Gate {gate_id_str} not found in self.gates for deletion. Will attempt DB delete."
            )

        try:
            if gate_id_str in self.active_global_gate_ids:
                self.active_global_gate_ids.remove(gate_id_str)
                logging.debug(
                    f"DataManager: Removed gate {gate_id_str} from active_global_gate_ids."
                )

            self.update_active_samples()

            self.db_manager.delete_gate(gate_id_str)

            if gate_id_str in self.gates:
                del self.gates[gate_id_str]

            logging.info(f"Gate {gate_id_str} deleted from DataManager and DB.")
            if plot_id_of_gate:
                self.gate_deleted_from_dm.emit(plot_id_of_gate, gate_id_str)
            else:
                logging.warning(
                    f"Could not determine plot_id for deleted gate {gate_id_str}, signal not emitted to presenter."
                )

        except Exception as e:
            logging.error(
                f"Error processing gate deletion for {gate_id_str}: {e}", exc_info=True
            )

    @Slot()
    def load_images_from_folder(self, folder_path: str) -> None:
        """
        Loads images from a folder, creates Sample objects, assigns new storage_indices,
        updates DB, and assigns all new samples to the 'Uncategorized' class.
        """
        if not os.path.isdir(folder_path):
            logging.error(f"Invalid folder path: {folder_path}")
            return

        supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

        # Ensure "Uncategorized" class exists
        uncategorized_class = self._create_default_class()
        if not uncategorized_class:
            logging.error(
                f"Cannot load images from folder: Failed to ensure 'Uncategorized' class exists."
            )
            return

        new_db_samples_data = (
            []
        )  # Tuples for db_manager.create_samples: (id, path, storage_index, class_id, mask_id)
        new_sample_objects_list = []  # List of new Sample objects for in-memory updates

        # Determine next available storage_index
        current_max_idx = -1
        if self._storage_index_to_sample_id:  # Check in-memory map first
            current_max_idx = (
                max(self._storage_index_to_sample_id.keys())
                if self._storage_index_to_sample_id
                else -1
            )
        else:  # If map is empty (e.g., first load or after clearing), check DB
            current_max_idx = self.db_manager.get_highest_storage_index()
        next_idx = current_max_idx + 1

        existing_paths_in_session = {s.path for s in self.samples.values()}

        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                image_path = os.path.join(folder_path, filename)

                if image_path in existing_paths_in_session:
                    logging.info(
                        f"Image path {image_path} already exists in session. Skipping."
                    )
                    continue

                image_id = str(uuid.uuid4())

                # Create Sample object, immediately linking to Uncategorized class ID
                sample = Sample(
                    id=image_id,
                    path=image_path,
                    storage_index=next_idx,
                    class_id=uncategorized_class.id,
                )

                # Add to in-memory structures
                self.samples[image_id] = sample
                self._sample_id_to_storage_index[image_id] = next_idx
                self._storage_index_to_sample_id[next_idx] = image_id
                new_sample_objects_list.append(sample)

                # Prepare data for DB batch insert (samples table)
                new_db_samples_data.append(
                    (image_id, image_path, next_idx, uncategorized_class.id, None)
                )

                next_idx += 1
                existing_paths_in_session.add(
                    image_path
                )  # Add to set to catch duplicates within the same folder scan

        if new_db_samples_data:
            self.db_manager.create_samples(new_db_samples_data)
            logging.info(f"Added {len(new_db_samples_data)} new sample records to DB.")

            new_image_ids = [s.id for s in new_sample_objects_list]
            self.assign_images_to_class(new_image_ids, uncategorized_class.id)
            num_total_samples_with_indices = len(self._storage_index_to_sample_id)

            if (
                self._all_features_array is not None
                and self._all_features_array.shape[0] < num_total_samples_with_indices
            ):
                if self.processor and self.processor.feature_dim:
                    old_array = self._all_features_array
                    self._all_features_array = np.zeros(
                        (num_total_samples_with_indices, self.processor.feature_dim),
                        dtype=np.float32,
                    )
                    self._all_features_array[: old_array.shape[0]] = (
                        old_array  # Copy old data
                    )
                    self._is_features_dirty = True
                    logging.info(
                        f"Resized _all_features_array to shape: {self._all_features_array.shape} due to new images."
                    )
                else:
                    logging.warning(
                        "Cannot resize features array: processor or feature_dim not available."
                    )

            if (
                self._all_masks_array is not None
                and len(self._all_masks_array) < num_total_samples_with_indices
            ):
                old_array_mask = self._all_masks_array
                self._all_masks_array = np.empty(
                    num_total_samples_with_indices, dtype=object
                )
                self._all_masks_array[: len(old_array_mask)] = (
                    old_array_mask  # Copy old object references
                )
                self._is_masks_dirty = True
                logging.info(
                    f"Resized _all_masks_array (dtype=object) to size: {len(self._all_masks_array)} due to new images."
                )
            elif (
                self._all_masks_array is None and new_db_samples_data
            ):  # If array didn't exist but new samples were added
                self._all_masks_array = np.empty(
                    num_total_samples_with_indices, dtype=object
                )
                logging.info(
                    f"Initialized _all_masks_array (dtype=object) to size: {len(self._all_masks_array)} for new images."
                )

            self.images_loaded.emit()
            self.samples_ready.emit(self.samples.copy())

            logging.info(
                f"Processed {len(new_db_samples_data)} new images from {folder_path}. Assigned to 'Uncategorized'."
            )

            if self._is_features_dirty or self._is_masks_dirty:
                QTimer.singleShot(
                    100,
                    lambda: self._persist_consolidated_data(
                        final_callback=lambda: logging.info(
                            "Consolidated .npy files persisted after new image import if resized."
                        )
                    ),
                )

        else:
            logging.info(f"No new valid images found in {folder_path} to add.")

    @Slot(str)  # Make it a slot if it can be called via interface directly
    def delete_cluster(self, cluster_id: str) -> None:
        """Deletes a cluster from memory and queues DB deletion."""
        cluster_to_delete = self.clusters.pop(cluster_id, None)
        if not cluster_to_delete:
            logging.warning(
                f"DataManager: Cluster ID {cluster_id} not found for deletion."
            )
            return

        # Remove cluster references from samples in memory
        for sample in list(cluster_to_delete.samples):  # Iterate over a copy
            sample.cluster_ids.discard(cluster_id)

        # Queue DB operations
        self.add_pending_db_operation(
            "DELETE FROM samples_clusters WHERE cluster_id = ?", (cluster_id,)
        )
        self.add_pending_db_operation(
            "DELETE FROM clusters WHERE id = ?", (cluster_id,)
        )

        self._flush_pending_db_operations(
            callback=lambda: self.clusters_invalidated.emit()
        )
        logging.info(
            f"DataManager: Cluster {cluster_id} deleted and DB operations queued."
        )

    @Slot(str, int, int, int)  # Connected to interface signal
    def split_cluster(
        self, cluster_id_to_split: str, n_sub_clusters: int, n_iter: int, n_redo: int
    ) -> None:
        """Splits a specified cluster into sub-clusters."""
        if not self.processor:
            logging.error(
                "DataManager: Cannot split cluster - Processor not initialized."
            )
            # Optionally emit a failure signal
            return

        original_cluster = self.get_cluster(cluster_id_to_split)
        if not original_cluster:
            logging.error(
                f"DataManager: Cluster {cluster_id_to_split} not found for splitting."
            )
            return

        active_samples_in_cluster = [
            s
            for s in original_cluster.samples
            if s.is_active and s.storage_index is not None
        ]
        if len(active_samples_in_cluster) < n_sub_clusters:
            logging.warning(
                f"DataManager: Not enough active samples ({len(active_samples_in_cluster)}) with valid storage_index in cluster {cluster_id_to_split} to split into {n_sub_clusters} sub-clusters."
            )
            return

        sample_indices_for_features = [
            s.storage_index for s in active_samples_in_cluster
        ]

        if self._all_features_array is None or not all(
            0 <= idx < self._all_features_array.shape[0]
            for idx in sample_indices_for_features
        ):
            logging.error(
                f"DataManager: Feature array not available or indices out of bounds for cluster {cluster_id_to_split} split."
            )
            return

        features_of_cluster_samples = self._all_features_array[
            sample_indices_for_features
        ]

        logging.info(
            f"DataManager: Splitting cluster {cluster_id_to_split} (samples: {len(features_of_cluster_samples)}) into {n_sub_clusters} sub-clusters."
        )
        try:
            # Processor.split_cluster expects features of only the samples to be split
            sub_cluster_labels = self.processor.split_cluster(
                features_of_cluster_samples, n_sub_clusters, n_iter, n_redo
            )
        except Exception as e:
            logging.error(
                f"DataManager: Error during K-means for splitting cluster {cluster_id_to_split}: {e}",
                exc_info=True,
            )
            return

        if not sub_cluster_labels or len(sub_cluster_labels) != len(
            active_samples_in_cluster
        ):
            logging.error(
                "DataManager: Splitting cluster did not return valid labels or label count mismatch."
            )
            return

        new_cluster_objects_map: Dict[int, Cluster] = {}
        new_cluster_ids_created: List[str] = []

        for i in range(n_sub_clusters):  # For each new sub-cluster label
            new_id = str(uuid.uuid4())
            new_color = self._generate_random_color()
            new_cluster_obj = Cluster(id=new_id, color=new_color)
            self.clusters[new_id] = new_cluster_obj
            new_cluster_objects_map[i] = new_cluster_obj
            new_cluster_ids_created.append(new_id)
            self.add_pending_db_operation(
                "INSERT INTO clusters (id, color) VALUES (?, ?)", (new_id, new_color)
            )

        self.add_pending_db_operation(
            "DELETE FROM samples_clusters WHERE cluster_id = ?", (cluster_id_to_split,)
        )

        for sample_idx, original_sample_obj in enumerate(active_samples_in_cluster):
            new_local_label = sub_cluster_labels[sample_idx]
            new_assigned_cluster_obj = new_cluster_objects_map[new_local_label]

            # Update sample's in-memory cluster_ids set
            original_sample_obj.cluster_ids.discard(cluster_id_to_split)
            original_sample_obj.cluster_ids.add(new_assigned_cluster_obj.id)

            # Update new cluster's in-memory sample set
            new_assigned_cluster_obj.add_image(
                original_sample_obj
            )  # add_image ensures reciprocal link

            # Queue DB insert for the new sample-cluster link
            self.add_pending_db_operation(
                "INSERT INTO samples_clusters (sample_id, cluster_id) VALUES (?, ?)",
                (original_sample_obj.id, new_assigned_cluster_obj.id),
            )

        if cluster_id_to_split in self.clusters:
            del self.clusters[cluster_id_to_split]
        self.add_pending_db_operation(
            "DELETE FROM clusters WHERE id = ?", (cluster_id_to_split,)
        )

        self._flush_pending_db_operations(
            callback=lambda: self.cluster_split.emit(
                cluster_id_to_split, new_cluster_ids_created
            )
        )
        logging.info(
            f"DataManager: Cluster {cluster_id_to_split} split into {new_cluster_ids_created}. DB ops queued."
        )

    @Slot(list)
    def merge_clusters(self, cluster_ids_to_merge: List[str]) -> None:
        """Merges multiple clusters into a new single cluster."""
        if len(cluster_ids_to_merge) < 2:
            logging.warning("DataManager: Need at least two clusters to merge.")
            return

        valid_old_clusters: List[Cluster] = []
        for cid in cluster_ids_to_merge:
            cluster = self.get_cluster(cid)
            if cluster:
                valid_old_clusters.append(cluster)
            else:
                logging.warning(
                    f"DataManager: Cluster ID {cid} not found during merge. Skipping it."
                )

        if len(valid_old_clusters) < 2:
            logging.warning(
                "DataManager: Not enough valid clusters found to perform merge."
            )
            return

        new_merged_cluster_id = str(uuid.uuid4())
        new_merged_cluster_color = self._generate_random_color()
        new_merged_cluster_obj = Cluster(
            id=new_merged_cluster_id, color=new_merged_cluster_color
        )
        self.clusters[new_merged_cluster_id] = new_merged_cluster_obj
        self.add_pending_db_operation(
            "INSERT INTO clusters (id, color) VALUES (?, ?)",
            (new_merged_cluster_id, new_merged_cluster_color),
        )

        all_samples_to_move: Dict[str, Sample] = {}
        for old_cluster in valid_old_clusters:
            for sample in list(old_cluster.samples):
                all_samples_to_move[sample.id] = sample
                sample.cluster_ids.discard(old_cluster.id)
                sample.cluster_ids.add(new_merged_cluster_id)
                new_merged_cluster_obj.add_image(sample)

        for old_cluster in valid_old_clusters:
            self.add_pending_db_operation(
                "DELETE FROM samples_clusters WHERE cluster_id = ?", (old_cluster.id,)
            )

        for sample_id in all_samples_to_move.keys():
            self.add_pending_db_operation(
                "INSERT OR IGNORE INTO samples_clusters (sample_id, cluster_id) VALUES (?, ?)",  # IGNORE if somehow already linked
                (sample_id, new_merged_cluster_id),
            )

        actual_old_cluster_ids_processed = []
        for old_cluster in valid_old_clusters:
            if old_cluster.id in self.clusters:
                del self.clusters[old_cluster.id]
            self.add_pending_db_operation(
                "DELETE FROM clusters WHERE id = ?", (old_cluster.id,)
            )
            actual_old_cluster_ids_processed.append(old_cluster.id)

        self._flush_pending_db_operations(
            callback=lambda: self.cluster_merged.emit(
                new_merged_cluster_id, actual_old_cluster_ids_processed
            )
        )
        logging.info(
            f"DataManager: Merged clusters {actual_old_cluster_ids_processed} into new cluster {new_merged_cluster_id}. DB ops queued."
        )

    def _get_active_dl_features_array(self) -> Optional[np.ndarray]:
        if self._all_features_array is None:
            logging.debug("DataManager: No _all_features_array for DL features.")
            self.active_sample_ids_for_current_clustering = []
            return None

        # Use self.active_samples (which are already globally filtered and Sample.is_active)
        if not self.active_samples:
            logging.debug("DataManager: No active_samples for DL features.")
            self.active_sample_ids_for_current_clustering = []
            return None

        temp_samples_info = []  # (storage_idx, sample_id)
        for (
            s_id,
            sample_obj,
        ) in self.active_samples.items():  # Iterate over current active_samples
            if sample_obj.storage_index is not None:
                if 0 <= sample_obj.storage_index < self._all_features_array.shape[0]:
                    temp_samples_info.append((sample_obj.storage_index, s_id))
                else:
                    logging.warning(
                        f"DM: Active sample {s_id} (idx {sample_obj.storage_index}) out of bounds for features array. Skipping for DL features."
                    )
            else:
                logging.warning(
                    f"DM: Active sample {s_id} has no storage_index. Skipping for DL features."
                )

        if not temp_samples_info:
            logging.debug(
                "DM: No valid active_samples with storage_index found for DL features."
            )
            self.active_sample_ids_for_current_clustering = []
            return None

        temp_samples_info.sort(key=lambda x: x[0])  # Sort by storage_index

        indices_to_fetch = [info[0] for info in temp_samples_info]
        ordered_sample_ids_for_features = [info[1] for info in temp_samples_info]

        self.active_sample_ids_for_current_clustering = ordered_sample_ids_for_features
        return self._all_features_array[indices_to_fetch]

    def _get_sample_ids_for_active_features(self) -> List[str]:
        """
        Returns a list of sample_ids corresponding to the rows in the
        array returned by _get_active_features_array, maintaining the same order.
        """
        if not self.samples:
            return []

        active_samples_with_indices = []
        for sample_id, sample in self.samples.items():
            if sample.is_active and sample.storage_index is not None:
                active_samples_with_indices.append(sample)

        # Sort by storage_index to match the order of _get_active_features_array
        active_samples_with_indices.sort(key=lambda s: s.storage_index)

        return [s.id for s in active_samples_with_indices]

    def _prepare_features_for_clustering(
        self, feature_sources_config: Dict[str, bool]
    ) -> Optional[np.ndarray]:
        """
        Prepares the final feature matrix for clustering based on selected sources.
        Feature sources: {'deep_learning': bool, 'morphological': bool, 'metadata': bool (placeholder)}
        Returns a 2D NumPy array (n_samples, n_combined_features) or None.
        Also populates `self.active_sample_ids_for_current_clustering` with the ordered list of sample IDs
        corresponding to the rows in the feature matrix.
        """
        self.active_sample_ids_for_current_clustering = []  # Reset

        use_dl = feature_sources_config.get("deep_learning", False)
        use_morph = feature_sources_config.get("morphological", False)

        dl_features_arr: Optional[np.ndarray] = None
        morph_features_arr: Optional[np.ndarray] = None

        if use_dl:
            dl_features_arr = (
                self._get_active_dl_features_array()
            )  # This also sets self.active_sample_ids_for_current_clustering
            if dl_features_arr is None:
                logging.error(
                    "DataManager: Deep learning features requested but not available or no active samples."
                )
                if not use_morph:
                    return None  # If only DL was requested and failed, abort
                return None

        # If morph features are used, get them for the *same set of active samples* defined by DL features (or all active if no DL)
        if use_morph:
            sample_ids_for_morph = []
            if self.active_sample_ids_for_current_clustering:  # If DL features set this
                sample_ids_for_morph = self.active_sample_ids_for_current_clustering
            elif not use_dl:  # If only morph is used, get all active samples
                active_samples_list = [
                    s
                    for s in self.samples.values()
                    if s.is_active and s.storage_index is not None
                ]
                active_samples_list.sort(
                    key=lambda s: s.storage_index
                )  # Ensure consistent order
                self.active_sample_ids_for_current_clustering = [
                    s.id for s in active_samples_list
                ]
                sample_ids_for_morph = self.active_sample_ids_for_current_clustering

            if sample_ids_for_morph:
                morph_features_arr = self._get_morphological_features_for_samples(
                    sample_ids_for_morph
                )
                if (
                    morph_features_arr is None and use_dl is False
                ):  # Only morph was requested and it failed
                    logging.error(
                        "DataManager: Morphological features requested but could not be retrieved."
                    )
                    return None
                if (
                    morph_features_arr is not None
                    and dl_features_arr is not None
                    and morph_features_arr.shape[0] != dl_features_arr.shape[0]
                ):
                    logging.error(
                        "DataManager: Mismatch in sample count between DL and Morphological features. This should not happen if using active_sample_ids_for_current_clustering."
                    )
                    return None  # Critical inconsistency
            elif not use_dl:  # Only morph was selected but no samples found
                logging.info(
                    "DataManager: Morphological features requested, but no active samples to process."
                )
                return None

        #  Concatenate features
        final_features: Optional[np.ndarray] = None
        if (
            use_dl
            and dl_features_arr is not None
            and use_morph
            and morph_features_arr is not None
        ):
            logging.info(
                f"DataManager: Concatenating DL features (shape: {dl_features_arr.shape}) and Morphological features (shape: {morph_features_arr.shape})."
            )
            # Ensure they have the same number of samples (rows)
            if dl_features_arr.shape[0] == morph_features_arr.shape[0]:
                final_features = np.concatenate(
                    (dl_features_arr, morph_features_arr), axis=1
                )
            else:
                logging.error(
                    "DataManager: Mismatch in number of samples for DL and Morphological features during concatenation."
                )
                return None
        elif use_dl and dl_features_arr is not None:
            final_features = dl_features_arr
        elif use_morph and morph_features_arr is not None:
            final_features = morph_features_arr
        else:
            logging.warning(
                "DataManager: No feature sources selected or features available for clustering."
            )
            return None

        if final_features is not None:
            logging.info(
                f"DataManager: Prepared final features for clustering, shape: {final_features.shape}"
            )
            scaler = StandardScaler()
            final_features = scaler.fit_transform(final_features)
            logging.info("DataManager: Applied StandardScaler to combined features.")

        return final_features

    def _get_morphological_features_for_samples(
        self, sample_ids: List[str]
    ) -> Optional[np.ndarray]:
        """
        Collects morphological features (mask attributes) for a given list of sample_ids.
        Returns a 2D NumPy array (n_samples, n_morph_features) or None.
        The order of rows in the returned array matches the order of sample_ids.
        """
        if not sample_ids:
            return None

        all_morph_features_list = []
        # Define a consistent order of morphological features to extract
        morph_feature_keys = sorted(
            [
                "area",
                "perimeter",
                "eccentricity",
                "solidity",
                "aspect_ratio",
                "circularity",
                "major_axis_length",
                "minor_axis_length",
                "mean_intensity",
                "std_intensity",
                "compactness",
                "convexity",
                "curl",
                "volume",
            ]
        )

        for sample_id in sample_ids:
            sample = self.samples.get(sample_id)
            sample_morph_features = []
            if sample and sample.mask_id and sample.mask_id in self.masks:
                mask_attributes = self.masks[sample.mask_id].attributes
                for key in morph_feature_keys:
                    sample_morph_features.append(float(mask_attributes.get(key, 0.0)))
            else:
                # Sample has no mask or attributes, fill with zeros or NaNs
                sample_morph_features = [0.0] * len(morph_feature_keys)
                logging.debug(
                    f"DataManager: No mask/attributes for sample {sample_id} for morphological features. Using zeros."
                )

            all_morph_features_list.append(sample_morph_features)

        if not all_morph_features_list:
            return None

        return np.array(all_morph_features_list, dtype=np.float32)

    @Slot(dict)  # ui_params from ClustersPresenter.control_panel
    def perform_clustering(self, ui_params: dict):
        """
        Slot to handle clustering request. Orchestrates feature prep, dim reduction, and clustering.
        """
        logging.debug(
            f"DataManager: Received request _on_perform_clustering_requested with params: {ui_params}"
        )
        if not self.processor:
            logging.error(
                "DataManager: Cannot perform clustering - Processor not initialized."
            )
            self.clustering_performed.emit()
            return

        feature_sources_config = ui_params.get("feature_sources", {})
        dl_settings = ui_params.get("dl_feature_extractor_settings", {})
        expected_dl_model = dl_settings.get("model_name")

        #  Sanity check for DL features if requested
        if feature_sources_config.get("deep_learning"):
            if self._all_features_array is None:
                logging.error(
                    "DataManager: DL features requested for clustering, but _all_features_array is None. Aborting clustering. Please extract features."
                )
                self.clustering_performed.emit()
                return
            if self._current_features_model_name != expected_dl_model:
                logging.error(
                    f"DataManager: DL features requested with model '{expected_dl_model}', "
                    f"but current features are for '{self._current_features_model_name}'. Aborting clustering. Please re-extract features with the correct model."
                )
                self.clustering_performed.emit()
                return
            if (
                self._current_features_dimension != self.processor.feature_dim
                and self.processor.model_name == expected_dl_model
            ):
                logging.error(
                    f"DataManager: DL features dimension mismatch. Array dim: {self._current_features_dimension}, "
                    f"Processor ({self.processor.model_name}) expects: {self.processor.feature_dim}. Aborting."
                )
                self.clustering_performed.emit()
                return

        combined_features = self._prepare_features_for_clustering(
            feature_sources_config
        )

        if combined_features is None or combined_features.shape[0] == 0:
            logging.error(
                "DataManager: No features available for clustering after preparation."
            )
            self.clustering_performed.emit()
            return

        active_sample_ids = self.active_sample_ids_for_current_clustering
        if (
            not active_sample_ids
            or len(active_sample_ids) != combined_features.shape[0]
        ):
            logging.error(
                "DataManager: Mismatch between combined features and active sample IDs. Clustering aborted."
            )
            self.clustering_performed.emit()
            return

        #  Dimensionality Reduction
        dim_reduction_settings = ui_params.get("dim_reduction_settings", {})
        reduction_method = dim_reduction_settings.get("method")
        reduction_params = dim_reduction_settings.get("params", {})

        features_for_clustering = combined_features
        if reduction_method:
            logging.info(
                f"DataManager: Applying {reduction_method} dimensionality reduction..."
            )
            if reduction_method.upper() == "PCA":
                features_for_clustering = self.processor.reduce_dimensions_pca(
                    combined_features, **reduction_params
                )
            elif reduction_method.upper() == "UMAP":
                features_for_clustering = self.processor.reduce_dimensions_umap(
                    combined_features, reduction_params
                )
            else:
                logging.warning(
                    f"DataManager: Unknown DR method '{reduction_method}'. Using combined features."
                )
        else:
            logging.info("DataManager: No dimensionality reduction selected.")

        if features_for_clustering is None or features_for_clustering.shape[0] == 0:
            logging.error(
                "DataManager: Features became empty after dimensionality reduction."
            )
            self.clustering_performed.emit()
            return

        #  K-Means Clustering
        kmeans_settings = ui_params.get("kmeans_settings", {})
        n_clusters = kmeans_settings.get("n_clusters", 10)
        find_k_elbow = kmeans_settings.get("find_k_elbow", False)
        n_iter = self.settings.get("CLUSTERING_N_ITER", 300)
        n_redo = self.settings.get("CLUSTERING_N_REDO", 10)

        logging.info(
            f"DataManager: Performing K-Means on {features_for_clustering.shape[0]} samples, K={n_clusters}, find_k={find_k_elbow}"
        )
        try:
            cluster_labels = self.processor.cluster_images(
                features_for_clustering, n_clusters, n_iter, n_redo, find_k_elbow
            )
        except Exception as e:
            logging.error(
                f"DataManager: Error during processor.cluster_images: {e}",
                exc_info=True,
            )
            self.clustering_performed.emit()
            return
        if not cluster_labels or len(cluster_labels) != len(active_sample_ids):
            logging.error("DataManager: Clustering labels error.")
            self.clustering_performed.emit()
            return

        #  Update Data Structures and DB
        self.clear_clusters(emit_signal=False)
        label_to_cluster_obj: Dict[int, Cluster] = {}
        unique_labels = sorted(list(set(cluster_labels)))
        for label_val in unique_labels:
            cluster_id = str(uuid.uuid4())
            color = self._generate_random_color()
            cluster_obj = Cluster(id=cluster_id, color=color)
            self.clusters[cluster_id] = cluster_obj
            label_to_cluster_obj[label_val] = cluster_obj
            self.add_pending_db_operation(
                "INSERT INTO clusters (id, color) VALUES (?, ?)", (cluster_id, color)
            )
        for i, sample_id in enumerate(active_sample_ids):
            label = cluster_labels[i]
            assigned_cluster_obj = label_to_cluster_obj.get(label)
            sample = self.samples.get(sample_id)
            if sample and assigned_cluster_obj:
                sample.cluster_ids.add(assigned_cluster_obj.id)
                assigned_cluster_obj.add_image(sample)
                self.add_pending_db_operation(
                    "INSERT INTO samples_clusters (sample_id, cluster_id) VALUES (?, ?)",
                    (sample_id, assigned_cluster_obj.id),
                )
        self._flush_pending_db_operations(
            callback=lambda: self.clustering_performed.emit()
        )
        logging.info(
            "DataManager: Clustering process completed and results queued/saved."
        )

    def clear_clusters(self, emit_signal=True):
        """Clears all cluster data from DB and memory."""
        logging.info("Clearing all clusters...")
        # Clear from DB
        all_cluster_ids = list(self.clusters.keys())
        for cluster_id in all_cluster_ids:
            # Queuing for batch delete
            self.add_pending_db_operation(
                "DELETE FROM samples_clusters WHERE cluster_id = ?", (cluster_id,)
            )
            self.add_pending_db_operation(
                "DELETE FROM clusters WHERE id = ?", (cluster_id,)
            )

        # Clear from memory
        for sample in self.samples.values():
            sample.cluster_ids.clear()
        self.clusters.clear()

        if all_cluster_ids:  # Only flush and emit if there was something to clear
            self._flush_pending_db_operations(
                callback=lambda: (
                    self.clusters_invalidated.emit() if emit_signal else None
                )
            )
        elif emit_signal:  # If nothing to clear, still emit if requested
            self.clusters_invalidated.emit()
        logging.info("Cluster clearing operations queued/completed.")

    @Slot()  # Make it a slot so invokeMethod can call it
    def close_db(self):
        """Explicitly closes the database connection for this DataManager instance."""
        logging.info(
            f"DataManager for session {self.session.id} received request to close DB."
        )
        # Before closing, ensure any dirty data is persisted
        self._flush_pending_db_operations(
            lambda: self._persist_consolidated_data(lambda: self._close_db_actual())
        )

    def _close_db_actual(self):
        if hasattr(self, "db_manager") and self.db_manager:
            try:
                self.db_manager.close()
                logging.info(
                    f"Database connection closed for session {self.session.id}."
                )
            except Exception as e:
                logging.error(
                    f"Error closing database connection for session {self.session.id}: {e}"
                )
        else:
            logging.warning(
                f"DataManager for session {self.session.id} has no db_manager to close."
            )

    # Getters
    def get_image(self, image_id: str) -> Optional[Sample]:
        return self.samples.get(image_id)

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        return self.clusters.get(cluster_id)

    def get_class(self, class_id: str) -> Optional[SampleClass]:
        return self.classes.get(class_id)

    def get_class_by_name(self, class_name: str) -> Optional[SampleClass]:
        for sc in self.classes.values():
            if sc.name.lower() == class_name.lower():
                return sc
        return None

    #  Utility Methods
    def _generate_random_color(self) -> str:
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def _create_default_class(self) -> Optional[SampleClass]:
        """
        Ensures the default "Uncategorized" class exists.
        If it doesn't, it creates it in memory and queues DB insertion.
        Returns the SampleClass object for "Uncategorized".
        """
        uncategorized_class = self.get_class_by_name("Uncategorized")
        if not uncategorized_class:
            logging.info("Default 'Uncategorized' class not found, creating it.")
            uncategorized_class = self.create_class(
                name="Uncategorized", color="#A9A9A9"
            )
            if not uncategorized_class:
                logging.error(
                    "Failed to create the default 'Uncategorized' class during _create_default_class."
                )
                return None
        return uncategorized_class

    #  Class Management Methods (need to use pending_db_operations)
    @Slot(str, str)
    def create_class(
        self, name: str, color: Optional[str] = None
    ) -> Optional[SampleClass]:
        if self.get_class_by_name(name):
            logging.warning(f"Class with name '{name}' already exists.")
            return self.get_class_by_name(name)

        class_id = str(uuid.uuid4())
        final_color = color or self._generate_random_color()
        image_class = SampleClass(id=class_id, name=name, color=final_color)
        self.classes[class_id] = image_class

        self.add_pending_db_operation(
            "INSERT INTO classes (id, name, color) VALUES (?, ?, ?)",
            (class_id, name, final_color),
        )
        self._flush_pending_db_operations()
        self.class_added.emit(class_id)
        logging.info(f"Class '{name}' (ID: {class_id}) created and DB insert queued.")
        return image_class

    def rename_class(self, class_id: str, new_name: str) -> None:
        if class_id not in self.classes:
            logging.error(f"Cannot rename class: ID {class_id} not found.")
            return
        if not new_name.strip():
            logging.error(f"Cannot rename class {class_id}: New name is empty.")
            return

        existing_class_with_new_name = self.get_class_by_name(new_name.strip())
        if existing_class_with_new_name and existing_class_with_new_name.id != class_id:
            logging.error(
                f"Cannot rename class {class_id} to '{new_name}': Name already exists."
            )
            return

        class_object = self.classes[class_id]
        old_name = class_object.name
        class_object.name = new_name.strip()

        self.add_pending_db_operation(
            "UPDATE classes SET name = ? WHERE id = ?", (new_name.strip(), class_id)
        )
        self._flush_pending_db_operations(
            callback=lambda: self.class_updated.emit(class_id)
        )
        logging.info(
            f"Class '{old_name}' (ID: {class_id}) renamed to '{new_name.strip()}', DB update queued."
        )

    @Slot(list, str)
    def assign_images_to_class(self, image_ids: List[str], new_class_id: str):
        new_class = self.get_class(new_class_id)
        if not new_class:
            logging.error(f"Target class ID {new_class_id} not found for assignment.")
            return

        updated_class_ids = {
            new_class_id
        }  # Track all affected class IDs for emitting signals

        for image_id in image_ids:
            sample = self.samples.get(image_id)
            if not sample:
                logging.warning(
                    f"Image ID {image_id} not found during class assignment."
                )
                continue

            old_class_id = sample.class_id
            if old_class_id == new_class_id:
                continue  # Already in the target class

            # Update DB (queued)
            if old_class_id:  # Remove from old class junction
                self.add_pending_db_operation(
                    "DELETE FROM samples_classes WHERE sample_id = ? AND class_id = ?",
                    (image_id, old_class_id),
                )
                updated_class_ids.add(old_class_id)

            self.add_pending_db_operation(  # Add to new class junction
                "INSERT OR IGNORE INTO samples_classes (sample_id, class_id) VALUES (?, ?)",
                (image_id, new_class_id),
            )
            self.add_pending_db_operation(  # Update sample's main class_id ref
                "UPDATE samples SET class_id = ? WHERE id = ?", (new_class_id, image_id)
            )

            # Update in-memory objects
            if old_class_id and old_class_id in self.classes:
                self.classes[old_class_id].samples.discard(sample)
            new_class.samples.add(sample)
            sample.class_id = new_class_id

        if image_ids:  # If any assignments were processed
            self._flush_pending_db_operations(
                callback=lambda: [
                    self.class_updated.emit(uid) for uid in updated_class_ids
                ]
            )
        logging.info(
            f"Assignment of {len(image_ids)} images to class {new_class.name} queued."
        )

    @Slot(str)
    def delete_class(self, class_id: str) -> None:
        class_to_delete = self.get_class(class_id)
        if not class_to_delete:
            logging.warning(f"Class ID {class_id} not found for deletion.")
            return
        if class_to_delete.name.lower() == "uncategorized":
            logging.warning("Cannot delete the default 'Uncategorized' class.")
            return

        uncategorized_class = self.get_class_by_name("Uncategorized")
        if not uncategorized_class:  # Should not happen if _create_default_class works
            logging.error(
                "Default 'Uncategorized' class not found. Cannot move images from deleted class."
            )
            return

        # Move images to Uncategorized first (in memory and queue DB ops)
        image_ids_to_move = [s.id for s in class_to_delete.samples]
        if image_ids_to_move:
            self.assign_images_to_class(
                image_ids_to_move, uncategorized_class.id
            )  # This queues its own DB ops

        # Queue deletion of the class itself from DB
        self.add_pending_db_operation("DELETE FROM classes WHERE id = ?", (class_id,))

        # Remove from in-memory
        del self.classes[class_id]

        # Flush and emit
        self._flush_pending_db_operations(
            callback=lambda: self.class_deleted.emit(class_id)
        )
        logging.info(f"Deletion of class {class_id} ('{class_to_delete.name}') queued.")

    @Slot(dict)
    def export_data(self, params: dict) -> None:
        """Exports data based on parameters. Charts export is currently ignored."""
        include_masks_previews = params.get("include_masks", False)
        include_clusters = params.get("include_clusters", False)
        include_params = params.get("include_params", False)
        # include_charts = params.get("include_charts", False) # Received but not used
        export_folder_path_base = params.get("export_folder_path", "")

        if not export_folder_path_base or not self.samples:
            logging.warning("Export cancelled: No export path or no samples.")
            self.export_finished.emit(
                False, "Export path not set or no data to export."
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_export_folder = os.path.join(
            export_folder_path_base, f"{self.session.name}_export_{timestamp}"
        )

        try:
            os.makedirs(session_export_folder, exist_ok=True)

            # 1. Export Images by Class
            if self.classes:  # Check if there are any classes to export
                classes_base_folder = os.path.join(
                    session_export_folder, "Images_by_Class"
                )
                os.makedirs(classes_base_folder, exist_ok=True)
                for class_obj in self.classes.values():
                    class_folder_name = class_obj.name.replace("/", "_").replace(
                        "\\", "_"
                    )
                    current_class_export_path = os.path.join(
                        classes_base_folder, class_folder_name
                    )
                    os.makedirs(current_class_export_path, exist_ok=True)
                    for sample in class_obj.samples:
                        if sample.is_active:
                            try:
                                if os.path.exists(sample.path):
                                    shutil.copy2(sample.path, current_class_export_path)
                                else:
                                    logging.warning(
                                        f"Sample path not found for class export: {sample.path}"
                                    )
                            except Exception as e_copy:
                                logging.error(
                                    f"Error copying {sample.path} for class export: {e_copy}"
                                )
            else:
                logging.info("No classes defined, skipping export of images by class.")

            # 2. Export Masked Image Previews (if selected)
            if include_masks_previews:
                if self.masks:  # Check if there are any masks
                    masked_previews_folder = os.path.join(
                        session_export_folder, "Masked_Image_Previews"
                    )
                    os.makedirs(masked_previews_folder, exist_ok=True)
                    for sample in self.samples.values():
                        if (
                            sample.is_active
                            and sample.mask_id
                            and sample.mask_id in self.masks
                        ):
                            mask_obj = self.masks[sample.mask_id]
                            if mask_obj.masked_image_path and os.path.exists(
                                mask_obj.masked_image_path
                            ):
                                try:
                                    shutil.copy2(
                                        mask_obj.masked_image_path,
                                        masked_previews_folder,
                                    )
                                except Exception as e_copy:
                                    logging.error(
                                        f"Error copying preview {mask_obj.masked_image_path}: {e_copy}"
                                    )
                            else:
                                logging.warning(
                                    f"Masked preview path not found: {mask_obj.masked_image_path}"
                                )
            else:
                logging.info("No masks available, skipping export of mask previews.")

            # 3. Export Images by Cluster
            if include_clusters:
                if self.clusters:  # Check if there are any clusters
                    clusters_base_folder = os.path.join(
                        session_export_folder, "Images_by_Cluster"
                    )
                    os.makedirs(clusters_base_folder, exist_ok=True)
                    for cluster_obj in self.clusters.values():
                        cluster_folder_name = cluster_obj.id[:8]
                        current_cluster_export_path = os.path.join(
                            clusters_base_folder, cluster_folder_name
                        )
                        os.makedirs(current_cluster_export_path, exist_ok=True)
                        for sample in cluster_obj.samples:
                            if sample.is_active:
                                try:
                                    if os.path.exists(sample.path):
                                        shutil.copy2(
                                            sample.path, current_cluster_export_path
                                        )
                                    else:
                                        logging.warning(
                                            f"Sample path not found for cluster export: {sample.path}"
                                        )
                                except Exception as e_copy:
                                    logging.error(
                                        f"Error copying {sample.path} for cluster export: {e_copy}"
                                    )
                else:
                    logging.info(
                        "No clusters defined, skipping export of images by cluster."
                    )

            # 4. Export Calculated Parameters (Mask Attributes CSV)
            if include_params:
                if self.masks:  # Check if there are any masks with attributes
                    params_csv_path = os.path.join(
                        session_export_folder, "mask_attributes.csv"
                    )
                    all_attr_keys = set()
                    for mask_obj in self.masks.values():
                        all_attr_keys.update(mask_obj.attributes.keys())

                    if all_attr_keys:  # Only create CSV if there are attributes
                        sorted_attr_keys = sorted(list(all_attr_keys))
                        fieldnames = [
                            "sample_id",
                            "sample_path",
                            "class_name",
                        ] + sorted_attr_keys

                        with open(
                            params_csv_path, "w", newline="", encoding="utf-8"
                        ) as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            for sample in self.samples.values():
                                if not sample.is_active:
                                    continue
                                row_data = {
                                    "sample_id": sample.id,
                                    "sample_path": sample.path,
                                    "class_name": (
                                        self.classes[sample.class_id].name
                                        if sample.class_id
                                        and sample.class_id in self.classes
                                        else "N/A"
                                    ),
                                }
                                for key in sorted_attr_keys:  # Initialize all attr keys
                                    row_data[key] = ""

                                if sample.mask_id and sample.mask_id in self.masks:
                                    mask_obj = self.masks[sample.mask_id]
                                    for (
                                        attr_name,
                                        attr_val,
                                    ) in mask_obj.attributes.items():
                                        if attr_name in row_data:
                                            row_data[attr_name] = attr_val
                                writer.writerow(row_data)
                    else:
                        logging.info("No mask attributes found to export to CSV.")
                else:
                    logging.info(
                        "No masks available, skipping export of calculated parameters."
                    )

            logging.info(f"Data exported successfully to {session_export_folder}")
            self.export_finished.emit(True, session_export_folder)

        except Exception as e:
            logging.error(f"Error during data export: {e}", exc_info=True)
            self.export_finished.emit(False, str(e))

    @Slot(str, int, float, int, int, dict)
    def _on_train_classifier_requested(
        self,
        feature_model_name_for_probe: str,
        epochs: int,
        lr: float,
        batch_size: int,
        patience: int,
        classes_data_for_training: Dict[str, List[str]],
    ):
        logging.info(
            f"DM: Received request to train classifier for model '{feature_model_name_for_probe}'."
        )

        if not self.processor or not self.processor.engine:
            logging.error("DM Train: Processor or its engine not initialized.")
            self.classifier_training_finished.emit(
                feature_model_name_for_probe, False, "Processor not ready."
            )
            return

        if self.processor.model_name != feature_model_name_for_probe or not isinstance(
            self.processor.engine, PyTorchFeatureExtractor
        ):
            logging.error(
                f"DM Train: Current processor model ('{self.processor.model_name}', type: {type(self.processor.engine)}) "
                f"is not the requested PyTorch model ('{feature_model_name_for_probe}') for training."
            )
            self.classifier_training_finished.emit(
                feature_model_name_for_probe,
                False,
                "Active processor model mismatch or not PyTorch.",
            )
            return

        pytorch_engine: PyTorchFeatureExtractor = self.processor.engine

        # 1. Prepare features and labels
        feature_tensors_list: List[torch.Tensor] = []
        integer_labels_list: List[int] = []

        # Create class_id to integer label mapping
        unique_class_ids_in_training_data = sorted(
            list(classes_data_for_training.keys())
        )
        class_id_to_int_label: Dict[str, int] = {
            class_id: i for i, class_id in enumerate(unique_class_ids_in_training_data)
        }
        num_classes = len(unique_class_ids_in_training_data)

        if num_classes < 2:
            logging.error(
                "DM Train: Not enough unique classes with samples provided for training."
            )
            self.classifier_training_finished.emit(
                feature_model_name_for_probe,
                False,
                "Need at least 2 classes with samples.",
            )
            return

        found_samples_for_training = 0
        for class_id, sample_ids in classes_data_for_training.items():
            if not sample_ids:
                continue  # Skip classes with no samples in the input

            int_label = class_id_to_int_label.get(class_id)
            if (
                int_label is None
            ):  # Should not happen if keys are from classes_data_for_training
                logging.warning(
                    f"DM Train: Class ID {class_id} not in mapping. Skipping its samples."
                )
                continue

            for sample_id in sample_ids:
                sample_obj = self.samples.get(sample_id)
                if not sample_obj or not sample_obj.is_active:
                    continue

                features_np: Optional[np.ndarray] = None
                if (
                    sample_obj.storage_index is not None
                    and self._all_features_array is not None
                    and 0
                    <= sample_obj.storage_index
                    < self._all_features_array.shape[0]
                ):
                    features_np = self._all_features_array[sample_obj.storage_index]
                elif (
                    sample_obj.features is not None
                ):  # Fallback to individual sample features if array is problematic
                    features_np = sample_obj.features

                if features_np is not None:
                    try:
                        # Ensure features are of the correct dimension for the probe
                        if features_np.shape[0] != pytorch_engine.feature_dim:
                            logging.warning(
                                f"DM Train: Feature dim mismatch for sample {sample_id} ({features_np.shape[0]}) vs engine ({pytorch_engine.feature_dim}). Skipping."
                            )
                            continue
                        feature_tensors_list.append(
                            torch.from_numpy(features_np.astype(np.float32))
                        )
                        integer_labels_list.append(int_label)
                        found_samples_for_training += 1
                    except Exception as e:
                        logging.error(
                            f"DM Train: Error converting features for sample {sample_id}: {e}"
                        )
                else:
                    logging.warning(
                        f"DM Train: No features found for active sample {sample_id} in class {class_id}."
                    )

        if (
            not feature_tensors_list or len(feature_tensors_list) < num_classes
        ):  # Need at least one sample per class effectively
            logging.error(
                "DM Train: Not enough samples with features available for training the classifier."
            )
            self.classifier_training_finished.emit(
                feature_model_name_for_probe,
                False,
                "Insufficient samples with features.",
            )
            return

        logging.info(
            f"DM Train: Prepared {len(feature_tensors_list)} feature vectors for {num_classes} classes."
        )

        def training_progress_update(epoch: int, metric: float):
            self.classifier_training_progress.emit(
                feature_model_name_for_probe, epoch, metric
            )

        try:
            pytorch_engine.train_linear_probe(
                features_list=feature_tensors_list,
                integer_labels_list=integer_labels_list,
                num_classes=num_classes,
                class_id_to_int_label_map=class_id_to_int_label,  # Pass the map
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                patience=patience,
                progress_callback=training_progress_update,
            )
            logging.info(
                f"DM Train: Linear probe training initiated for '{feature_model_name_for_probe}'."
            )
            self.classifier_training_finished.emit(
                feature_model_name_for_probe, True, "Training completed."
            )
        except Exception as e:
            logging.error(
                f"DM Train: Error during call to train_linear_probe for '{feature_model_name_for_probe}': {e}",
                exc_info=True,
            )
            self.classifier_training_finished.emit(
                feature_model_name_for_probe, False, f"Training error: {e}"
            )

    @Slot(str)
    def _on_run_classification_requested(self, classifier_name: str):
        logging.info(
            f"DM: Received request to run classification using probe for '{classifier_name}'."
        )

        if not self.processor or not self.processor.engine:
            logging.error("DM Classify: Processor or its engine not initialized.")
            self.classification_run_finished.emit(
                classifier_name, False, "Processor not ready."
            )
            return

        if self.processor.model_name != classifier_name or not isinstance(
            self.processor.engine, PyTorchFeatureExtractor
        ):
            logging.error(
                f"DM Classify: Current processor ('{self.processor.model_name}') is not the one probe was trained for ('{classifier_name}') or not PyTorch."
            )
            self.classification_run_finished.emit(
                classifier_name, False, "Processor-probe mismatch or not PyTorch."
            )
            return

        pytorch_engine: PyTorchFeatureExtractor = self.processor.engine
        if pytorch_engine.linear_probe is None:
            logging.error(
                f"DM Classify: Linear probe for '{classifier_name}' is not trained/available."
            )
            self.classification_run_finished.emit(
                classifier_name, False, "Probe not trained."
            )
            return

        if not pytorch_engine.int_label_to_class_id_map:
            logging.error(
                f"DM Classify: Label-to-classID map missing in probe for '{classifier_name}'. Cannot map predictions."
            )
            self.classification_run_finished.emit(
                classifier_name, False, "Probe label map missing."
            )
            return

        logging.info(
            f"DM Classify: Classifying {len(self.active_samples)} active samples..."
        )
        samples_reclassified_count = 0
        for sample_id, sample_obj in self.active_samples.items():
            if not sample_obj.is_active:
                continue  # Should be redundant if iterating active_samples

            features_np: Optional[np.ndarray] = None
            if (
                sample_obj.storage_index is not None
                and self._all_features_array is not None
                and 0 <= sample_obj.storage_index < self._all_features_array.shape[0]
            ):
                features_np = self._all_features_array[sample_obj.storage_index]
            elif sample_obj.features is not None:
                features_np = sample_obj.features

            if features_np is not None:
                feature_tensor = torch.from_numpy(features_np.astype(np.float32))
                predicted_int_label = pytorch_engine.classify_with_probe(feature_tensor)

                if predicted_int_label is not None:
                    predicted_class_id = pytorch_engine.int_label_to_class_id_map.get(
                        predicted_int_label
                    )
                    if predicted_class_id:
                        if sample_obj.class_id != predicted_class_id:
                            # Use assign_images_to_class for proper DB update and signal emission
                            # This method already handles queuing and flushing.
                            self.assign_images_to_class(
                                image_ids=[sample_id], new_class_id=predicted_class_id
                            )
                            samples_reclassified_count += 1
                            # logging.debug(f"Sample {sample_id} re-classified from {sample_obj.class_id} to {predicted_class_id}")
                    else:
                        logging.warning(
                            f"DM Classify: Predicted int label {predicted_int_label} has no mapping to class_id."
                        )
                else:
                    logging.warning(
                        f"DM Classify: Probe failed to classify sample {sample_id}."
                    )
            else:
                logging.warning(
                    f"DM Classify: No features for active sample {sample_id}. Cannot classify."
                )

        message = f"{samples_reclassified_count} samples had their class updated."
        logging.info(f"DM Classify: {message}")
        self.classification_run_finished.emit(classifier_name, True, message)
