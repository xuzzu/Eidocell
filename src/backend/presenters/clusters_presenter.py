# backend/presenters/clusters_presenter.py
import logging
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition

from backend.config import (
    CLUSTERING_DEFAULT_N_CLUSTERS,
    COLLAGE_RES_SCALE,
    get_available_models,
)
from backend.data_manager import DataManager  # For type hint
from backend.data_manager_interface import DataManagerInterface
from backend.helpers.context_menu_handler import ContextMenuHandler
from backend.helpers.ctrl_helper import ControlHelper
from backend.utils.image_utils import merge_images_collage
from UI.dialogs.class_cluster_summary import ClassClusterSummary
from UI.dialogs.class_cluster_view import ClassClusterViewer

# from UI.dialogs.custom_info_bar import CustomInfoBar # Using qfluentwidgets.InfoBar directly
from UI.dialogs.progress_infobar import ProgressInfoBar

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClustersPresenter(QObject):
    def __init__(
        self,
        clusters_view_widget_ref: Any,  # ClustersViewWidget
        data_manager_interface: DataManagerInterface,
        data_manager_instance: DataManager,
        control_panel_ref: Any,  # ControlPanel
        images_per_preview: int = 25,
    ):
        super().__init__()
        self.clusters_view_widget = clusters_view_widget_ref
        self.data_manager_interface = data_manager_interface
        self.data_manager = data_manager_instance
        self.control_panel = control_panel_ref

        self.context_menu_handler = ContextMenuHandler(self)
        self.control_helper = ControlHelper(self.clusters_view_widget)

        self.last_clicked_card_id: Optional[str] = None
        self.selected_card_ids: Set[str] = set()
        self.ctrl_pressed: bool = False
        self.images_per_preview: int = self.data_manager.settings.get(
            "images_per_collage", images_per_preview
        )
        self.progress_info_bar: Optional[ProgressInfoBar] = None

        #  Flags for tracking changes since last successful clustering run
        self.dl_feature_config_changed: bool = (
            True  # True if DL model or sources using DL changed
        )
        self.morph_feature_config_changed: bool = (
            True  # True if morphological sources changed (currently just on/off)
        )
        self.dim_reduction_config_changed: bool = (
            True  # True if DR method or its params changed
        )
        self.kmeans_config_changed: bool = True  # True if K-Means params (K) changed

        # Store the parameters of the last successful full run
        self._last_successful_run_params: Optional[Dict[str, Any]] = None
        # Store parameters for a run that might be pending feature extraction
        self._pending_clustering_params_after_extraction: Optional[Dict[str, Any]] = (
            None
        )
        # Store params for current run attempt, to update _last_successful_run_params on success
        self._current_run_ui_params: Optional[Dict[str, Any]] = None

        self._connect_ui_signals()
        self._connect_control_panel_param_changes()  # New method to connect param changes
        logging.info("ClustersPresenter initialized.")

    def _connect_ui_signals(self):
        if not self.control_panel or not self.clusters_view_widget:
            return
        try:
            self.control_panel.analysis_requested.connect(
                self.start_analysis_from_ui_params
            )
            self.control_panel.reset_analysis_requested.connect(self.reset_analysis)

            # Connect extractor_model_selector change to _on_dl_model_config_changed
            # (This was already in your provided code, just ensuring it's clear)
            if hasattr(self.control_panel, "extractor_model_selector"):
                self.control_panel.extractor_model_selector.currentTextChanged.connect(
                    self._on_dl_model_config_changed
                )
            elif hasattr(
                self.control_panel, "kmeans_model_selector"
            ):  # Fallback if old name
                self.control_panel.kmeans_model_selector.currentTextChanged.connect(
                    self.on_model_selected_in_ui
                )

            self.control_helper.ctrl_signal.connect(self.set_ctrl_pressed)
        except AttributeError as e:
            logging.error(
                f"Failed to connect UI signals in ClustersPresenter: {e}. View/ControlPanel structure might have changed."
            )

    def _connect_control_panel_param_changes(self):
        """Connect signals from ControlPanel parameters to flag changes."""
        if not self.control_panel:
            return
        try:
            # Feature Sources
            self.control_panel.cb_use_dl_features.stateChanged.connect(
                self._on_feature_source_config_changed
            )
            self.control_panel.cb_use_morph_features.stateChanged.connect(
                self._on_feature_source_config_changed
            )
            # self.control_panel.cb_use_meta_features.stateChanged.connect(self._on_feature_source_config_changed) # If added

            # Dimensionality Reduction
            self.control_panel.dim_reduction_method_selector.currentTextChanged.connect(
                self._on_dim_reduction_config_changed
            )
            # PCA Params
            if hasattr(self.control_panel.pca_params_widget, "n_components_spinbox"):
                self.control_panel.pca_params_widget.n_components_spinbox.valueChanged.connect(
                    self._on_dim_reduction_config_changed
                )
            # UMAP Params
            if hasattr(self.control_panel.umap_params_widget, "n_neighbors_spinbox"):
                self.control_panel.umap_params_widget.n_neighbors_spinbox.valueChanged.connect(
                    self._on_dim_reduction_config_changed
                )
                self.control_panel.umap_params_widget.min_dist_spinbox.valueChanged.connect(
                    self._on_dim_reduction_config_changed
                )
                self.control_panel.umap_params_widget.n_components_umap_spinbox.valueChanged.connect(
                    self._on_dim_reduction_config_changed
                )
            # TSNE Params (if added later)
            if hasattr(self.control_panel, "tsne_params_widget"):
                if hasattr(self.control_panel.tsne_params_widget, "perplexity_spinbox"):
                    self.control_panel.tsne_params_widget.perplexity_spinbox.valueChanged.connect(
                        self._on_dim_reduction_config_changed
                    )
                    self.control_panel.tsne_params_widget.learning_rate_spinbox.valueChanged.connect(
                        self._on_dim_reduction_config_changed
                    )
                    self.control_panel.tsne_params_widget.n_components_tsne_spinbox.valueChanged.connect(
                        self._on_dim_reduction_config_changed
                    )
                    self.control_panel.tsne_params_widget.n_iter_spinbox.valueChanged.connect(
                        self._on_dim_reduction_config_changed
                    )

            # K-Means Params
            self.control_panel.clusters_slider.valueChanged.connect(
                self._on_kmeans_config_changed
            )
            # if hasattr(self.control_panel, 'cb_find_k_elbow'): # If elbow method is used
            #     self.control_panel.cb_find_k_elbow.stateChanged.connect(self._on_kmeans_config_changed)
        except AttributeError as e:
            logging.error(f"Error connecting ControlPanel param change signals: {e}")

    #  Slots for UI configuration changes, setting flags
    @Slot()
    def _on_dl_model_config_changed(self):
        logging.debug("ClustersPresenter: DL model or source config changed.")
        self.dl_feature_config_changed = True
        self.dim_reduction_config_changed = True  # DR depends on features
        self.kmeans_config_changed = True  # K-Means depends on DR output

    @Slot()
    def _on_feature_source_config_changed(self):
        logging.debug("ClustersPresenter: Feature source selection changed.")
        # Check if DL feature usage specifically changed
        if self.control_panel and self.control_panel.cb_use_dl_features.isChecked() != (
            self._last_successful_run_params.get("feature_sources", {}).get(
                "deep_learning", False
            )
            if self._last_successful_run_params
            else True
        ):
            self.dl_feature_config_changed = True

        # Check if Morphological feature usage specifically changed
        if (
            self.control_panel
            and self.control_panel.cb_use_morph_features.isChecked()
            != (
                self._last_successful_run_params.get("feature_sources", {}).get(
                    "morphological", False
                )
                if self._last_successful_run_params
                else False
            )
        ):
            self.morph_feature_config_changed = True

        self.dim_reduction_config_changed = True
        self.kmeans_config_changed = True

    @Slot()
    def _on_dim_reduction_config_changed(self):
        logging.debug(
            "ClustersPresenter: Dimensionality reduction configuration changed."
        )
        self.dim_reduction_config_changed = True
        self.kmeans_config_changed = True  # K-Means depends on DR output

    @Slot()
    def _on_kmeans_config_changed(self):
        logging.debug("ClustersPresenter: K-Means configuration changed.")
        self.kmeans_config_changed = True

    @Slot()
    def handle_features_invalidated(self):
        logging.info("ClustersPresenter: DataManager invalidated features.")
        self.dl_feature_config_changed = True
        self.morph_feature_config_changed = True
        self.dim_reduction_config_changed = True
        self.kmeans_config_changed = True
        self._last_successful_run_params = None

        if self.clusters_view_widget:
            InfoBar.warning(
                title="Features Invalidated",
                content="Feature model changed or features cleared. Re-extraction and re-clustering needed.",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
            )
        if self.control_panel:
            self.control_panel.startButton.setEnabled(True)
        # Also clear existing cluster cards from UI as they are based on old features
        if self.clusters_view_widget:
            self.clusters_view_widget.clear_cluster_cards()
        self.selected_card_ids.clear()

    @Slot()
    def handle_clusters_invalidated(self):
        logging.info(
            "ClustersPresenter received clusters_invalidated signal (likely due to gate changes or reset)."
        )

        # Reset flags to indicate a full re-run is needed
        self.dl_feature_config_changed = True
        self.morph_feature_config_changed = True
        self.dim_reduction_config_changed = True
        self.kmeans_config_changed = True
        self._last_successful_run_params = (
            None  # Previous successful parameters are no longer valid
        )
        self._current_run_ui_params = None  # Clear current attempt

        if self.clusters_view_widget and len(self.clusters_view_widget.clusters) > 0:
            self.clusters_view_widget.clear_cluster_cards()
            InfoBar.warning(  # Use warning as it's a significant state change
                title="Clusters Outdated",
                content="The set of active images has changed due to gating, or clusters were reset. "
                "Please re-run the clustering analysis.",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
            )
        self.selected_card_ids.clear()
        if self.control_panel:
            self.control_panel.startButton.setEnabled(True)  # Re-enable start button
            # Optionally reset control panel UI elements to defaults if desired,
            # or leave them as the user set them. For now, just re-enable start.

    @Slot()
    def handle_clustering_performed(self):
        logging.info(
            "ClustersPresenter received clustering_performed. Loading cluster cards."
        )
        if self.progress_info_bar:
            self.progress_info_bar.set_progress(100)
            self.progress_info_bar.set_title("Clustering Complete")
            self.progress_info_bar.set_content("Cluster analysis finished.")
            QTimer.singleShot(1500, self.progress_info_bar.customClose)
            self.progress_info_bar = None
        if self.control_panel:
            self.control_panel.startButton.setEnabled(True)

        # Reset flags after a successful run using current parameters
        self.dl_feature_config_changed = False
        self.morph_feature_config_changed = False
        self.dim_reduction_config_changed = False
        self.kmeans_config_changed = False
        if (
            self._current_run_ui_params
        ):  # Store the params that led to this successful run
            self._last_successful_run_params = self._current_run_ui_params.copy()
        self._current_run_ui_params = None  # Clear after use

        self.load_clusters()

    @Slot(int)
    def handle_feature_extraction_progress(self, percentage: int):
        if self.progress_info_bar and self.progress_info_bar.isVisible():
            if "Extracting Features" in self.progress_info_bar.titleLabel.text():
                self.progress_info_bar.set_progress(percentage)

    @Slot()
    def handle_features_extracted(
        self,
    ):  # This means DM finished self.data_manager_interface.extract_all_features()
        logging.info(
            "ClustersPresenter received features_extracted_all (DL features ready)."
        )
        if (
            self.progress_info_bar
            and "Extracting Features" in self.progress_info_bar.titleLabel.text()
        ):
            self.progress_info_bar.set_progress(100)
            self.progress_info_bar.set_title("Feature Extraction Complete")

        # DL features are now considered "up-to-date" with current model
        self.dl_feature_config_changed = False
        # If only DL was requested, and morph was not, then active sources might be considered stable for now
        if (
            self._pending_clustering_params_after_extraction
            and not self._pending_clustering_params_after_extraction.get(
                "feature_sources", {}
            ).get("morphological")
        ):
            self.morph_feature_config_changed = (
                False  # Or set based on if morph was part of this extraction request
            )

        if self._pending_clustering_params_after_extraction:
            logging.info("Proceeding with clustering after feature extraction.")
            if self.progress_info_bar and self.progress_info_bar.isVisible():
                self.progress_info_bar.set_title("Clustering")
                kmeans_settings = self._pending_clustering_params_after_extraction.get(
                    "kmeans_settings", {}
                )
                self.progress_info_bar.set_content(
                    f"Running K-means (K={kmeans_settings.get('n_clusters',10)})..."
                )
                self.progress_info_bar.set_progress(0)  # Reset for clustering stage

            self.proceed_with_clustering(
                self._pending_clustering_params_after_extraction
            )
            self._pending_clustering_params_after_extraction = (
                None  # Clear pending params
            )
        else:
            # Features extracted, but no pending clustering (e.g., user just wanted to extract)
            if (
                self.progress_info_bar
                and "Extracting Features" in self.progress_info_bar.titleLabel.text()
            ):
                QTimer.singleShot(1000, self.progress_info_bar.customClose)
                self.progress_info_bar = None
            InfoBar.success(
                "Features Ready",
                "Feature extraction complete. You can now start clustering.",
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
                duration=3000,
            )
            if self.control_panel:
                self.control_panel.startButton.setEnabled(True)

    # ... (handle_cluster_split, handle_cluster_merged remain the same)
    @Slot(str, list)
    def handle_cluster_split(self, original_cluster_id: str, new_cluster_ids: list):
        logging.info(
            f"ClustersPresenter received cluster_split. Original: {original_cluster_id}, New: {new_cluster_ids}"
        )
        if self.clusters_view_widget:
            self.clusters_view_widget.remove_cluster_card(original_cluster_id)
        self.selected_card_ids.discard(original_cluster_id)
        if new_cluster_ids:
            self.load_clusters(cluster_ids_to_load=new_cluster_ids)  # Load only new
        else:
            self.load_clusters()
        InfoBar.success(
            "Cluster Split",
            f"Cluster {original_cluster_id[:8]} split into {len(new_cluster_ids)} new clusters.",
            parent=(
                self.clusters_view_widget.window()
                if self.clusters_view_widget
                else None
            ),
            duration=3000,
        )

    @Slot(str, list)
    def handle_cluster_merged(self, new_cluster_id: str, old_cluster_ids: list):
        logging.info(
            f"ClustersPresenter received cluster_merged. New: {new_cluster_id}, Old: {old_cluster_ids}"
        )
        if self.clusters_view_widget:
            self.clusters_view_widget.clear_cluster_cards(
                old_cluster_ids
            )  # Pass IDs to clear specific cards
        self.selected_card_ids.difference_update(old_cluster_ids)
        self.last_clicked_card_id = (
            None  # Reset last clicked as it might have been one of the merged
        )
        self.load_clusters(cluster_ids_to_load=[new_cluster_id])
        InfoBar.success(
            "Merge Successful",
            f"Clusters merged into new cluster {new_cluster_id[:8]}.",
            duration=3000,
            parent=(
                self.clusters_view_widget.window()
                if self.clusters_view_widget
                else None
            ),
            position=InfoBarPosition.BOTTOM_RIGHT,
        )

    #  Methods Triggered by UI Actions
    @Slot(dict)
    def start_analysis_from_ui_params(self, current_ui_params: dict):
        if not self.data_manager_interface or not self.data_manager:
            logging.error("Cannot start analysis: DataManager or Interface not ready.")
            InfoBar.error(
                title="Error",
                content="Backend not fully initialized.",
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
            )
            return

        if self.control_panel:
            self.control_panel.startButton.setEnabled(False)
        self._current_run_ui_params = current_ui_params.copy()  # Store for this attempt

        #  Determine if feature re-extraction or reprocessing is needed
        feature_sources_config = current_ui_params.get("feature_sources", {})
        dl_model_config = current_ui_params.get("dl_feature_extractor_settings", {})

        needs_dl_feature_extraction = False
        if feature_sources_config.get("deep_learning"):
            # Reason 1: DL model in UI changed from DataManager's current processor
            current_dm_processor_model = (
                self.data_manager.processor.model_name
                if self.data_manager.processor
                else None
            )
            ui_selected_dl_model = dl_model_config.get("model_name")
            if (
                ui_selected_dl_model
                and current_dm_processor_model != ui_selected_dl_model
            ):
                logging.info(
                    f"DL Model changed (UI: {ui_selected_dl_model}, DM: {current_dm_processor_model}). Requesting processor update and feature re-extraction."
                )
                # This will trigger features_invalidated, which sets all change flags to True
                self.data_manager_interface.update_processor(
                    {
                        "selected_model": ui_selected_dl_model,
                        "provider": self.data_manager.settings.get("provider"),
                    }
                )
                # After update_processor, DM's features are cleared. We must re-extract.
                needs_dl_feature_extraction = True

            # Reason 2: DL features are selected, but DataManager has no _all_features_array
            elif self.data_manager._all_features_array is None:
                logging.info(
                    "DL features selected, but no features array in DataManager. Requesting extraction."
                )
                needs_dl_feature_extraction = True

            # Reason 3: DL config flag indicates a change (e.g., first run, or model changed via settings previously)
            elif self.dl_feature_config_changed:
                logging.info(
                    "DL feature configuration flag is set. Requesting re-extraction."
                )
                needs_dl_feature_extraction = True

        # If any feature source config changed (e.g., toggling morph on/off), features need reprocessing by DM.
        # This doesn't necessarily mean re-extracting DL features from images, but DM needs to combine them differently.
        # The _prepare_features_for_clustering in DM will handle this.
        # If DL features are needed for this new combination and aren't "current", needs_dl_feature_extraction handles it.

        if needs_dl_feature_extraction:
            if self.progress_info_bar and self.progress_info_bar.isVisible():
                self.progress_info_bar.customClose()
            self.progress_info_bar = ProgressInfoBar.new(
                icon=InfoBarIcon.INFORMATION,
                title="Extracting Features",
                content=f"Using model: {dl_model_config.get('model_name', 'N/A')}...",
                duration=-1,
                position=InfoBarPosition.BOTTOM_RIGHT,
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
            )
            self._pending_clustering_params_after_extraction = current_ui_params
            self.data_manager_interface.extract_all_features()
            # Clustering proceeds via handle_features_extracted
        elif not any(feature_sources_config.values()):
            logging.warning("No feature sources selected for clustering.")
            InfoBar.warning(
                "No Features",
                "Please select at least one feature source for clustering.",
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
            )
            if self.control_panel:
                self.control_panel.startButton.setEnabled(True)
        else:
            # Features are considered ready, or only non-DL features are used,
            # or only DR/KMeans params changed.
            logging.info(
                "Features seem up-to-date or not DL-based. Proceeding with clustering using current features."
            )
            self.proceed_with_clustering(current_ui_params)

    def proceed_with_clustering(self, ui_params_for_run: dict):
        if not self.data_manager_interface or not self.data_manager:
            # ... (existing error handling) ...
            return

        # ... (existing checks for feature sources and _all_features_array if DL is used) ...
        feature_sources = ui_params_for_run.get("feature_sources", {})
        if not any(feature_sources.values()):
            # ... (existing error handling) ...
            return
        if (
            feature_sources.get("deep_learning")
            and self.data_manager._all_features_array is None
        ):
            # ... (existing error handling) ...
            return

        if self.clusters_view_widget:
            self.clusters_view_widget.clear_cluster_cards()
        self.selected_card_ids.clear()

        # Add flags to ui_params_for_run so DataManager knows if it can reuse DR output
        # This is a more refined way than DataManager inferring everything.
        params_to_dm = ui_params_for_run.copy()
        params_to_dm["config_flags"] = {
            "dl_feature_config_changed": self.dl_feature_config_changed,
            "morph_feature_config_changed": self.morph_feature_config_changed,
            "dim_reduction_config_changed": self.dim_reduction_config_changed,
            "kmeans_config_changed": self.kmeans_config_changed,
        }
        # Also pass the model name that the features *should* correspond to
        if feature_sources.get("deep_learning"):
            params_to_dm["expected_dl_model_name"] = params_to_dm.get(
                "dl_feature_extractor_settings", {}
            ).get("model_name")

        kmeans_settings = params_to_dm.get("kmeans_settings", {})
        n_clusters = kmeans_settings.get("n_clusters", CLUSTERING_DEFAULT_N_CLUSTERS)

        if (
            not self.progress_info_bar
            or not self.progress_info_bar.isVisible()
            or "Extracting Features" in self.progress_info_bar.titleLabel.text()
        ):  # New bar if old was for features
            if self.progress_info_bar:
                self.progress_info_bar.customClose()
            self.progress_info_bar = ProgressInfoBar.new(
                icon=InfoBarIcon.INFORMATION,
                title="Clustering",
                content=f"Processing features and running K-means (K={n_clusters})...",
                duration=-1,
                position=InfoBarPosition.BOTTOM_RIGHT,
                parent=(
                    self.clusters_view_widget.window()
                    if self.clusters_view_widget
                    else None
                ),
            )
        else:
            self.progress_info_bar.set_title("Clustering")
            self.progress_info_bar.set_content(
                f"Processing features and running K-means (K={n_clusters})..."
            )
            self.progress_info_bar.set_progress(0)

        self.data_manager_interface.perform_clustering(params_to_dm)

        if (
            hasattr(self, "_pending_clustering_params_after_extraction")
            and self._pending_clustering_params_after_extraction == ui_params_for_run
        ):
            self._pending_clustering_params_after_extraction = None

    @Slot()
    def reset_analysis(self):
        logging.info("ClustersPresenter: Requesting cluster reset.")
        if self.data_manager_interface:
            self.data_manager_interface.clear_clusters()
            # Flags should also be reset as if it's a first run after clear
            self.dl_feature_config_changed = True
            self.morph_feature_config_changed = True
            self.dim_reduction_config_changed = True
            self.kmeans_config_changed = True
            self._last_successful_run_params = None
            if self.control_panel:
                self.control_panel.startButton.setEnabled(
                    True
                )  # Re-enable start button
        else:
            logging.error("Cannot reset analysis: DataManagerInterface not available.")

    @Slot(str)
    def on_model_selected_in_ui(
        self, model_name: str
    ):  # This is connected to ControlPanel's model selector
        # This method is now primarily to set the flag.
        # The actual DataManager.update_processor call happens in start_analysis_from_ui_params
        # if a model mismatch is detected.
        logging.debug(f"ClustersPresenter: UI selected DL model: {model_name}")
        self._on_dl_model_config_changed()  # Set the flag

    # ... (split_selected_cluster, merge_selected_clusters, assign_clusters_to_class, selection methods,
    #      show_cluster_viewer, show_summary, load_clusters, _generate_cluster_preview,
    #      update_model_selector, clear - these remain largely the same as in your provided full file,
    #      as their core logic of interacting with DataManager for reads or DataManagerInterface for writes
    #      is mostly compatible. Ensure that load_clusters clears and re-populates UI cards correctly.)

    #  Make sure these existing methods are present and correct
    def select_card(self, card_id: str):
        self.selected_card_ids.add(card_id)
        if self.clusters_view_widget:
            for card in self.clusters_view_widget.clusters:
                if card.cluster_id == card_id:
                    card.set_selected(True)
                    break

    def deselect_card(self, card_id: str):
        self.selected_card_ids.discard(card_id)
        if self.clusters_view_widget:
            for card in self.clusters_view_widget.clusters:
                if card.cluster_id == card_id:
                    card.set_selected(False)
                    break

    def clear_selection(self):
        if self.clusters_view_widget:
            for card_id_to_deselect in list(self.selected_card_ids):
                for card in self.clusters_view_widget.clusters:
                    if card.cluster_id == card_id_to_deselect:
                        card.set_selected(False)
                        break
        self.selected_card_ids.clear()

    @Slot(str, Qt.KeyboardModifiers, Qt.MouseButton)
    def on_card_clicked(
        self, card_id: str, modifiers: Qt.KeyboardModifiers, button: Qt.MouseButton
    ):
        if button == Qt.RightButton:
            if not self.selected_card_ids or card_id not in self.selected_card_ids:
                self.clear_selection()
                self.select_card(card_id)
        elif button == Qt.LeftButton:
            is_ctrl = modifiers == Qt.ControlModifier or (
                os.name == "posix" and modifiers == Qt.MetaModifier
            )
            is_shift = modifiers == Qt.ShiftModifier
            if is_ctrl:
                if card_id in self.selected_card_ids:
                    self.deselect_card(card_id)
                else:
                    self.select_card(card_id)
            elif is_shift:
                logging.warning(
                    "Shift-click range selection not yet fully implemented for clusters."
                )
                self.clear_selection()
                self.select_card(card_id)
            else:
                self.clear_selection()
                self.select_card(card_id)
        self.last_clicked_card_id = card_id

    @Slot(bool)
    def set_ctrl_pressed(self, pressed: bool):
        self.ctrl_pressed = pressed

    @Slot(str)
    def show_cluster_viewer(self, cluster_id: str):
        if not self.data_manager or not self.clusters_view_widget:
            return
        cluster = self.data_manager.get_cluster(cluster_id)
        if not cluster:
            logging.error(f"Cannot show viewer: Cluster {cluster_id} not found.")
            return

        viewer = ClassClusterViewer(
            f"Cluster: {cluster.id[:8]}", self, self.clusters_view_widget.window()
        )
        active_samples_in_cluster_count = 0
        for sample in cluster.samples:
            if (
                sample.id in self.data_manager.active_samples
            ):  # Check against global active samples
                viewer.add_card(sample.id)
                active_samples_in_cluster_count += 1
        logging.info(
            f"Showing viewer for cluster: {cluster.id[:8]} with {active_samples_in_cluster_count} active samples."
        )
        viewer.show()

    @Slot(str)
    def show_summary(self, cluster_id: str):
        if not self.data_manager or not self.clusters_view_widget:
            return
        cluster = self.data_manager.get_cluster(cluster_id)
        if not cluster:
            logging.error(f"Cannot show summary: Cluster {cluster_id} not found.")
            return

        parameter_data = {}
        property_keys = [
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

        session_units = (
            self.data_manager.session.scale_units if self.data_manager.session else None
        )

        active_samples_for_summary_count = 0
        # Store actual attribute values for statistics
        temp_values_for_stats: Dict[str, List[float]] = {
            key: [] for key in property_keys
        }
        # Keep track of sample IDs processed for the count to avoid double counting
        processed_sample_ids_for_count: Set[str] = set()

        for sample in cluster.samples:
            if (
                sample.id in self.data_manager.active_samples
                and sample.mask_id
                and sample.mask_id in self.data_manager.masks
            ):

                # Count each active, masked sample once for the total
                if sample.id not in processed_sample_ids_for_count:
                    active_samples_for_summary_count += 1
                    processed_sample_ids_for_count.add(sample.id)

                mask_attrs = self.data_manager.masks[sample.mask_id].attributes
                for param_key in property_keys:
                    if param_key in mask_attrs:
                        try:
                            temp_values_for_stats[param_key].append(
                                float(mask_attrs[param_key])
                            )
                        except (ValueError, TypeError):
                            pass

        for param_key, values_list in temp_values_for_stats.items():
            if values_list:
                display_key = param_key.replace("_", " ").title()
                if session_units:
                    if "area" == param_key:
                        display_key += f" ({session_units}²)"
                    elif param_key in [
                        "perimeter",
                        "major_axis_length",
                        "minor_axis_length",
                        "curl",
                    ]:  # Added curl
                        display_key += f" ({session_units})"
                    elif "volume" == param_key:
                        display_key += f" ({session_units}³)"
                try:
                    parameter_data[display_key] = (
                        np.mean(values_list),
                        np.std(values_list),
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not calc stats for {param_key} in cluster {cluster_id}: {e}"
                    )

        summary_window = ClassClusterSummary(
            f"Cluster Summary: {cluster.id[:8]}",
            parent=self.clusters_view_widget.window(),
        )
        summary_window.set_summary_data(
            cluster.id[:8], active_samples_for_summary_count, parameter_data
        )
        summary_window.show()

    def load_clusters(self, cluster_ids_to_load: Optional[List[str]] = None):
        if not self.data_manager or not self.clusters_view_widget:
            logging.warning("Cannot load clusters: DataManager or View not available.")
            return

        clusters_to_display_map: Dict[str, Any] = {}
        if cluster_ids_to_load is None:  # Load all active clusters
            logging.info("ClustersPresenter: Loading all clusters into UI.")
            self.clusters_view_widget.clear_cluster_cards()
            self.selected_card_ids.clear()
            clusters_to_display_map = (
                self.data_manager.clusters
            )  # This contains Cluster objects
        else:  # Load specific clusters
            logging.info(
                f"ClustersPresenter: Loading specific clusters into UI: {cluster_ids_to_load}"
            )
            for cid in cluster_ids_to_load:
                cluster = self.data_manager.get_cluster(cid)
                if cluster:
                    clusters_to_display_map[cid] = cluster
                else:
                    logging.warning(
                        f"Cluster {cid} requested for load, but not found in DataManager."
                    )

        if not clusters_to_display_map:
            logging.info("ClustersPresenter: No cluster data to display.")
            if cluster_ids_to_load is None:
                self.clusters_view_widget.clear_cluster_cards()
                self.selected_card_ids.clear()
            return

        for cluster_id, cluster_obj in clusters_to_display_map.items():
            if (
                cluster_ids_to_load is not None
                and self.clusters_view_widget.find_cluster_card(cluster_id)
            ):
                logging.debug(
                    f"Card for cluster {cluster_id} already exists. Skipping creation."
                )
                continue  # Avoid duplicates if loading specific and card already there

            preview_path = self._generate_cluster_preview(cluster_id)
            card = self.clusters_view_widget.create_cluster_card(
                cluster_id, preview_path
            )
            if card:  # create_cluster_card might return None if path is bad
                card.cluster_color = cluster_obj.color
                card.clusters_presenter = self  # Ensure card has presenter reference
                # Reconnect signals for new cards (important if cards are recreated)
                try:
                    card.card_clicked.disconnect()
                except RuntimeError:
                    pass
                card.card_clicked.connect(self.on_card_clicked)

                try:
                    card.cluster_double_clicked.disconnect()
                except RuntimeError:
                    pass
                card.cluster_double_clicked.connect(self.show_cluster_viewer)
                card.merge_requested.connect(
                    self.merge_selected_clusters
                )  # DragDrop merge

        if self.clusters_view_widget:
            self.clusters_view_widget.clustersGallery.flow_layout.update()

    @Slot(list)
    def merge_selected_clusters(
        self, selected_ids_list: list
    ):  # From context menu or drag/drop
        if not self.data_manager_interface:
            logging.error("Cannot merge clusters: DataManagerInterface not available.")
            return

        cluster_ids_to_merge = list(selected_ids_list)  # Ensure it's a list copy
        if len(cluster_ids_to_merge) < 2:
            InfoBar.warning(
                "Merge Error",
                "Please select at least two clusters to merge.",
                parent=self.clusters_view_widget.window(),
            )
            return
        logging.info(f"Requesting merge for clusters: {cluster_ids_to_merge}")
        self.data_manager_interface.merge_clusters(cluster_ids_to_merge)

    @Slot()
    def split_selected_cluster(self):
        if not self.data_manager_interface:
            logging.error("Cannot split cluster: DataManagerInterface not available.")
            return
        if len(self.selected_card_ids) != 1:
            InfoBar.warning(
                "Split Error",
                "Please select exactly one cluster to split.",
                parent=self.clusters_view_widget.window(),
            )
            return

        cluster_id_to_split = list(self.selected_card_ids)[0]

        # Use K value from the K-Means settings in ControlPanel for the number of sub-clusters
        n_sub_clusters = (
            self.control_panel.clusters_slider.value() if self.control_panel else 2
        )  # Default to 2 if panel not found
        # Default iter/redo for split, can be made configurable later
        n_iter = self.data_manager.settings.get("CLUSTERING_N_ITER_SPLIT", 100)
        n_redo = self.data_manager.settings.get("CLUSTERING_N_REDO_SPLIT", 5)

        logging.info(
            f"Requesting split for cluster: {cluster_id_to_split} into {n_sub_clusters} sub-clusters."
        )
        self.data_manager_interface.split_cluster(
            cluster_id_to_split, n_sub_clusters, n_iter, n_redo
        )

    @Slot(str)  # class_name
    def assign_clusters_to_class(self, class_name: str):
        if not self.data_manager_interface or not self.data_manager:
            logging.error(
                "Cannot assign class: DataManager or Interface not available."
            )
            return

        selected_c_ids = list(self.selected_card_ids)
        if not selected_c_ids:
            logging.warning(
                "Assign clusters to class called with no clusters selected."
            )
            return

        class_object = self.data_manager.get_class_by_name(class_name)
        if not class_object:
            logging.error(f"ClustersPresenter: Class '{class_name}' not found.")
            InfoBar.error(
                title="Error",
                content=f"Class '{class_name}' not found.",
                parent=self.clusters_view_widget.window(),
            )
            return

        all_image_ids_to_assign = []
        for c_id in selected_c_ids:
            cluster = self.data_manager.get_cluster(c_id)
            if cluster:
                all_image_ids_to_assign.extend(
                    [sample.id for sample in cluster.samples if sample.is_active]
                )

        if not all_image_ids_to_assign:
            logging.warning(
                "No active images found in the selected clusters to assign."
            )
            return

        logging.info(
            f"Requesting assignment of {len(all_image_ids_to_assign)} images to class {class_object.id} ({class_name})."
        )
        self.data_manager_interface.assign_images_to_class(
            all_image_ids_to_assign, class_object.id
        )
        self.clear_selection()
        InfoBar.success(
            title="Assignment Queued",
            content=f"Assignment of images from {len(selected_c_ids)} cluster(s) to class '{class_name}' processing.",
            duration=3000,
            parent=self.clusters_view_widget.window(),
        )

    def _generate_cluster_preview(self, cluster_id: str) -> Optional[str]:
        if not self.data_manager:
            return None

        cluster = self.data_manager.get_cluster(cluster_id)
        if not cluster or not cluster.samples:
            return None

        active_sample_paths_in_cluster = []
        for sample_in_cluster in cluster.samples:
            # Check if the sample is in the DataManager's global active_samples list
            if (
                sample_in_cluster.id in self.data_manager.active_samples
                and sample_in_cluster.path
                and os.path.exists(sample_in_cluster.path)
            ):
                active_sample_paths_in_cluster.append(sample_in_cluster.path)

        # Log the count
        logging.debug(
            f"Cluster '{cluster_id[:8]}': Found {len(active_sample_paths_in_cluster)} active samples for preview (out of {len(cluster.samples)} total in cluster)."
        )

        if not active_sample_paths_in_cluster:
            return None  # No active samples in this cluster for preview

        image_paths_for_collage = active_sample_paths_in_cluster[
            : self.images_per_preview
        ]
        collage_image = merge_images_collage(
            image_paths_for_collage, scale=COLLAGE_RES_SCALE
        )

        if collage_image:
            # Use session-specific temp folder
            temp_preview_dir = os.path.join(
                self.data_manager.session.metadata_directory, "cluster_previews_temp"
            )
            os.makedirs(temp_preview_dir, exist_ok=True)
            preview_path = os.path.join(
                temp_preview_dir, f"cluster_preview_{cluster_id}.png"
            )
            try:
                collage_image.save(preview_path)
                return preview_path
            except Exception as e:
                logging.error(f"Failed to save cluster preview {preview_path}: {e}")
        return None

    @Slot()
    def update_model_selector(self):
        if self.control_panel and self.data_manager:
            available_models_dict = get_available_models()
            current_model_name = self.data_manager.settings.get("selected_model", "")
            if (
                current_model_name not in available_models_dict
                and available_models_dict
            ):
                current_model_name = list(available_models_dict.keys())[0]
            elif not available_models_dict:
                current_model_name = ""
            self.control_panel.populate_models(
                available_models_dict, current_model_name
            )

    def clear(self) -> None:
        logging.info("Clearing ClustersPresenter state.")
        if self.control_panel:
            try:
                self.control_panel.analysis_requested.disconnect(
                    self.start_analysis_from_ui_params
                )
                self.control_panel.reset_analysis_requested.disconnect(
                    self.reset_analysis
                )
                if hasattr(self.control_panel, "extractor_model_selector"):
                    self.control_panel.extractor_model_selector.currentTextChanged.disconnect(
                        self._on_dl_model_config_changed
                    )
                # Disconnect other new signals from control panel params
                self.control_panel.cb_use_dl_features.stateChanged.disconnect(
                    self._on_feature_source_config_changed
                )
                self.control_panel.cb_use_morph_features.stateChanged.disconnect(
                    self._on_feature_source_config_changed
                )
                self.control_panel.dim_reduction_method_selector.currentTextChanged.disconnect(
                    self._on_dim_reduction_config_changed
                )
                # ... disconnect DR param widgets ...
                if hasattr(
                    self.control_panel.pca_params_widget, "n_components_spinbox"
                ):
                    self.control_panel.pca_params_widget.n_components_spinbox.valueChanged.disconnect(
                        self._on_dim_reduction_config_changed
                    )
                if hasattr(
                    self.control_panel.umap_params_widget, "n_neighbors_spinbox"
                ):
                    self.control_panel.umap_params_widget.n_neighbors_spinbox.valueChanged.disconnect(
                        self._on_dim_reduction_config_changed
                    )
                    self.control_panel.umap_params_widget.min_dist_spinbox.valueChanged.disconnect(
                        self._on_dim_reduction_config_changed
                    )
                    self.control_panel.umap_params_widget.n_components_umap_spinbox.valueChanged.disconnect(
                        self._on_dim_reduction_config_changed
                    )
                if hasattr(self.control_panel, "tsne_params_widget") and hasattr(
                    self.control_panel.tsne_params_widget, "perplexity_spinbox"
                ):
                    self.control_panel.tsne_params_widget.perplexity_spinbox.valueChanged.disconnect(
                        self._on_dim_reduction_config_changed
                    )  # etc. for all TSNE params
                self.control_panel.clusters_slider.valueChanged.disconnect(
                    self._on_kmeans_config_changed
                )

            except (AttributeError, TypeError, RuntimeError) as e:
                logging.warning(
                    f"Error disconnecting ControlPanel signals in clear: {e}"
                )

        if self.control_helper:
            try:
                self.control_helper.ctrl_signal.disconnect(self.set_ctrl_pressed)
            except (TypeError, RuntimeError):
                pass
            self.control_helper.deleteLater()
        if self.progress_info_bar:
            self.progress_info_bar.close()
        if self.clusters_view_widget:
            self.clusters_view_widget.clear_cluster_cards()

        self.selected_card_ids.clear()
        self.data_manager_interface = None
        self.data_manager = None
        self.clusters_view_widget = None
        self.control_panel = None
        self.context_menu_handler = None
        self.progress_info_bar = None
        self._last_successful_run_params = None
        self._pending_clustering_params_after_extraction = None
        self._current_run_ui_params = None
        logging.info("ClustersPresenter cleared successfully.")
