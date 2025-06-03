### backend\backend_initializer.py

import logging
from typing import Any, Optional

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer
from PySide6.QtWidgets import QMessageBox, QWidget
from qfluentwidgets import InfoBar, InfoBarPosition

from backend.config import load_settings
from backend.data_manager import DataManager
from backend.data_manager_interface import DataManagerInterface
from backend.objects.session import Session
from backend.presenters.analysis_presenter import AnalysisPresenter
from backend.presenters.classes_presenter import ClassesPresenter
from backend.presenters.clusters_presenter import ClustersPresenter
from backend.presenters.gallery_presenter import GalleryPresenter
from backend.presenters.segmentation_presenter import SegmentationPresenter
from backend.presenters.sessions_presenter import SessionPresenter
from backend.segmentation import SegmentationModel
from backend.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BackendInitializer(QObject):
    def __init__(self, workspace: QWidget):
        super().__init__()
        self.settings = load_settings()

        self.session_manager: SessionManager = SessionManager()
        self.session_presenter: Optional[SessionPresenter] = None
        self.active_session: Optional[Session] = None
        self.workspace: QWidget = workspace

        self.segmentation_model_instance: SegmentationModel = SegmentationModel()

        self.data_manager: Optional[DataManager] = None
        self.data_manager_thread: Optional[QThread] = None
        self.data_manager_interface: Optional[DataManagerInterface] = None

        self.gallery_presenter: Optional[GalleryPresenter] = None
        self.classes_presenter: Optional[ClassesPresenter] = None
        self.clusters_presenter: Optional[ClustersPresenter] = None
        self.analysis_presenter: Optional[AnalysisPresenter] = None
        self.segmentation_presenter: Optional[SegmentationPresenter] = None

    def apply_settings(self):
        logging.info("Applying settings changes...")
        old_settings = self.settings.copy()
        self.settings = load_settings()  # Reload settings from file

        model_changed = old_settings.get("selected_model") != self.settings.get(
            "selected_model"
        )
        provider_changed = old_settings.get("provider") != self.settings.get("provider")

        if (
            self.data_manager_interface
            and self.data_manager_thread
            and self.data_manager_thread.isRunning()
        ):
            if model_changed or provider_changed:
                logging.info(
                    "Model or provider changed, requesting DataManager's processor update..."
                )
                # DataManager will handle invalidating features/clusters if model changed
                self.data_manager_interface.update_processor(self.settings.copy())
                if model_changed:
                    InfoBar.warning(
                        title="Feature Model Changed",
                        content="The feature extraction model has changed. Existing features and clusters have been invalidated and will need to be re-calculated.",
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP_RIGHT,
                        duration=2000,
                        parent=self.workspace.window(),
                    )
        elif model_changed or provider_changed:
            logging.warning(
                "Settings changed (model/provider), but no active DataManager to update."
            )

        # Update presenters that might depend on settings
        if self.clusters_presenter:
            self.clusters_presenter.update_model_selector()  # Reads from DataManager.settings
        if self.gallery_presenter:
            self.gallery_presenter.thumbnail_quality = self.settings.get(
                "thumbnail_quality", 75
            )
        if self.classes_presenter:
            self.classes_presenter.images_per_preview = self.settings.get(
                "images_per_collage", 25
            )

        if hasattr(self.workspace.window(), "setQss"):  # Apply theme
            self.workspace.window().setQss()
        logging.info("Settings applied.")

    def on_session_chosen(self, session_id: str) -> None:
        logging.info(f"Session chosen with ID: {session_id}")

        if (
            self.active_session
            and self.active_session.id == session_id
            and self.data_manager
        ):
            logging.info(f"Session {session_id} is already active. No change needed.")
            if not self.workspace.isEnabled():
                self.workspace.setEnabled(True)
            return

        if self.active_session:  # If a different session was active
            self._cleanup_current_session_resources()

        chosen_session_obj = self.session_manager.open_session(
            session_id
        )  # Updates last_opened
        if not chosen_session_obj:
            logging.error(
                f"Failed to open session with ID {session_id} via SessionManager."
            )
            QMessageBox.critical(
                self.workspace.window(),
                "Error",
                f"Could not open session {session_id}.",
            )
            return

        self.active_session = chosen_session_obj
        self.settings = load_settings()  # Reload settings, could have changed globally

        #  Create DataManager in its Thread
        self.data_manager_thread = QThread(
            self
        )  # Parent thread to self for proper cleanup
        try:
            self.data_manager = DataManager(
                session=self.active_session,
                settings=self.settings,
                segmentation_instance=self.segmentation_model_instance,
            )
        except Exception as e:
            logging.critical(
                f"Failed to initialize DataManager for session {session_id}: {e}",
                exc_info=True,
            )
            QMessageBox.critical(
                self.workspace.window(),
                "Initialization Error",
                f"Failed to initialize backend for session:\n{e}\n\nPlease check configuration and paths.",
            )
            self.active_session = None
            if self.data_manager_thread:
                self.data_manager_thread.quit()
                self.data_manager_thread.wait()
            self.data_manager_thread = None
            return

        self.data_manager.moveToThread(self.data_manager_thread)
        self.data_manager_thread.finished.connect(
            self.data_manager.deleteLater
        )  # Ensure DM is deleted when thread finishes

        # Signal from DM (worker thread) to BI (main thread) after data is loaded
        self.data_manager.session_data_loaded.connect(
            self.initialize_presenters_and_ui, Qt.QueuedConnection
        )
        self.data_manager_thread.start()

        #  Create Interface (lives in main thread)
        self.data_manager_interface = DataManagerInterface()
        self._connect_interface_to_manager()

        #  Request initial load via Interface (signal to DM in its thread)
        logging.info("Requesting initial session data load via DataManagerInterface.")
        self.data_manager_interface.load_session()

    def initialize_presenters_and_ui(self):
        """Initializes presenters, connects signals, and enables UI after DM loaded data."""
        if (
            not self.data_manager
            or not self.data_manager_interface
            or not self.active_session
        ):
            logging.warning(
                "Initialization of presenters skipped: DataManager, Interface or active_session no longer available."
            )
            if self.workspace:
                self.workspace.setEnabled(False)  # Ensure UI is disabled
            return

        logging.info(f"Initializing presenters for session: {self.active_session.name}")
        try:
            # Pass interface for actions, and DataManager instance for direct reads by presenters
            self.gallery_presenter = GalleryPresenter(
                self.workspace.galleryView,
                self.data_manager_interface,
                self.data_manager,
            )
            self.workspace.galleryView.set_presenter(self.gallery_presenter)

            self.classes_presenter = ClassesPresenter(
                self.workspace.classesView,
                self.data_manager_interface,
                self.data_manager,
            )
            self.workspace.classesView.set_presenter(self.classes_presenter)

            self.clusters_presenter = ClustersPresenter(
                self.workspace.clustersView,
                self.data_manager_interface,
                self.data_manager,
                self.workspace.clustersView.controlPanel,
            )
            self.workspace.clustersView.set_presenter(self.clusters_presenter)

            self.analysis_presenter = AnalysisPresenter(
                self.workspace.analysisView,
                self.data_manager_interface,
                self.data_manager,
                self.segmentation_model_instance,
            )
            self.workspace.analysisView.set_presenter(self.analysis_presenter)

            self.segmentation_presenter = SegmentationPresenter(
                self.workspace.segmentationView,
                self.data_manager_interface,
                self.data_manager,
                self.segmentation_model_instance,
            )
            self.workspace.segmentationView.set_presenter(self.segmentation_presenter)

            self._connect_manager_to_presenters()

            # Trigger initial UI population in presenters
            if self.gallery_presenter:
                self.gallery_presenter.load_gallery()  # Uses DM directly for initial load
            if self.classes_presenter:
                self.classes_presenter.load_classes()

            if self.clusters_presenter:
                self.clusters_presenter.load_clusters()
                QTimer.singleShot(100, self.clusters_presenter.update_model_selector)
            if (
                self.segmentation_presenter
            ):  # Request initial samples for segmentation preview
                self.segmentation_presenter.on_samples_ready(
                    self.data_manager.samples
                )  # Pass current samples
            if self.analysis_presenter:
                self.analysis_presenter.load_plots_and_gates()

            self.workspace.pivot.setCurrentItem(self.workspace.galleryView.objectName())
            self.workspace.stackedWidget.setCurrentWidget(self.workspace.galleryView)
            # self.workspace.galleryView.reset_ui_elements() # Consider if this is needed or handled by presenter
            self.workspace.setEnabled(True)
            logging.info("Presenters initialized and UI enabled.")

        except Exception as e:
            logging.error(f"Error during presenter initialization: {e}", exc_info=True)
            QMessageBox.critical(
                self.workspace.window(),
                "Presenter Error",
                f"Failed to initialize UI components: {e}",
            )
            self._cleanup_current_session_resources()  # Attempt cleanup on error
            self.workspace.setEnabled(False)

    def _connect_interface_to_manager(self):
        """Connect signals from Interface (main thread) to DataManager slots (worker thread)."""
        if not self.data_manager or not self.data_manager_interface:
            return
        logging.debug("Connecting DataManagerInterface signals to DataManager slots...")
        intf = self.data_manager_interface
        mgr = self.data_manager
        conn_type = Qt.QueuedConnection

        # Existing connections
        intf.request_load_session.connect(mgr._on_load_session_requested, conn_type)
        intf.request_load_images_from_folder.connect(
            mgr._on_load_images_from_folder_requested, conn_type
        )
        intf.request_samples.connect(mgr._on_request_samples_requested, conn_type)
        intf.request_extract_all_features.connect(
            mgr._on_extract_all_features_requested, conn_type
        )
        intf.request_extract_feature.connect(
            mgr._on_extract_feature_requested, conn_type
        )
        intf.request_delete_features.connect(
            mgr._on_delete_features_requested, conn_type
        )
        intf.request_perform_clustering.connect(
            mgr._on_perform_clustering_requested, conn_type
        )
        intf.request_split_cluster.connect(mgr._on_split_cluster_requested, conn_type)
        intf.request_merge_clusters.connect(mgr._on_merge_clusters_requested, conn_type)
        intf.request_clear_clusters.connect(mgr._on_clear_clusters_requested, conn_type)
        intf.request_delete_cluster.connect(mgr._on_delete_cluster_requested, conn_type)
        intf.request_create_class.connect(mgr._on_create_class_requested, conn_type)
        intf.request_rename_class.connect(mgr._on_rename_class_requested, conn_type)
        intf.request_assign_images_to_class.connect(
            mgr._on_assign_images_to_class_requested, conn_type
        )
        intf.request_delete_class.connect(mgr._on_delete_class_requested, conn_type)
        intf.request_process_new_mask.connect(
            mgr._on_process_new_mask_requested, conn_type
        )
        intf.request_create_plot.connect(mgr._on_create_plot_requested, conn_type)
        intf.request_delete_plot.connect(mgr._on_delete_plot_requested, conn_type)
        intf.request_create_gate.connect(mgr._on_create_gate_requested, conn_type)
        intf.request_update_gate.connect(mgr._on_update_gate_requested, conn_type)
        intf.request_delete_gate.connect(mgr._on_delete_gate_requested, conn_type)
        intf.request_delete_mask.connect(mgr._on_delete_mask_requested, conn_type)
        intf.request_update_processor.connect(
            mgr._on_update_processor_requested, conn_type
        )
        intf.request_export_data.connect(mgr._on_export_data_requested, conn_type)
        intf.request_set_session_scale.connect(
            mgr._on_set_session_scale_requested, conn_type
        )
        intf.request_flush_db.connect(mgr._on_flush_db_requested, conn_type)
        logging.debug("DataManagerInterface signals connected.")

    def _connect_manager_to_presenters(self):
        """Connect signals FROM DataManager (worker) TO Presenters (main)."""
        if not self.data_manager:
            return
        logging.debug("Connecting DataManager result signals to Presenter slots...")
        mgr = self.data_manager
        conn_type = Qt.QueuedConnection  # Ensure slots run in presenter's (main) thread

        # Gallery Presenter
        if self.gallery_presenter:
            mgr.active_samples_updated.connect(
                self.gallery_presenter.on_active_samples_updated, conn_type
            )
            mgr.class_updated.connect(
                self.gallery_presenter.on_class_updated, conn_type
            )
            mgr.mask_created_in_db.connect(
                self.gallery_presenter.on_mask_created_or_updated, conn_type
            )
            mgr.mask_deleted_from_db.connect(
                self.gallery_presenter.on_mask_deleted, conn_type
            )

        # Classes Presenter
        if self.classes_presenter:
            mgr.class_added.connect(self.classes_presenter.on_class_added, conn_type)
            mgr.class_updated.connect(
                self.classes_presenter.on_class_updated, conn_type
            )
            mgr.class_deleted.connect(
                self.classes_presenter.on_class_deleted, conn_type
            )
            mgr.classifier_training_progress.connect(
                self.classes_presenter.handle_classifier_training_progress, conn_type
            )
            mgr.classifier_training_finished.connect(
                self.classes_presenter.handle_classifier_training_finished, conn_type
            )
            mgr.classification_run_finished.connect(
                self.classes_presenter.handle_classification_run_finished, conn_type
            )
            mgr.active_samples_updated.connect(
                self.classes_presenter.on_active_samples_changed, conn_type
            )

        # Clusters Presenter
        if self.clusters_presenter:
            mgr.features_invalidated.connect(
                self.clusters_presenter.handle_features_invalidated, conn_type
            )
            mgr.clusters_invalidated.connect(
                self.clusters_presenter.handle_clusters_invalidated, conn_type
            )
            mgr.clustering_performed.connect(
                self.clusters_presenter.handle_clustering_performed, conn_type
            )
            mgr.features_extracted_all.connect(
                self.clusters_presenter.handle_features_extracted, conn_type
            )
            mgr.feature_extraction_progress.connect(
                self.clusters_presenter.handle_feature_extraction_progress, conn_type
            )
            mgr.cluster_split.connect(
                self.clusters_presenter.handle_cluster_split, conn_type
            )
            mgr.cluster_merged.connect(
                self.clusters_presenter.handle_cluster_merged, conn_type
            )

        # Segmentation Presenter
        if self.segmentation_presenter:
            mgr.samples_ready.connect(
                self.segmentation_presenter.on_samples_ready, conn_type
            )
            # SegmentationPresenter primarily sends requests or gets data via direct calls after initial load.
            # It might listen to mask_created_in_db if it needs to update its view of available masks.
            mgr.mask_created_in_db.connect(
                self.segmentation_presenter.on_external_mask_change, conn_type
            )
            mgr.mask_deleted_from_db.connect(
                self.segmentation_presenter.on_external_mask_change, conn_type
            )

        # Analysis Presenter
        if self.analysis_presenter:
            mgr.plot_added_to_dm.connect(
                self.analysis_presenter.on_plot_added_to_dm, conn_type
            )
            mgr.plot_deleted_from_dm.connect(
                self.analysis_presenter.on_plot_deleted_from_dm, conn_type
            )
            mgr.gate_added_to_dm.connect(
                self.analysis_presenter.on_gate_added_to_dm, conn_type
            )
            mgr.gate_updated_in_dm.connect(
                self.analysis_presenter.on_gate_updated_in_dm, conn_type
            )
            mgr.gate_deleted_from_dm.connect(
                self.analysis_presenter.on_gate_deleted_from_dm, conn_type
            )
            mgr.all_gate_populations_recalculated.connect(
                self.analysis_presenter.on_all_gate_populations_recalculated_from_dm,
                conn_type,
            )

        # Export Finished signal (e.g., to show InfoBar in main window)
        if hasattr(
            self.workspace.window(), "show_export_result"
        ):  # Check if method exists
            mgr.export_finished.connect(
                self.workspace.window().show_export_result, conn_type
            )
        logging.debug("DataManager result signals connected to presenters.")

    def _cleanup_current_session_resources(self) -> None:
        """Cleans up resources for the currently active session before switching or closing."""
        logging.info(
            f"Cleaning up resources for session: {self.active_session.id if self.active_session else 'None'}"
        )

        # Disconnect signals to prevent calls to deleted objects
        if self.data_manager_interface and self.data_manager:
            logging.debug("Disconnecting DataManagerInterface from DataManager...")
            pass

        if self.data_manager:
            logging.debug("Disconnecting DataManager from Presenters...")
            pass

        # Schedule deletion of presenters (QObjects, should be deleted on main thread)
        presenters_to_clean = [
            ("gallery_presenter", self.gallery_presenter),
            ("classes_presenter", self.classes_presenter),
            ("clusters_presenter", self.clusters_presenter),
            ("analysis_presenter", self.analysis_presenter),
            ("segmentation_presenter", self.segmentation_presenter),
        ]
        for name, presenter_instance in presenters_to_clean:
            if presenter_instance:
                if hasattr(presenter_instance, "clear"):  # Call clear method if exists
                    try:
                        presenter_instance.clear()
                    except Exception as e:
                        logging.error(f"Error calling clear() on {name}: {e}")
                presenter_instance.deleteLater()
                setattr(self, name, None)
                logging.debug(f"Scheduled deletion for {name}.")

        # Handle DataManager and its thread
        if self.data_manager_thread:
            if self.data_manager and self.data_manager_thread.isRunning():
                logging.info("Requesting DataManager to close DB and persist data...")
                QMetaObject.invokeMethod(
                    self.data_manager, "close_db", Qt.BlockingQueuedConnection
                )

                logging.info("Requesting DataManager thread to quit...")
                self.data_manager_thread.quit()
                if not self.data_manager_thread.wait(5000):  # Wait up to 5s
                    logging.warning(
                        "DataManager thread did not quit gracefully, terminating."
                    )
                    self.data_manager_thread.terminate()
                    self.data_manager_thread.wait()  # Wait for termination
                else:
                    logging.info("DataManager thread quit gracefully.")

            # self.data_manager is scheduled for deleteLater via thread.finished connection
            self.data_manager = None
            self.data_manager_thread = None  # Clear ref to thread object

        # Delete interface (lives in main thread)
        if self.data_manager_interface:
            self.data_manager_interface.deleteLater()
            self.data_manager_interface = None
            logging.debug("DataManagerInterface scheduled for deletion.")

        self.active_session = None  # Clear active session reference
        logging.info("Session resources cleanup complete.")

    def init_sessions_presenter(self, session_view_widget: QWidget) -> None:
        """Initializes or re-initializes the SessionPresenter."""
        if self.session_presenter:  # If one exists, disconnect and delete
            try:
                self.session_presenter.session_chosen.disconnect(self.on_session_chosen)
            except RuntimeError:
                pass  # Already disconnected or deleted
            self.session_presenter.deleteLater()

        self.session_presenter = SessionPresenter(
            self.session_manager, session_view_widget, self
        )  # Pass self (BackendInitializer)
        self.session_presenter.session_chosen.connect(self.on_session_chosen)
        logging.info("Initialized SessionPresenter.")

    def on_current_session_deleted_by_user(self):
        """
        Called by SessionPresenter when the *currently active* session is deleted by the user
        from the session management dialog.
        """
        logging.info(
            "Current active session was deleted by user. Cleaning up UI and backend state."
        )
        if (
            self.active_session
        ):  # Should always be true if this is called for current session
            self._cleanup_current_session_resources()
            if self.workspace:
                self.workspace.setEnabled(False)  # Disable workspace UI
            logging.info(
                f"Cleanup complete after deletion of active session. Workspace disabled."
            )
        else:
            logging.warning(
                "on_current_session_deleted_by_user called but no session was marked active in BackendInitializer."
            )
