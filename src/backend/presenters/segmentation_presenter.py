# backend/presenters/segmentation_presenter.py
import json  # For serializing attributes if needed, though DataManager handles it
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from qfluentwidgets import InfoBarIcon

# from UI.navigation_interface.workspace.views.segmentation.segmentation_view_widget import SegmentationViewWidget # Type hint
from backend.data_manager_interface import DataManagerInterface  # Import interface

# from backend.data_manager import DataManager # For type hint of data_manager_for_reads
from backend.helpers.segmentation_thread import SegmentationThread
from UI.dialogs.progress_infobar import ProgressInfoBar

# from backend.segmentation import SegmentationModel # For type hint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SegmentationPresenter(QObject):
    """Presenter for the Segmentation View."""

    # Signal emitted when a full segmentation run (applying to all/selected and saving) is complete
    # and data has been requested to be persisted.
    full_segmentation_run_finished = Signal()

    # Note: request_create_mask and request_flush_db signals are removed from here.
    # This presenter will now use DataManagerInterface.request_process_new_mask and DataManagerInterface.flush_db

    def __init__(
        self,
        segmentation_view_widget_ref: Any,  # Keep as Any to avoid circular import
        data_manager_interface: DataManagerInterface,
        data_manager_for_reads: Any,  # Actual type: DataManager
        segmentation_model_instance: Any,
    ):  # Actual type: SegmentationModel
        super().__init__()
        self.segmentation_view_widget_ref = segmentation_view_widget_ref
        self.data_manager_interface = data_manager_interface
        self.data_manager_for_reads = data_manager_for_reads
        self.segmentation_model = segmentation_model_instance

        self.selected_samples_for_preview: List[str] = (
            []
        )  # IDs of samples currently in preview grid
        self.preview_widgets_map: Dict[str, QLabel] = (
            {}
        )  # Maps image_id to its QLabel in preview grid

        self.current_segmentation_thread: Optional[SegmentationThread] = None
        self.progress_info_bar: Optional[ProgressInfoBar] = None

        # Debounce timer for preview updates
        self.preview_debounce_timer = QTimer(self)
        self.preview_debounce_timer.setInterval(500)  # ms
        self.preview_debounce_timer.setSingleShot(True)
        self.preview_debounce_timer.timeout.connect(
            self.trigger_preview_segmentation_for_current_selection
        )

        self._connect_ui_signals()
        logging.info("SegmentationPresenter initialized.")

    def _connect_ui_signals(self):
        if not self.segmentation_view_widget_ref:
            return
        try:
            # Assuming controls are accessed via segmentation_view_widget_ref.controls (BaseSegmentationWidget)
            # or directly if the new AdvancedSegmentationWidget has its own signal sources.
            # For now, focusing on BaseSegmentationWidget's controls.
            base_pane = self.segmentation_view_widget_ref.base_segmentation_pane
            base_pane.controls.method_selector.currentIndexChanged.connect(
                self.request_preview_update_debounced
            )
            base_pane.controls.parameter_slider.valueChanged.connect(
                self.request_preview_update_debounced
            )
            base_pane.controls.parameter_slider_2.valueChanged.connect(
                self.request_preview_update_debounced
            )
            base_pane.controls.resample_button.clicked.connect(
                self.resample_preview_samples
            )
            base_pane.controls.apply_button.clicked.connect(
                self.run_full_segmentation_on_all
            )  # "Apply to all"
        except AttributeError as e:
            logging.error(
                f"Failed to connect UI signals in SegmentationPresenter: {e}. View structure might have changed."
            )

    @Slot(dict)  # Connected in BackendInitializer
    def on_samples_ready(self, samples_dict: Dict[str, Any]):
        """Called when initial samples are loaded by DataManager or new samples are added."""
        logging.debug(
            "SegmentationPresenter received samples_ready. Resampling preview if needed."
        )
        if (
            samples_dict and not self.selected_samples_for_preview
        ):  # If preview is empty and samples available
            self.resample_preview_samples()
        elif not samples_dict:  # No samples, clear preview
            self.selected_samples_for_preview.clear()
            self.preview_widgets_map.clear()
            if self.segmentation_view_widget_ref:
                self.segmentation_view_widget_ref.base_segmentation_pane.preview.update_previews(
                    []
                )

    @Slot(
        str
    )  # Receives mask_id. Can be from mask_created_in_db or mask_deleted_from_db
    def on_external_mask_change(self, mask_id: str):
        """
        Slot to handle notifications when a mask is created or deleted externally
        (i.e., not directly by this presenter's segmentation run).
        This might be used to refresh UI elements that depend on the list/status of masks.
        """
        logging.debug(
            f"SegmentationPresenter received on_external_mask_change for mask_id: {mask_id}"
        )

        # What should the presenter do here?
        # 1. If it maintains a list of images that have masks (e.g., for the advanced segmentation strip),
        #    it might need to update that list or the status of an image.
        # 2. If the currently displayed image in an advanced view had its mask changed, refresh that view.
        # 3. For the "Base Segmentation" view, this might not have an immediate effect unless
        #    it influences which samples are chosen for preview (e.g., prefer samples without masks).

        # Example: If the advanced segmentation strip is active and needs an update
        # if self.segmentation_view_widget_ref and \
        #    hasattr(self.segmentation_view_widget_ref, 'advanced_segmentation_pane') and \
        #    self.segmentation_view_widget_ref.stacked_widget.currentWidget() == self.segmentation_view_widget_ref.advanced_segmentation_pane:

        #     adv_pane = self.segmentation_view_widget_ref.advanced_segmentation_pane

        #     # Find the image_id associated with this mask_id
        #     image_id_affected = None
        #     mask_obj = self.data_manager_for_reads.get_mask(mask_id) # Check if it still exists (for created/updated)

        #     if mask_obj: # Mask was created or updated
        #         image_id_affected = mask_obj.image_id
        #         if adv_pane.image_strip_widget:
        #             adv_pane.image_strip_widget.update_card_mask_status(image_id_affected, True)
        #         # If this image is currently displayed in the advanced viewer, refresh its mask display
        #         if adv_pane.current_image_id == image_id_affected:
        #             # TODO: Logic to reload/redisplay mask overlays for current_image_id in advanced view
        #             logging.info(f"Mask for currently viewed image {image_id_affected} changed externally. Advanced view may need refresh.")
        #     else: # Mask was likely deleted
        #         # Find which sample *had* this mask_id
        #         for sample in self.data_manager_for_reads.samples.values():
        #             # This check is tricky because sample.mask_id might have already been cleared by DataManager
        #             # when the mask was deleted from DB.
        #             # Ideally, DataManager.mask_deleted_from_db would emit the image_id whose mask was deleted.
        #             # For now, we can't reliably get image_id from a deleted mask_id here.
        #             # Let's assume for now that if a mask is deleted, the adv strip might need a broader refresh
        #             # or the GalleryPresenter handles updating the 'has_mask' status on ImageCard data.
        #             pass
        #         logging.info(f"Mask {mask_id} was deleted externally. Advanced segmentation strip might need full refresh if it shows mask status.")
        # adv_pane.image_strip_widget.populate_images(...) # Potentially re-populate

        # For the Base Segmentation preview, this signal might not directly change anything
        # unless the resampling logic specifically tries to pick images without masks.
        # If a previewed image just got a mask, its next preview run would show it.

        logging.debug(
            f"SegmentationPresenter: Processed external mask change for {mask_id}."
        )

    @Slot()
    def request_preview_update_debounced(self):
        """Restarts the debounce timer for preview updates."""
        self.preview_debounce_timer.start()

    @Slot()
    def resample_preview_samples(self):
        """Selects new random samples for the preview grid and updates previews."""
        if not self.data_manager_for_reads or not self.data_manager_for_reads.samples:
            logging.warning(
                "Cannot resample previews: DataManager or samples not available."
            )
            if self.segmentation_view_widget_ref:
                self.segmentation_view_widget_ref.base_segmentation_pane.preview.update_previews(
                    []
                )
            return

        all_active_sample_ids = [
            sid for sid, s in self.data_manager_for_reads.samples.items() if s.is_active
        ]
        if not all_active_sample_ids:
            logging.info("No active samples to resample for preview.")
            self.selected_samples_for_preview.clear()
            self.preview_widgets_map.clear()
            if self.segmentation_view_widget_ref:
                self.segmentation_view_widget_ref.base_segmentation_pane.preview.update_previews(
                    []
                )
            return

        num_preview_slots = len(
            self.segmentation_view_widget_ref.base_segmentation_pane.preview.preview_widgets
        )
        k = min(num_preview_slots, len(all_active_sample_ids))

        if k == 0:
            logging.warning("No preview slots available or no active samples.")
            self.selected_samples_for_preview = []
            self.preview_widgets_map = {}
            return

        self.selected_samples_for_preview = random.sample(all_active_sample_ids, k)
        logging.info(
            f"Resampled preview with {k} samples: {self.selected_samples_for_preview}"
        )

        self.preview_widgets_map.clear()
        preview_labels = (
            self.segmentation_view_widget_ref.base_segmentation_pane.preview.preview_widgets
        )
        for i, sample_id in enumerate(self.selected_samples_for_preview):
            if i < len(preview_labels):
                self.preview_widgets_map[sample_id] = preview_labels[i]
                sample = self.data_manager_for_reads.samples.get(sample_id)
                if sample and sample.path and os.path.exists(sample.path):
                    try:
                        pixmap = QPixmap(sample.path)
                        if not pixmap.isNull():
                            preview_labels[i].setPixmap(
                                pixmap.scaled(
                                    preview_labels[i].size(),
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation,
                                )
                            )
                        else:
                            preview_labels[i].setText("Load Err")
                    except Exception:
                        preview_labels[i].setText("Load Err")
                else:
                    preview_labels[i].setText("No Path")

        for i in range(len(self.selected_samples_for_preview), len(preview_labels)):
            preview_labels[i].clear()
            preview_labels[i].setText("N/A")

        self.trigger_preview_segmentation_for_current_selection()

    @Slot()
    def trigger_preview_segmentation_for_current_selection(self):
        """Segments the currently selected preview samples."""
        if not self.selected_samples_for_preview:
            logging.debug("No samples selected for preview update.")
            return

        # Abort any ongoing preview thread
        if (
            self.current_segmentation_thread
            and self.current_segmentation_thread.isRunning()
            and self.current_segmentation_thread.is_preview_run
        ):
            logging.debug(
                "Preview segmentation already running, requesting interruption for new preview."
            )
            self.current_segmentation_thread.requestInterruption()
            # Don't wait, just start the new one. Old one will terminate.

        logging.debug(
            f"Triggering preview segmentation for samples: {self.selected_samples_for_preview}"
        )
        self._start_segmentation_thread(
            self.selected_samples_for_preview, is_preview=True
        )

    @Slot()
    def run_full_segmentation_on_all(self):
        """Starts segmentation for ALL active images in the dataset for persistence."""
        if not self.data_manager_for_reads or not self.data_manager_for_reads.samples:
            logging.warning(
                "Cannot run full segmentation: DataManager or samples not available."
            )
            return

        if (
            self.current_segmentation_thread
            and self.current_segmentation_thread.isRunning()
        ):
            logging.warning("A segmentation process is already running.")
            # TODO: Show InfoBar to user
            return

        all_active_sample_ids = [
            sid for sid, s in self.data_manager_for_reads.samples.items() if s.is_active
        ]
        if not all_active_sample_ids:
            logging.info("No active samples to run full segmentation on.")
            return

        logging.info(
            f"Starting full segmentation for all {len(all_active_sample_ids)} active images..."
        )
        self._start_segmentation_thread(all_active_sample_ids, is_preview=False)

    def _start_segmentation_thread(
        self, image_ids_to_process: List[str], is_preview: bool
    ):
        if (
            not self.segmentation_model
            or not self.data_manager_for_reads
            or not self.data_manager_for_reads.session
        ):
            logging.error(
                "Cannot start segmentation: Model, DataManager, or Session not available."
            )
            return
        if not image_ids_to_process:
            logging.info("No image IDs provided to _start_segmentation_thread.")
            return

        image_paths_to_process = []
        valid_ids_for_thread = []
        for img_id in image_ids_to_process:
            sample = self.data_manager_for_reads.samples.get(img_id)
            if sample and sample.path and os.path.exists(sample.path):
                image_paths_to_process.append(sample.path)
                valid_ids_for_thread.append(img_id)
            else:
                logging.warning(
                    f"Sample {img_id} has invalid path or does not exist. Skipping in this segmentation batch."
                )

        if not valid_ids_for_thread:
            logging.warning(
                "No valid images to process in _start_segmentation_thread after path validation."
            )
            return

        # Get UI parameters from the BaseSegmentationWidget's controls
        # This assumes the presenter has a reference to the view widget, and the view widget has `base_segmentation_pane`
        try:
            controls = self.segmentation_view_widget_ref.base_segmentation_pane.controls
            method = controls.method_selector.currentText()
            param1 = controls.parameter_slider.value()
            param2 = (
                controls.parameter_slider_2.value()
            )  # This might be hidden, ensure graceful handling in thread
        except AttributeError:
            logging.error(
                "Could not get segmentation parameters from UI. Using defaults."
            )
            method, param1, param2 = "Otsu's Thresholding", 30, 10

        if not is_preview:  # For full run, show progress bar
            self.progress_info_bar = ProgressInfoBar.new(
                icon=InfoBarIcon.INFORMATION,
                title="Segmentation Progress",
                content=f"Segmenting {len(valid_ids_for_thread)} images using {method}...",
                duration=-1,
                parent=(
                    self.segmentation_view_widget_ref.window()
                    if hasattr(self.segmentation_view_widget_ref, "window")
                    else self.segmentation_view_widget_ref
                ),
            )

        # Abort previous thread IF it's the same type (preview aborts preview, full aborts full)
        if (
            self.current_segmentation_thread
            and self.current_segmentation_thread.isRunning()
        ):
            if self.current_segmentation_thread.is_preview_run == is_preview:
                logging.info(
                    f"Requesting interruption of existing {'preview' if is_preview else 'full'} segmentation thread."
                )
                self.current_segmentation_thread.requestInterruption()
                # self.current_segmentation_thread.quit() # Not strictly needed if interruption is handled well
                # self.current_segmentation_thread.wait(500) # Give it a moment
            else:  # Different type of run, let the other one continue (e.g. full run while preview was active)
                # This might need more complex logic if only one thread instance is desired globally.
                pass

        if self.data_manager_for_reads.session:
            session_scale_factor = self.data_manager_for_reads.session.scale_factor
            if session_scale_factor is None:
                logging.warning(
                    "Session scale_factor is None. Properties will be in pixels."
                )

        self.current_segmentation_thread = SegmentationThread(
            image_ids=valid_ids_for_thread,
            image_paths=image_paths_to_process,
            masked_images_preview_dir=self.data_manager_for_reads.session.masked_images_directory,
            segmentation_model_instance=self.segmentation_model,
            segmentation_method_name=method,
            param1_value=param1,
            param2_value=param2,
            is_preview_run=is_preview,
            session_scale_factor=session_scale_factor,
        )

        self.current_segmentation_thread.progress_updated.connect(
            self._handle_thread_progress
        )
        if is_preview:
            self.current_segmentation_thread.preview_mask_ready.connect(
                self._handle_preview_mask_ready_from_thread
            )
        else:  # Full run
            self.current_segmentation_thread.raw_mask_generated.connect(
                self._handle_raw_mask_from_thread_for_full_run
            )

        self.current_segmentation_thread.finished.connect(
            lambda: self._on_segmentation_thread_finished(
                is_preview_run_that_finished=is_preview
            )
        )
        self.current_segmentation_thread.start()

    @Slot(int)
    def _handle_thread_progress(self, value: int):
        """Updates the progress bar if it's for a full run."""
        if (
            self.progress_info_bar
            and self.current_segmentation_thread
            and not self.current_segmentation_thread.is_preview_run
        ):
            self.progress_info_bar.set_progress(value)

    @Slot(str, object)  # image_id, combined_masked_image_numpy_array (RGB)
    def _handle_preview_mask_ready_from_thread(
        self, image_id: str, masked_image_np_array: Optional[np.ndarray]
    ):
        """Updates the specific preview widget with the generated mask image."""
        if image_id not in self.preview_widgets_map:
            logging.warning(
                f"Received preview mask for ID {image_id}, but it's no longer in the preview map."
            )
            return

        preview_label_widget = self.preview_widgets_map[image_id]
        if masked_image_np_array is None or not isinstance(
            masked_image_np_array, np.ndarray
        ):
            preview_label_widget.clear()
            preview_label_widget.setText("Seg Err")
            return
        try:
            h, w, ch = masked_image_np_array.shape
            if ch != 3:
                raise ValueError("Preview image must be 3-channel RGB.")
            q_image = QImage(
                masked_image_np_array.data, w, h, 3 * w, QImage.Format_RGB888
            )
            if q_image.isNull():
                raise ValueError("QImage conversion failed.")
            pixmap = QPixmap.fromImage(q_image)
            preview_label_widget.setPixmap(
                pixmap.scaled(
                    preview_label_widget.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        except Exception as e:
            logging.error(
                f"Error displaying preview for {image_id}: {e}", exc_info=True
            )
            preview_label_widget.clear()
            preview_label_widget.setText("Disp Err")

    @Slot(
        str, object, object, str
    )  # image_id, raw_mask_np, attributes_dict, preview_png_path
    def _handle_raw_mask_from_thread_for_full_run(
        self,
        image_id: str,
        raw_mask_np: np.ndarray,
        attributes: Dict[str, Any],
        masked_preview_path: str,
    ):
        """
        Receives raw mask data from SegmentationThread (full run) and forwards it to DataManager via interface.
        """
        logging.debug(
            f"SegmentationPresenter: Received raw mask (full run) for {image_id}. Forwarding to DataManagerInterface."
        )

        # Uses the new signal on DataManagerInterface (needs to be added)
        # DataManagerInterface.request_process_new_mask = Signal(str, object, object, str)
        # This signal connects to DataManager._on_process_new_mask_requested,
        # which calls DataManager.add_raw_mask_data_for_sample.
        if hasattr(self.data_manager_interface, "request_process_new_mask"):
            self.data_manager_interface.request_process_new_mask.emit(
                image_id, raw_mask_np, attributes, masked_preview_path
            )
        else:
            logging.error(
                "SegmentationPresenter: DataManagerInterface.request_process_new_mask signal is missing. Cannot process new mask for full run."
            )

    @Slot()
    def _on_segmentation_thread_finished(self, is_preview_run_that_finished: bool):
        """Handles completion of either a preview or a full segmentation thread."""
        logging.info(
            f"SegmentationPresenter: A segmentation thread (preview={is_preview_run_that_finished}) has finished."
        )

        # Clear the reference to the thread that just finished
        # This check ensures we only clear if it's the one we are tracking for that type
        if (
            self.current_segmentation_thread
            and self.current_segmentation_thread.is_preview_run
            == is_preview_run_that_finished
        ):
            self.current_segmentation_thread = None
        else:
            logging.warning(
                "Finished signal from an unexpected or already replaced segmentation thread."
            )

        if not is_preview_run_that_finished:  # Full run finished
            if self.progress_info_bar:
                self.progress_info_bar.set_progress(100)
                self.progress_info_bar.set_title("Segmentation Run Complete")
                self.progress_info_bar.set_content("All images processed.")
                QTimer.singleShot(2000, self.progress_info_bar.customClose)
                self.progress_info_bar = None

            logging.info(
                "SegmentationPresenter: Full segmentation run complete. Requesting DataManager to flush and persist mask data."
            )
            self.data_manager_interface.flush_db(
                callback=lambda: logging.info(
                    "SegmentationPresenter: DataManager confirmed flush/persistence after full segmentation."
                )
            )
            self.full_segmentation_run_finished.emit()
        else:  # Preview run finished
            logging.debug("SegmentationPresenter: Preview segmentation run finished.")
            # No special action needed beyond clearing thread ref, UI updates were per-image.

    def clear(self) -> None:
        logging.info("Clearing SegmentationPresenter state.")
        if (
            self.current_segmentation_thread
            and self.current_segmentation_thread.isRunning()
        ):
            logging.info(
                "Requesting interruption of running segmentation thread during clear."
            )
            self.current_segmentation_thread.requestInterruption()
            self.current_segmentation_thread.quit()
            self.current_segmentation_thread.wait(
                1000
            )  # Allow time to finish/terminate
        self.current_segmentation_thread = None

        self.preview_debounce_timer.stop()

        # Disconnect signals from UI elements
        if self.segmentation_view_widget_ref:
            try:
                base_pane = self.segmentation_view_widget_ref.base_segmentation_pane
                base_pane.controls.method_selector.currentIndexChanged.disconnect(
                    self.request_preview_update_debounced
                )
                base_pane.controls.parameter_slider.valueChanged.disconnect(
                    self.request_preview_update_debounced
                )
                base_pane.controls.parameter_slider_2.valueChanged.disconnect(
                    self.request_preview_update_debounced
                )
                base_pane.controls.resample_button.clicked.disconnect(
                    self.resample_preview_samples
                )
                base_pane.controls.apply_button.clicked.disconnect(
                    self.run_full_segmentation_on_all
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logging.warning(
                    f"Error disconnecting UI signals in SegmentationPresenter clear: {e}"
                )

        self.data_manager_interface = None
        self.data_manager_for_reads = None
        self.segmentation_model = None
        self.segmentation_view_widget_ref = None
        self.selected_samples_for_preview.clear()
        self.preview_widgets_map.clear()
        if self.progress_info_bar:
            self.progress_info_bar.close()
        logging.info("SegmentationPresenter cleared successfully.")
