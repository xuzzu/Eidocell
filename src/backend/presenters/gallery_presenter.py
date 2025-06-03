### backend\presenters\gallery_presenter.py
# backend/presenters/gallery_presenter.py

import logging
import os
from typing import Any, Dict, List, Optional, Union

from PySide6.QtCore import QModelIndex, QObject, Qt, Signal, Slot
from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition

from backend.data_manager import DataManager
from backend.data_manager_interface import DataManagerInterface
from backend.helpers.context_menu_handler import ContextMenuHandler
from backend.helpers.ctrl_helper import ControlHelper
from backend.helpers.sort_cards_thread import SortCardsThread
from backend.objects.mask import Mask
from backend.objects.sample import Sample
from UI.dialogs.custom_info_bar import CustomInfoBar
from UI.dialogs.gallery_filter_dialog import GalleryFilterDialog
from UI.dialogs.scale_calibration_dialog import ScaleCalibrationDialog
from UI.navigation_interface.workspace.views.gallery.image_card import ImageCard

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GalleryPresenter(QObject):
    gallery_sorted = Signal(list)  # For SortCardsThread result

    def __init__(
        self,
        gallery_view_widget_ref: Any,  # GalleryViewWidget
        data_manager_interface: DataManagerInterface,
        data_manager_instance: DataManager,
    ):  # Renamed for clarity
        super().__init__()
        self.gallery_view_widget = gallery_view_widget_ref  # Keep as direct ref if safe
        self.data_manager_interface = data_manager_interface
        self.data_manager = data_manager_instance  # Direct access for reads

        self.context_menu_handler = ContextMenuHandler(self)
        self.control_helper = ControlHelper(self.gallery_view_widget)

        self.last_clicked_card_id: Optional[str] = None
        self.mask_view_enabled: bool = False  # Delegate now handles this
        self.ctrl_pressed: bool = False
        self.thumbnail_quality: int = self.data_manager.settings.get(
            "thumbnail_quality", 75
        )
        self.sort_thread: Optional[SortCardsThread] = None
        self.scale_dialog: Optional[ScaleCalibrationDialog] = None
        self.active_filters: List[Dict[str, Any]] = []

        self._connect_ui_signals()
        logging.info("GalleryPresenter initialized.")

    def _connect_ui_signals(self):
        if not self.gallery_view_widget:
            return
        try:
            # Assuming GalleryViewWidget has 'controls' and 'gallery_container'
            self.gallery_view_widget.controls.mask_toggle.toggled.connect(
                self.toggle_mask_view
            )  # Use toggled
            self.gallery_view_widget.controls.scale_slider.valueChanged.connect(
                self.gallery_view_widget.resize_tiles
            )
            self.gallery_view_widget.controls.sortAscButton.toggled.connect(
                self.gallery_view_widget.updateSortingOrder
            )
            self.gallery_view_widget.controls.sortDescButton.toggled.connect(
                self.gallery_view_widget.updateSortingOrder
            )
            self.gallery_view_widget.controls.parameterComboBox.currentTextChanged.connect(
                self.gallery_view_widget.updateSortingParameter
            )  # Use currentTextChanged
            self.gallery_view_widget.controls.inspectButton.clicked.connect(
                self.handle_inspect_button
            )

            self.control_helper.ctrl_signal.connect(self.set_ctrl_pressed)

            # Connect double-click from the QListView (GalleryView)
            view = self.gallery_view_widget.gallery_container.gallery_view
            view.doubleClicked.connect(self.handle_image_double_click)

        except AttributeError as e:
            logging.error(
                f"Failed to connect UI signals in GalleryPresenter: {e}. View structure might have changed."
            )

    def load_gallery(self):
        """Populates gallery from DataManager's current active samples."""
        logging.info("GalleryPresenter: Loading gallery...")
        if not self.data_manager:
            logging.error(
                "GalleryPresenter: DataManager not available for loading gallery."
            )
            return
        # Gallery is now populated based on active_samples which is handled by its own slot.
        # Initial population happens when DM emits active_samples_updated after load_session_data.
        # This method can now just ensure the view is refreshed if called post-initialization.
        self._refresh_gallery_view_model()

    def _create_image_card_data_from_sample(
        self, sample_obj: Sample
    ) -> Optional[ImageCard]:
        """Helper to create an ImageCard data object from a Sample object."""
        if not sample_obj or not isinstance(sample_obj, Sample):
            return None

        class_color = "#FFFFFF"  # Default
        if sample_obj.class_id and sample_obj.class_id in self.data_manager.classes:
            class_color = self.data_manager.classes[sample_obj.class_id].color

        masked_preview_path = None
        if sample_obj.mask_id and sample_obj.mask_id in self.data_manager.masks:
            masked_preview_path = self.data_manager.masks[
                sample_obj.mask_id
            ].masked_image_path

        return ImageCard(
            id=sample_obj.id,
            name=os.path.basename(sample_obj.path),
            path=sample_obj.path,
            class_id=sample_obj.class_id,
            class_color=class_color,
            mask_path=masked_preview_path,
        )

    @Slot(dict)
    def on_active_samples_updated(
        self, active_samples_dict: Dict[str, Any]
    ):  # Sample objects
        logging.info(
            f"GalleryPresenter received active_samples_updated with {len(active_samples_dict)} samples."
        )
        self._refresh_gallery_view_model()

    @Slot(str)  # class_id
    def on_class_updated(self, class_id: str):
        logging.debug(f"GalleryPresenter received class_updated for ID: {class_id}")
        if not self.gallery_view_widget or not self.data_manager:
            return

        # Instead of directly manipulating the model, we trigger a full refresh
        # if the class change affects any of the currently displayed (active & filtered) samples.
        # This is simpler and leverages the existing refresh logic.

        # Check if any active sample belongs to the updated class
        affects_active_gallery = False
        for sample_obj in self.data_manager.active_samples.values():
            if sample_obj.class_id == class_id:
                affects_active_gallery = True
                break

        if affects_active_gallery:
            logging.debug(
                f"Class update for {class_id} affects active gallery. Refreshing model."
            )
            self._refresh_gallery_view_model()

    @Slot(object)
    def on_mask_created_or_updated(self, mask_obj: Mask):
        if not isinstance(mask_obj, Mask):
            logging.error(
                f"GalleryPresenter.on_mask_created_or_updated received invalid type: {type(mask_obj)}"
            )
            return

        image_id_affected = mask_obj.image_id
        logging.debug(
            f"GalleryPresenter: Mask created/updated for image {image_id_affected}"
        )

        # Check if the affected image is currently in the active_samples list
        if image_id_affected in self.data_manager.active_samples:
            logging.debug(
                f"Mask change for {image_id_affected} affects active gallery. Refreshing model."
            )
            # self._refresh_gallery_view_model()

    @Slot(
        str
    )  # Slot for DataManager.mask_deleted_from_db (passes mask_id, NOT image_id)
    def on_mask_deleted(self, deleted_mask_id: str):
        logging.debug(f"GalleryPresenter: Mask {deleted_mask_id} deleted.")

        # Find the image_id that was associated with this mask
        image_id_affected = None
        # Check DataManager's master list of samples (self.data_manager.samples)
        # as active_samples might already have removed it if it became inactive.
        for sample in self.data_manager.samples.values():
            # This is imperfect if sample.mask_id was already nulled by DM when mask was deleted.
            # A better signal from DM would include the image_id.
            # For now, let's assume if a mask is deleted, we might need a general refresh
            # if sorting/filtering is based on mask attributes.
            # Or, if the image whose mask was deleted is still in active_samples.
            # Let's just refresh if *any* mask is deleted for simplicity for now,
            # as filters or sort orders might depend on mask attributes.
            pass  # This logic needs a more robust way to find the affected image if sample.mask_id is already null

        # A simpler, though broader, approach: always refresh gallery if any mask is deleted,
        # as filters or sort order might depend on mask data.
        logging.debug(
            f"A mask ({deleted_mask_id}) was deleted. Refreshing gallery model as a precaution for filters/sort."
        )
        self._refresh_gallery_view_model()

    @Slot()
    def toggle_mask_view(self, checked: bool):
        self.mask_view_enabled = checked
        logging.info(
            f"GalleryPresenter: Mask view toggled. Enabled: {self.mask_view_enabled}"
        )

        if self.mask_view_enabled:
            self._refresh_gallery_view_model()
            if (
                not self.data_manager or not self.data_manager.active_samples
            ):  # Check active_samples
                all_have_mask_previews = False
            else:
                all_have_mask_previews = True
                for (
                    sample_obj
                ) in (
                    self.data_manager.active_samples.values()
                ):  # Iterate active_samples
                    if not (
                        sample_obj.mask_id
                        and sample_obj.mask_id in self.data_manager.masks
                        and self.data_manager.masks[
                            sample_obj.mask_id
                        ].masked_image_path
                        and os.path.exists(
                            self.data_manager.masks[
                                sample_obj.mask_id
                            ].masked_image_path
                        )
                    ):
                        all_have_mask_previews = False
                        break

            if not all_have_mask_previews:
                self.gallery_view_widget.controls.mask_toggle.setChecked(False)
                self.mask_view_enabled = False
                InfoBar.error(
                    title="Mask Previews Required",
                    content="Not all currently displayed images have generated mask previews. Please run segmentation.",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.BOTTOM_RIGHT,
                    duration=4000,
                    parent=self.gallery_view_widget.window(),
                )
                logging.warning(
                    "GalleryPresenter: Mask view toggle prevented: Not all displayed images have mask previews."
                )
                return

        delegate = self.gallery_view_widget.gallery_container.gallery_view.delegate
        delegate.view_mode = "mask" if self.mask_view_enabled else "image"

        model = self.gallery_view_widget.gallery_container.gallery_view.model
        model.invalidate_pixmap_cache_and_refresh()
        logging.info(
            f"GalleryPresenter: View mode set to '{delegate.view_mode}'. Cache invalidated."
        )

    @Slot()
    def show_filter_dialog(self):
        if (
            not self.data_manager or not self.data_manager.active_samples
        ):  # Check active_samples
            InfoBar.warning(
                title="No Data",
                content="No images currently displayed to filter.",
                parent=self.gallery_view_widget.window(),
                position=InfoBarPosition.BOTTOM_RIGHT,
            )
            return

        can_filter_quantitatively = False
        for (
            sample_obj
        ) in self.data_manager.active_samples.values():  # Check active_samples
            if sample_obj.mask_id and sample_obj.mask_id in self.data_manager.masks:
                if self.data_manager.masks[sample_obj.mask_id].attributes:
                    can_filter_quantitatively = True
                    break

        if not can_filter_quantitatively:
            InfoBar.warning(
                title="No Mask Attributes",
                content="No currently displayed images have mask attributes for filtering.",
                duration=5000,
                parent=self.gallery_view_widget.window(),
                position=InfoBarPosition.BOTTOM_RIGHT,
            )
            return

        available_features = ["Name", "ID", "Class ID"]
        first_mask_with_attrs = None
        for (
            sample_obj
        ) in self.data_manager.active_samples.values():  # Check active_samples
            if sample_obj.mask_id and sample_obj.mask_id in self.data_manager.masks:
                mask = self.data_manager.masks[sample_obj.mask_id]
                if mask.attributes:
                    first_mask_with_attrs = mask.attributes
                    break

        if first_mask_with_attrs:
            display_attr_keys = [
                key.replace("_", " ").title() for key in first_mask_with_attrs.keys()
            ]
            available_features.extend(display_attr_keys)

        available_features = sorted(list(set(available_features)))

        dialog = GalleryFilterDialog(
            available_features, self.gallery_view_widget.window()
        )
        dialog.filters_applied.connect(self.apply_gallery_filters)
        dialog.exec()

    @Slot(list)
    def apply_gallery_filters(self, filters: List[Dict[str, Any]]):
        logging.info(f"GalleryPresenter: Applying gallery filters: {filters}")
        self.active_filters = filters
        self._refresh_gallery_view_model()  # Re-apply filters and update model
        InfoBar.success(
            title="Filters Applied",
            content=f"{len(filters)} filter condition(s) are now active.",
            duration=3000,
            parent=self.gallery_view_widget.window(),
            position=InfoBarPosition.BOTTOM_RIGHT,
        )

    def _refresh_gallery_view_model(self):
        """
        Refreshes the GalleryModel based on current DataManager.active_samples
        and applies local gallery filters.
        """
        if not self.data_manager or not self.gallery_view_widget:
            return

        model = self.gallery_view_widget.gallery_container.gallery_view.model

        source_sample_objects = (
            self.data_manager.active_samples.values()
        )  # Use active samples

        all_potential_cards_data = []
        for sample_obj in source_sample_objects:
            card_data = self._create_image_card_data_from_sample(sample_obj)
            if card_data:
                all_potential_cards_data.append(card_data)

        final_cards_to_display = []
        if not self.active_filters:
            final_cards_to_display = all_potential_cards_data
        else:
            for card_data_item in all_potential_cards_data:
                sample_obj = self.data_manager.samples.get(
                    card_data_item.id
                )  # Get from master list for attributes
                if not sample_obj:
                    continue

                passes_all_filters = True
                for f_condition in self.active_filters:
                    feature_display_name = f_condition["feature"]
                    op = f_condition["operator"]
                    val1 = f_condition["value1"]
                    val2 = f_condition.get("value2")

                    attr_val = None
                    if feature_display_name == "Name":
                        attr_val = card_data_item.name
                    elif feature_display_name == "ID":
                        attr_val = card_data_item.id
                    elif feature_display_name == "Class ID":
                        attr_val = card_data_item.class_id
                    else:
                        internal_feature_key = feature_display_name.lower().replace(
                            " ", "_"
                        )
                        if (
                            sample_obj.mask_id
                            and sample_obj.mask_id in self.data_manager.masks
                        ):
                            mask_attributes = self.data_manager.masks[
                                sample_obj.mask_id
                            ].attributes
                            if internal_feature_key in mask_attributes:
                                attr_val = mask_attributes[internal_feature_key]
                            else:
                                passes_all_filters = False
                                break
                        else:
                            passes_all_filters = False
                            break

                    if attr_val is None:
                        passes_all_filters = False
                        break

                    try:
                        if isinstance(attr_val, str):
                            s_attr_val = str(attr_val).lower()
                            s_val1 = str(val1).lower()
                            if op == "==" and not (s_attr_val == s_val1):
                                passes_all_filters = False
                                break
                            elif op == "!=" and not (s_attr_val != s_val1):
                                passes_all_filters = False
                                break
                            elif op == "Contains" and not (s_val1 in s_attr_val):
                                passes_all_filters = False
                                break
                            elif op == "Not Contains" and not (
                                s_val1 not in s_attr_val
                            ):
                                passes_all_filters = False
                                break
                            elif op in [">", ">=", "<", "<=", "Between"]:
                                passes_all_filters = False
                                break
                        else:
                            f_attr_val = float(attr_val)
                            f_val1 = float(val1)
                            if op == ">" and not (f_attr_val > f_val1):
                                passes_all_filters = False
                                break
                            elif op == ">=" and not (f_attr_val >= f_val1):
                                passes_all_filters = False
                                break
                            elif op == "<" and not (f_attr_val < f_val1):
                                passes_all_filters = False
                                break
                            elif op == "<=" and not (f_attr_val <= f_val1):
                                passes_all_filters = False
                                break
                            elif op == "==" and not (abs(f_attr_val - f_val1) < 1e-9):
                                passes_all_filters = False
                                break
                            elif op == "!=" and not (abs(f_attr_val - f_val1) >= 1e-9):
                                passes_all_filters = False
                                break
                            elif op == "Between":
                                f_val2 = float(val2)
                                if not (
                                    min(f_val1, f_val2)
                                    <= f_attr_val
                                    <= max(f_val1, f_val2)
                                ):
                                    passes_all_filters = False
                                    break
                    except ValueError:
                        passes_all_filters = False
                        break

                if passes_all_filters:
                    final_cards_to_display.append(card_data_item)

        model.clear()
        model.addImages(final_cards_to_display)  # Use batch add

        logging.info(
            f"Gallery model refreshed with {len(final_cards_to_display)} items."
        )

    def clear(self) -> None:
        logging.info("Clearing GalleryPresenter state.")

        if self.sort_thread:
            if self.sort_thread.isRunning():
                logging.info("Stopping sort thread during GalleryPresenter clear...")
                self.sort_thread.requestInterruption()
                self.sort_thread.quit()
                if not self.sort_thread.wait(1000):
                    self.sort_thread.terminate()
                    self.sort_thread.wait()
            try:
                self.sort_thread.sorted_data.disconnect(self.on_cards_sorted)
            except (TypeError, RuntimeError):
                pass
            try:
                self.sort_thread.finished.disconnect(self._handle_sort_thread_finished)
            except (TypeError, RuntimeError):
                pass
            try:
                self.sort_thread.finished.disconnect(self.sort_thread.deleteLater)
            except (TypeError, RuntimeError):
                pass
            self.sort_thread.deleteLater()
        self.sort_thread = None

        if self.control_helper:
            try:
                self.control_helper.ctrl_signal.disconnect(self.set_ctrl_pressed)
            except (TypeError, RuntimeError):
                pass
            self.control_helper.deleteLater()
            self.control_helper = None

        if self.gallery_view_widget:
            try:
                self.gallery_view_widget.controls.mask_toggle.toggled.disconnect(
                    self.toggle_mask_view
                )
                self.gallery_view_widget.controls.scale_slider.valueChanged.disconnect(
                    self.gallery_view_widget.resize_tiles
                )
                self.gallery_view_widget.controls.sortAscButton.toggled.disconnect(
                    self.gallery_view_widget.updateSortingOrder
                )
                self.gallery_view_widget.controls.sortDescButton.toggled.disconnect(
                    self.gallery_view_widget.updateSortingOrder
                )
                self.gallery_view_widget.controls.parameterComboBox.currentTextChanged.disconnect(
                    self.gallery_view_widget.updateSortingParameter
                )
                self.gallery_view_widget.controls.inspectButton.clicked.disconnect(
                    self.handle_inspect_button
                )
                view = self.gallery_view_widget.gallery_container.gallery_view
                view.doubleClicked.disconnect(self.handle_image_double_click)
            except (AttributeError, TypeError, RuntimeError) as e:
                logging.warning(f"Error disconnecting GalleryPresenter UI signals: {e}")

            self.gallery_view_widget.gallery_container.gallery_view.model.clear()

        if self.scale_dialog:
            self.scale_dialog.close()
            self.scale_dialog = None

        self.data_manager_interface = None
        self.data_manager = None
        self.gallery_view_widget = None
        self.context_menu_handler = None
        self.active_filters.clear()

        logging.info("GalleryPresenter cleared successfully.")

    def get_selected_images(self) -> List[ImageCard]:
        if not self.gallery_view_widget:
            return []
        view = self.gallery_view_widget.gallery_container.gallery_view
        selected_indexes = view.selectionModel().selectedIndexes()
        return [
            index.data(Qt.UserRole)
            for index in selected_indexes
            if index.isValid() and isinstance(index.data(Qt.UserRole), ImageCard)
        ]

    def perform_class_assignment(self, class_name: str):
        selected_image_cards = self.get_selected_images()
        if not selected_image_cards:
            return
        image_ids_to_assign = [card.id for card in selected_image_cards]
        if not self.data_manager or not self.data_manager_interface:
            return
        class_object = self.data_manager.get_class_by_name(class_name)
        if not class_object:
            InfoBar.error(
                title="Error",
                content=f"Class '{class_name}' not found.",
                parent=self.gallery_view_widget.window(),
                position=InfoBarPosition.BOTTOM_RIGHT,
            )
            return
        self.data_manager_interface.assign_images_to_class(
            image_ids_to_assign, class_object.id
        )

    def sort_gallery(self):
        if not self.gallery_view_widget or not self.data_manager:
            return
        param = self.gallery_view_widget.sorting_parameter
        order = self.gallery_view_widget.sorting_order

        requires_masks_for_sort = param not in ["Name", "ID", "Class ID"]
        if requires_masks_for_sort:
            can_sort = False
            # Check against active_samples only
            for sample_id in self.data_manager.active_samples:
                sample_obj = self.data_manager.samples.get(
                    sample_id
                )  # Get full sample from master list
                if (
                    sample_obj
                    and sample_obj.mask_id
                    and sample_obj.mask_id in self.data_manager.masks
                    and self.data_manager.masks[sample_obj.mask_id].attributes
                ):
                    can_sort = True
                    break
            if not can_sort:
                InfoBar.warning(
                    title="Sorting Not Possible",
                    content=f"Cannot sort by '{param}' as no currently displayed images have the required mask attributes.",
                    parent=self.gallery_view_widget.window(),
                    position=InfoBarPosition.BOTTOM_RIGHT,
                )
                self.gallery_view_widget.on_sorting_failed()
                return

        if self.sort_thread and self.sort_thread.isRunning():
            self.sort_thread.requestInterruption()
            self.sort_thread.quit()
            self.sort_thread.wait(500)

        self.sort_thread = SortCardsThread(
            self.data_manager, param, order
        )  # Pass full DM, thread will use active_samples
        self.sort_thread.sorted_data.connect(self.on_cards_sorted)
        self.sort_thread.finished.connect(self._handle_sort_thread_finished)
        self.sort_thread.finished.connect(self.sort_thread.deleteLater)
        self.sort_thread.start()

    @Slot(list)
    def on_cards_sorted(self, sorted_image_ids: list):
        if not self.gallery_view_widget:
            return
        if not sorted_image_ids and self.gallery_view_widget.sorting_parameter not in [
            "Name",
            "ID",
            "Class ID",
        ]:
            logging.warning(
                "Sort thread returned empty list, possibly due to error or missing masks for sorting parameter."
            )
            return

        # The GalleryModel expects ImageCard objects, not just IDs, for reordering if it's
        # maintaining its own list. However, if reorderImagesByIds just tells it the new order of IDs,
        # and the model already has all cards, this is fine.
        # Let's assume reorderImagesByIds correctly handles this.
        self.gallery_view_widget.gallery_container.gallery_view.model.reorderImagesByIds(
            sorted_image_ids
        )

    @Slot()
    def _handle_sort_thread_finished(self):
        sender_thread = self.sender()
        if sender_thread == self.sort_thread:
            self.sort_thread = None

    @Slot(QModelIndex)
    def handle_image_double_click(self, index: QModelIndex):
        if index.isValid():
            image_card_data = index.data(Qt.UserRole)
            if isinstance(image_card_data, ImageCard):
                self.show_scale_calibration_dialog(image_card_data.path)

    @Slot()
    def handle_inspect_button(self):
        selected_image_cards = self.get_selected_images()

        if not selected_image_cards:
            InfoBar.warning(
                title="No Image Selected",
                content="Please select an image from the gallery to set its scale.",
                parent=(
                    self.gallery_view_widget.window()
                    if self.gallery_view_widget
                    else None
                ),
                position=InfoBarPosition.BOTTOM_RIGHT,
            )
            return

        if len(selected_image_cards) > 1:
            InfoBar.warning(
                title="Multiple Images Selected",
                content="Please select only one image at a time to set its scale.",
                parent=(
                    self.gallery_view_widget.window()
                    if self.gallery_view_widget
                    else None
                ),
                position=InfoBarPosition.BOTTOM_RIGHT,
            )
            return

        image_to_calibrate = selected_image_cards[0]
        self.show_scale_calibration_dialog(image_to_calibrate.path)

    def show_scale_calibration_dialog(self, image_path: str):
        if not image_path or not os.path.exists(image_path):
            InfoBar.error(
                title="Error",
                content="Image path is invalid or file does not exist.",
                parent=self.gallery_view_widget.window(),
            )
            return
        if self.scale_dialog and self.scale_dialog.isVisible():
            self.scale_dialog.raise_()
            self.scale_dialog.activateWindow()
            return
        self.scale_dialog = ScaleCalibrationDialog(image_path)
        self.scale_dialog.scale_applied.connect(self.handle_scale_applied)
        self.scale_dialog.finished.connect(self._on_scale_dialog_finished)
        self.scale_dialog.show()

    @Slot(float, str)
    def handle_scale_applied(self, scale_factor: float, units: str):
        InfoBar.success(
            title="Scale Set",
            content=f"Image scale set to {scale_factor:.4f} {units}/pixel.",
            parent=self.gallery_view_widget.window(),
        )
        # New: Send scale to DataManager
        if self.data_manager_interface:
            self.data_manager_interface.set_session_scale(scale_factor, units)
        else:
            logging.error(
                "GalleryPresenter: DataManagerInterface not available to set session scale."
            )

    @Slot()
    def _on_scale_dialog_finished(self):
        if self.scale_dialog:
            try:
                self.scale_dialog.scale_applied.disconnect(self.handle_scale_applied)
            except (TypeError, RuntimeError):
                pass
            try:
                self.scale_dialog.finished.disconnect(self._on_scale_dialog_finished)
            except (TypeError, RuntimeError):
                pass
            self.scale_dialog = None
        logging.debug("Scale dialog reference cleared.")

    @Slot(bool)
    def set_ctrl_pressed(self, pressed: bool):
        self.ctrl_pressed = pressed
