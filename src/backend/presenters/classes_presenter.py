# backend/presenters/classes_presenter.py
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem
from qfluentwidgets import (
    Flyout,
    FlyoutAnimationType,
    FlyoutView,
    InfoBar,
    InfoBarIcon,
    InfoBarPosition,
    PrimaryPushButton,
)
from qfluentwidgets.components.material import AcrylicLineEdit

from backend.config import CLASS_FRAME_PATH, COLLAGE_RES_SCALE, get_available_models
from backend.data_manager import DataManager
from backend.data_manager_interface import DataManagerInterface
from backend.objects.sample_class import SampleClass
from backend.utils.image_utils import composite_images, merge_images_collage, tint_image
from UI.dialogs.class_cluster_summary import ClassClusterSummary
from UI.dialogs.class_cluster_view import ClassClusterViewer
from UI.dialogs.progress_infobar import ProgressInfoBar
from UI.dialogs.train_classifier_dialog import TrainClassifierDialog

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClassesPresenter(QObject):
    def __init__(
        self,
        classes_view_widget_ref: Any,
        data_manager_interface: DataManagerInterface,
        data_manager_instance: DataManager,
        images_per_preview: Optional[int] = None,
    ):
        super().__init__()
        self.classes_view_widget = classes_view_widget_ref
        self.data_manager_interface = data_manager_interface
        self.data_manager = data_manager_instance

        self.images_per_preview = (
            images_per_preview
            if images_per_preview is not None
            else self.data_manager.settings.get("images_per_collage", 25)
        )

        self.tree_widget = None
        if self.classes_view_widget and hasattr(
            self.classes_view_widget, "class_tree_view"
        ):
            self.tree_widget = self.classes_view_widget.class_tree_view
        else:
            logging.error(
                "ClassesPresenter: ClassesViewWidget does not have 'class_tree_view'. Tree functionality will be limited."
            )

        self._connect_ui_signals()
        logging.info("ClassesPresenter initialized.")

    def _connect_ui_signals(self):
        if not self.classes_view_widget:
            return
        try:
            self.classes_view_widget.create_class_requested.connect(
                self.request_create_class
            )
        except AttributeError as e:
            logging.error(f"Failed to connect UI signals in ClassesPresenter: {e}.")

    #  Slots for Signals FROM DataManager (Connected in BackendInitializer)
    @Slot(str)  # class_id
    def on_class_added(self, class_id: str):
        logging.debug(f"ClassesPresenter received class_added for ID: {class_id}")
        if not self.classes_view_widget or not self.data_manager:
            return

        class_object = self.data_manager.get_class(class_id)
        if class_object:
            preview_image_path = self._generate_class_preview(class_id)
            self.classes_view_widget.create_class_card(
                class_object.name,
                class_object.id,
                class_object.color,
                preview_image_path,
            )
            if self.tree_widget:
                self.add_class_to_tree(class_object)
        else:
            logging.error(
                f"Class {class_id} not found in DataManager after class_added signal."
            )

    @Slot(str)  # class_id
    def on_class_updated(self, class_id: str):
        logging.debug(f"ClassesPresenter received class_updated for ID: {class_id}")
        if not self.classes_view_widget or not self.data_manager:
            return

        class_object = self.data_manager.get_class(class_id)
        if not class_object:
            logging.warning(
                f"Class {class_id} not found in DataManager during update signal. Removing from UI."
            )
            self.classes_view_widget.delete_class_card(class_id)
            if self.tree_widget:
                root_item = self.tree_widget.invisibleRootItem()
                item_to_remove = self._find_tree_item_by_id_recursive(
                    root_item, class_id
                )
                if item_to_remove:
                    (item_to_remove.parent() or root_item).removeChild(item_to_remove)
            return

        preview_image_path = self._generate_class_preview(class_id)
        self.classes_view_widget.update_class_card(
            class_id, preview_image_path, class_object.name, class_object.color
        )

        if self.tree_widget:
            root_item = self.tree_widget.invisibleRootItem()
            item_to_update = self._find_tree_item_by_id_recursive(root_item, class_id)
            if item_to_update:
                item_to_update.setText(0, class_object.name)
            else:  # If class was just added and updated (e.g. images assigned during creation)
                self.add_class_to_tree(class_object)

    @Slot(str)  # class_id
    def on_class_deleted(self, class_id: str):
        logging.debug(f"ClassesPresenter received class_deleted for ID: {class_id}")
        if not self.classes_view_widget:
            return

        self.classes_view_widget.delete_class_card(class_id)
        if self.tree_widget:
            root_item = self.tree_widget.invisibleRootItem()
            item_to_remove = self._find_tree_item_by_id_recursive(root_item, class_id)
            if item_to_remove:
                (item_to_remove.parent() or root_item).removeChild(item_to_remove)
            else:
                logging.warning(
                    f"Could not find tree item for deleted class ID {class_id}."
                )

    #  Methods Triggered by UI Actions (Requesting changes via Interface)
    @Slot(str, str)  # name, color_hex (from ClassesViewWidget.create_class_requested)
    def request_create_class(self, class_name: str, color_hex: str):
        logging.info(
            f"ClassesPresenter: Requesting creation of class '{class_name}' with color {color_hex}"
        )
        if not class_name:
            QMessageBox.warning(
                self.classes_view_widget, "Error", "Class name cannot be empty."
            )
            return

        if self.data_manager and self.data_manager.get_class_by_name(class_name):
            QMessageBox.warning(
                self.classes_view_widget,
                "Error",
                "A class with this name already exists.",
            )
            return

        if not color_hex:  # Should be provided by view, but fallback
            color_hex = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            logging.warning(
                f"No color provided for new class '{class_name}', using random: {color_hex}"
            )

        if self.data_manager_interface:
            self.data_manager_interface.create_class(class_name, color_hex)
        else:
            logging.error("Cannot create class: DataManagerInterface not available.")

    @Slot(str)  # class_id (from context menu action, usually)
    def request_delete_class(self, class_id: str):
        if not self.data_manager or not self.data_manager_interface:
            logging.error(
                "Cannot delete class: DataManager or Interface not available."
            )
            return

        class_to_delete = self.data_manager.get_class(class_id)
        if not class_to_delete:
            logging.error(f"Cannot delete class: ID {class_id} not found.")
            return

        if class_to_delete.name.lower() == "uncategorized":
            QMessageBox.warning(
                self.classes_view_widget,
                "Action Denied",
                "Cannot delete the default 'Uncategorized' class.",
            )
            return

        reply = QMessageBox.question(
            self.classes_view_widget,
            "Confirm Deletion",
            f"Are you sure you want to delete class '{class_to_delete.name}'?\n"
            f"Images within this class will be moved to 'Uncategorized'.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            logging.info(
                f"Requesting deletion of class: {class_id} ({class_to_delete.name})"
            )
            uncategorized_class = self.data_manager.get_class_by_name("Uncategorized")
            if not uncategorized_class:
                logging.error(
                    "Cannot proceed with class deletion: 'Uncategorized' class not found."
                )
                InfoBar.error(
                    title="Error",
                    content="Default 'Uncategorized' class is missing.",
                    parent=self.classes_view_widget.window(),
                )
                return

            # DataManager will handle moving images to Uncategorized before deleting the class row.
            # The assign_images_to_class call is now implicit in DM.delete_class.
            self.data_manager_interface.delete_class(class_id)
            # UI updates (card removal, tree removal) handled by on_class_deleted slot

    @Slot(str)  # class_id (from context menu or similar UI trigger)
    def handle_rename_class_ui_request(self, class_id: str):
        """Shows the flyout for renaming a class."""
        if not self.data_manager or not self.classes_view_widget:
            return
        class_to_rename = self.data_manager.get_class(class_id)
        if not class_to_rename:
            logging.error(f"Cannot initiate rename: Class {class_id} not found.")
            return

        view = FlyoutView(
            title=f'Rename Class "{class_to_rename.name}"', content=""
        )  # Simpler content
        name_input = AcrylicLineEdit(parent=view)
        name_input.setPlaceholderText("Enter new class name")
        name_input.setText(class_to_rename.name)
        view.addWidget(name_input)

        rename_button = PrimaryPushButton("Rename", view)
        view.addWidget(rename_button)
        rename_button.clicked.connect(
            lambda: self._confirm_and_request_rename(class_id, name_input.text(), view)
        )

        target_card = next(
            (
                card
                for card in self.classes_view_widget.classes
                if card.class_id == class_id
            ),
            None,
        )
        if target_card:
            Flyout.make(
                view=view,
                target=target_card,
                parent=self.classes_view_widget,
                aniType=FlyoutAnimationType.DROP_DOWN,
            )
        else:
            logging.warning(
                f"Could not find target card for rename flyout (Class ID: {class_id})"
            )

    def _confirm_and_request_rename(
        self, class_id: str, new_class_name: str, flyout_view: FlyoutView
    ):
        """Validates input and requests class rename via the interface."""
        new_name_stripped = new_class_name.strip()
        if not new_name_stripped:
            QMessageBox.warning(
                self.classes_view_widget, "Error", "Class name cannot be empty."
            )
            return
        if not self.data_manager or not self.data_manager_interface:
            logging.error(
                "Cannot rename class: DataManager or Interface not available."
            )
            flyout_view.close()
            return

        current_class = self.data_manager.get_class(class_id)
        if not current_class:
            logging.error(f"Cannot rename class: Original class {class_id} not found.")
            flyout_view.close()
            return

        if current_class.name == new_name_stripped:
            logging.info("Class name unchanged, skipping rename request.")
            flyout_view.close()
            return

        existing_class_with_new_name = self.data_manager.get_class_by_name(
            new_name_stripped
        )
        if existing_class_with_new_name and existing_class_with_new_name.id != class_id:
            QMessageBox.warning(
                self.classes_view_widget,
                "Error",
                "A class with this name already exists.",
            )
            return

        logging.info(
            f"Requesting rename of class {class_id} from '{current_class.name}' to '{new_name_stripped}'"
        )
        self.data_manager_interface.rename_class(class_id, new_name_stripped)
        flyout_view.close()  # UI update handled by on_class_updated

    @Slot(dict)  # active_samples_dict from DataManager
    def on_active_samples_changed(self, active_samples_dict: Dict[str, Any]):
        """
        Called when DataManager's active_samples list has changed.
        Regenerates previews for all class cards.
        """
        logging.debug(
            f"ClassesPresenter: Received active_samples_changed with {len(active_samples_dict)} active samples. Regenerating all class previews."
        )
        if not self.data_manager or not self.classes_view_widget:
            return

        # Regenerate preview for all existing classes
        for class_id in list(
            self.data_manager.classes.keys()
        ):  # Iterate over a copy of keys
            class_object = self.data_manager.get_class(class_id)
            if class_object:
                self.on_class_updated(class_id)
            else:
                # This case should be rare if DM is consistent, but handle defensively.
                logging.warning(
                    f"ClassesPresenter: Class {class_id} not found while regenerating previews for active sample change."
                )

    #  UI Population and Helper Methods
    def load_classes(self):
        """Loads existing classes into the UI using direct reads from DataManager."""
        logging.info("ClassesPresenter: Loading initial classes...")
        if not self.classes_view_widget or not self.data_manager:
            return

        self.classes_view_widget.clear_classes()  # Clear existing UI elements first

        for (
            class_object
        ) in (
            self.data_manager.classes.values()
        ):  # Iterate sorted by name for consistent order
            preview_image_path = self._generate_class_preview(class_object.id)
            self.classes_view_widget.create_class_card(
                class_object.name,
                class_object.id,
                class_object.color,
                preview_image_path,
            )
            if self.tree_widget:
                self.add_class_to_tree(class_object)
        logging.info(
            f"ClassesPresenter: Loaded {len(self.data_manager.classes)} classes into UI."
        )

    def add_class_to_tree(
        self, class_object: SampleClass, parent_node_id: Optional[str] = None
    ):
        if not self.tree_widget:
            return

        # Avoid adding if already exists (e.g., due to race condition or complex update)
        if self._find_tree_item_by_id_recursive(
            self.tree_widget.invisibleRootItem(), class_object.id
        ):
            return

        item = QTreeWidgetItem([class_object.name])
        item.setData(0, Qt.UserRole, class_object.id)

        parent_tree_item = None
        if parent_node_id:  # If implementing hierarchy later
            root = self.tree_widget.invisibleRootItem()
            parent_tree_item = self._find_tree_item_by_id_recursive(
                root, parent_node_id
            )

        if parent_tree_item:
            parent_tree_item.addChild(item)
        else:
            self.tree_widget.addTopLevelItem(item)

    def _find_tree_item_by_id_recursive(
        self, parent_item: QTreeWidgetItem, class_id: str
    ) -> Optional[QTreeWidgetItem]:
        if not self.tree_widget:
            return None  # Guard
        for i in range(parent_item.childCount()):
            child_item = parent_item.child(i)
            if child_item.data(0, Qt.UserRole) == class_id:
                return child_item
            found_in_grandchild = self._find_tree_item_by_id_recursive(
                child_item, class_id
            )
            if found_in_grandchild:
                return found_in_grandchild
        return None

    @Slot(str)  # class_id
    def show_class_viewer(self, class_id: str):
        if not self.data_manager or not self.classes_view_widget:
            return
        class_object = self.data_manager.get_class(class_id)
        if not class_object:
            logging.error(f"Cannot show viewer: Class {class_id} not found.")
            return

        viewer = ClassClusterViewer(
            f"Class: {class_object.name}", self, self.classes_view_widget.window()
        )

        active_samples_in_class_count = 0
        for sample in class_object.samples:
            if (
                sample.id in self.data_manager.active_samples
            ):  # Check against DM's global active samples
                viewer.add_card(sample.id)
                active_samples_in_class_count += 1
        logging.info(
            f"Showing viewer for class: {class_object.name} with {active_samples_in_class_count} active samples."
        )
        viewer.show()

    def _generate_class_preview(self, class_id: str) -> Optional[str]:
        if not self.data_manager:
            return None
        class_object = self.data_manager.get_class(class_id)
        if not class_object:
            return None

        try:
            frame_base_image = Image.open(CLASS_FRAME_PATH).convert("RGBA")
        except Exception as e:
            logging.error(f"Error loading class frame image {CLASS_FRAME_PATH}: {e}")
            return None

        tinted_frame = tint_image(frame_base_image, class_object.color)

        active_sample_paths_in_class = []
        for sample_in_class in class_object.samples:
            # Check if the sample is in the DataManager's global active_samples list
            if (
                sample_in_class.id in self.data_manager.active_samples
                and sample_in_class.path
                and os.path.exists(sample_in_class.path)
            ):
                active_sample_paths_in_class.append(sample_in_class.path)

        # Log the count of active samples being used for this class preview
        logging.debug(
            f"Class '{class_object.name}': Found {len(active_sample_paths_in_class)} active samples for preview collage (out of {len(class_object.samples)} total in class)."
        )

        image_paths_for_collage = active_sample_paths_in_class[
            : self.images_per_preview
        ]

        collage_image_pil = None
        if image_paths_for_collage:
            collage_image_pil = merge_images_collage(
                image_paths_for_collage, scale=COLLAGE_RES_SCALE
            )

        collage_position_in_frame = (20, 36)
        collage_target_size_in_frame = (135, 135)

        final_card_image_pil = None
        if collage_image_pil:
            final_card_image_pil = composite_images(
                tinted_frame,
                collage_image_pil,
                collage_position_in_frame,
                collage_target_size_in_frame,
            )
        else:
            final_card_image_pil = tinted_frame.resize(
                collage_target_size_in_frame, Image.LANCZOS
            )
            # logging.debug(f"No active images for collage in class {class_id}, using tinted frame only for preview.")

        if final_card_image_pil:
            # Use session-specific temp folder for previews to avoid clashes if multiple sessions run/debugged
            # and to make cleanup easier if previews are tied to a session.
            temp_preview_dir = os.path.join(
                self.data_manager.session.metadata_directory, "class_previews_temp"
            )
            os.makedirs(temp_preview_dir, exist_ok=True)
            preview_path = os.path.join(
                temp_preview_dir, f"class_card_preview_{class_id}.png"
            )
            try:
                final_card_image_pil.save(preview_path)
                return preview_path
            except Exception as e:
                logging.error(
                    f"Failed to save final class card image {preview_path}: {e}"
                )
        return None

    @Slot(str)  # class_id
    def show_summary(self, class_id: str):
        if not self.data_manager or not self.classes_view_widget:
            return
        class_object = self.data_manager.get_class(class_id)
        if not class_object:
            logging.error(f"Cannot show summary: Class {class_id} not found.")
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

        for sample in class_object.samples:
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
                        f"Could not calc stats for {param_key} in class {class_id}: {e}"
                    )

        summary_window = ClassClusterSummary(
            f"Class Summary: {class_object.name}",
            parent=self.classes_view_widget.window(),
        )
        summary_window.set_summary_data(
            class_object.name, active_samples_for_summary_count, parameter_data
        )
        summary_window.show()

    def clear(self) -> None:
        logging.info("Clearing ClassesPresenter state.")
        if self.classes_view_widget:
            try:
                self.classes_view_widget.create_class_requested.disconnect(
                    self.request_create_class
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logging.warning(f"Error disconnecting ClassesPresenter UI signals: {e}")
            self.classes_view_widget.clear_classes()

        self.data_manager_interface = None
        self.data_manager = None
        self.classes_view_widget = None
        self.tree_widget = None
        logging.info("ClassesPresenter cleared successfully.")

    @Slot(str, int, float)
    def handle_classifier_training_progress(
        self, classifier_name: str, epoch_or_step: int, metric_value: float
    ):
        if self.training_progress_bar:
            total_epochs = (
                100  # Placeholder, ideally get this from training initiation params
            )
            if (
                hasattr(self, "_current_training_epochs")
                and self._current_training_epochs > 0
            ):
                total_epochs = self._current_training_epochs

            progress_percent = (
                int((epoch_or_step / total_epochs) * 100) if total_epochs > 0 else 0
            )
            self.training_progress_bar.set_progress(progress_percent)
            self.training_progress_bar.set_content(
                f"Training '{classifier_name}': Epoch {epoch_or_step}/{total_epochs}, Metric: {metric_value:.4f}"
            )

    @Slot(str, bool, str)
    def handle_classifier_training_finished(
        self, classifier_name: str, success: bool, message: str
    ):
        if self.training_progress_bar:
            self.training_progress_bar.set_progress(100)  # Ensure it shows 100%
            if success:
                self.training_progress_bar.set_title("Training Complete")
                self.training_progress_bar.set_content(
                    f"Classifier '{classifier_name}' trained. {message}"
                )
                self.training_progress_bar.setIcon(InfoBarIcon.SUCCESS)
                if classifier_name not in self.trained_classifier_names:
                    self.trained_classifier_names.append(classifier_name)
                if self.classes_view_widget:
                    self.classes_view_widget.update_trained_classifiers_list(
                        self.trained_classifier_names
                    )
            else:
                self.training_progress_bar.set_title("Training Failed")
                self.training_progress_bar.set_content(
                    f"Training for '{classifier_name}' failed: {message}"
                )
                self.training_progress_bar.setIcon(InfoBarIcon.ERROR)
            QTimer.singleShot(4000, self.training_progress_bar.customClose)
            self.training_progress_bar = None

    @Slot(str, bool, str)
    def handle_classification_run_finished(
        self, classifier_name: str, success: bool, message: str
    ):
        if self.classification_progress_bar:
            self.classification_progress_bar.set_progress(100)
            if success:
                self.classification_progress_bar.set_title("Classification Complete")
                self.classification_progress_bar.set_content(
                    f"Finished classifying with '{classifier_name}'. {message}"
                )
                self.classification_progress_bar.setIcon(InfoBarIcon.SUCCESS)
            else:
                self.classification_progress_bar.set_title("Classification Failed")
                self.classification_progress_bar.set_content(
                    f"Classification with '{classifier_name}' failed: {message}"
                )
                self.classification_progress_bar.setIcon(InfoBarIcon.ERROR)
            QTimer.singleShot(4000, self.classification_progress_bar.customClose)
            self.classification_progress_bar = None

    @Slot()
    def handle_open_train_classifier_dialog(self):
        if not self.data_manager or len(self.data_manager.classes) < 2:
            QMessageBox.warning(
                self.classes_view_widget,
                "Not Enough Classes",
                "At least two classes with assigned images are required to train a classifier.",
            )
            return

        # Check if any PyTorch models are available for feature extraction
        available_models = get_available_models()
        pytorch_models_exist = any(
            info.get("type") == "pytorch" for info in available_models.values()
        )
        if not pytorch_models_exist:
            QMessageBox.warning(
                self.classes_view_widget,
                "No PyTorch Models",
                "No PyTorch-based feature extractors are configured. Linear probing requires PyTorch models.",
            )
            return

        dialog = TrainClassifierDialog(parent=self.classes_view_widget.window())
        dialog.start_training_requested.connect(self._initiate_classifier_training)
        dialog.exec()

    @Slot(str, int, float, int, int)  # model_name, epochs, lr, batch_size, patience
    def _initiate_classifier_training(
        self, model_name: str, epochs: int, lr: float, batch_size: int, patience: int
    ):
        if not self.data_manager or not self.data_manager_interface:
            logging.error(
                "Cannot start training: DataManager or Interface not available."
            )
            return

        # 1. Prepare data for training: Dict[class_id, List[sample_ids]]
        # Only include classes that have samples assigned.
        classes_with_samples = {
            cls_id: [s.id for s in cls_obj.samples if s.is_active]
            for cls_id, cls_obj in self.data_manager.classes.items()
            if cls_obj.samples  # and any(s.is_active for s in cls_obj.samples) # Ensure active samples too
        }

        active_classes_count = 0
        total_samples_for_training = 0
        for cls_id, s_ids in classes_with_samples.items():
            if s_ids:  # If class has active samples
                active_classes_count += 1
                total_samples_for_training += len(s_ids)

        if active_classes_count < 2:
            InfoBar.warning(
                title="Training Error",
                content="Need at least two classes with active samples to train.",
                parent=self.classes_view_widget.window(),
                duration=5000,
            )
            return
        if total_samples_for_training == 0:
            InfoBar.warning(
                title="Training Error",
                content="No active samples found in classes for training.",
                parent=self.classes_view_widget.window(),
                duration=5000,
            )
            return

        logging.info(
            f"Initiating classifier training with model '{model_name}'. "
            f"Using {active_classes_count} classes and {total_samples_for_training} samples."
        )

        # For now, placeholder signal from ClassesPresenter
        self.request_train_classifier.emit(
            model_name, epochs, lr, batch_size, patience, classes_with_samples
        )

        # Show progress bar
        if self.training_progress_bar:
            self.training_progress_bar.customClose()  # Close existing
        self.training_progress_bar = ProgressInfoBar.new(
            icon=InfoBarIcon.INFORMATION,
            title="Classifier Training",
            content=f"Preparing data and training model: {model_name}...",
            duration=-1,
            position=InfoBarPosition.BOTTOM_RIGHT,
            parent=self.classes_view_widget.window(),
        )
        self.training_progress_bar.set_progress(0)  # Initial progress

    @Slot(str)  # classifier_name (which is likely the feature_model_name for now)
    def handle_run_classification(self, classifier_name: str):
        if not self.data_manager_interface:
            logging.error(
                "Cannot run classification: DataManagerInterface not available."
            )
            return

        if not classifier_name or classifier_name == "No Classifier Trained":
            QMessageBox.information(
                self.classes_view_widget,
                "No Classifier",
                "No classifier selected or available.",
            )
            return

        logging.info(
            f"Requesting classification of active samples using classifier based on: {classifier_name}"
        )
        # TODO: Emit a signal to DataManager to run classification on all active samples
        # using the specified trained probe (identified by `classifier_name`).
        # DataManager will extract features for active samples and use the probe.
        # Results (new class assignments) will come back via class_updated signals.

        # For now, placeholder signal from ClassesPresenter
        self.request_run_classification_in_dm.emit(classifier_name)

        if self.training_progress_bar:
            self.training_progress_bar.customClose()
        self.training_progress_bar = ProgressInfoBar.new(
            icon=InfoBarIcon.INFORMATION,
            title="Classification",
            content=f"Classifying images using '{classifier_name}' probe...",
            duration=-1,
            position=InfoBarPosition.BOTTOM_RIGHT,
            parent=self.classes_view_widget.window(),
        )
        self.training_progress_bar.set_progress(0)

    @Slot(str, bool, str)  # classifier_name, success_status, message
    def handle_classification_run_finished(
        self, classifier_name: str, success: bool, message: str
    ):
        if (
            self.training_progress_bar
        ):  # Re-using training_progress_bar for classification run status
            if success:
                self.training_progress_bar.set_title("Classification Complete")
                self.training_progress_bar.set_content(
                    f"Finished classifying with '{classifier_name}'. {message}"
                )
                self.training_progress_bar.setIcon(InfoBarIcon.SUCCESS)
                # Class updates will trigger UI refresh (on_class_updated -> _generate_class_preview)
            else:
                self.training_progress_bar.set_title("Classification Failed")
                self.training_progress_bar.set_content(
                    f"Classification with '{classifier_name}' failed: {message}"
                )
                self.training_progress_bar.setIcon(InfoBarIcon.ERROR)

            QTimer.singleShot(4000, self.training_progress_bar.customClose)
            self.training_progress_bar = None
