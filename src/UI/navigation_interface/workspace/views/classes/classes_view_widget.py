# src/UI/navigation_interface/workspace/views/classes/classes_view_widget.py
import logging
import os
import random

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import ComboBox
from qfluentwidgets import (
    BodyLabel,
    FluentIcon,
    Flyout,
    FlyoutView,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    TreeWidget,
)
from qfluentwidgets.components.widgets.flyout import FlyoutAnimationType

from UI.common.style_sheet import EidocellStyleSheet
from UI.navigation_interface.workspace.views.classes.class_card import ClassCard
from UI.utils.flow_gallery import FlowGallery


class ClassesViewWidget(QWidget):
    create_class_requested = Signal(str, str)
    open_train_classifier_dialog_requested = Signal()
    run_classification_requested = Signal(str)  # Pass classifier_id/name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("classesView")
        self.classes = []
        EidocellStyleSheet.CLASSES_VIEW_WIDGET.apply(self)

        self.class_gallery = FlowGallery(self)
        self.class_gallery.flow_layout.needAni = True
        self.class_gallery.flow_layout.duration = 150

        self.class_frame = QFrame(self)
        self.class_frame.setObjectName("classCreationTreeFrame")
        self.class_frame.setFrameShape(QFrame.NoFrame)
        shadow_tree = QGraphicsDropShadowEffect(self)
        shadow_tree.setBlurRadius(8)
        shadow_tree.setOffset(0, 0)
        shadow_tree.setColor(QColor(0, 0, 0, 20))
        self.class_frame.setGraphicsEffect(shadow_tree)

        self.tree_layout = QVBoxLayout(self.class_frame)
        self.tree_layout.setContentsMargins(10, 10, 10, 10)
        self.tree_layout.setSpacing(10)

        #  Class Creation Controls
        cc_title_lbl = QLabel("Create New Class", self.class_frame)
        cc_title_lbl.setObjectName("creationTitleLabel")
        self.tree_layout.addWidget(cc_title_lbl)
        self.new_class_name_edit = LineEdit(self.class_frame)
        self.new_class_name_edit.setPlaceholderText("Enter class name…")
        self.tree_layout.addWidget(self.new_class_name_edit)
        cc_colour_lbl = QLabel("Select Color", self.class_frame)
        cc_colour_lbl.setObjectName("colorSelectionLabel")
        self.tree_layout.addWidget(cc_colour_lbl)
        self.class_color_palette_layout = QHBoxLayout()
        self.class_color_palette_layout.setSpacing(5)
        self.class_color_buttons = []
        self.selected_class_color_hex = None
        default_colors = [
            "#A162D8",
            "#F47070",
            "#4CAF50",
            "#2196F3",
            "#FFC107",
            "#FF9800",
            "#E91E63",
            "#00BCD4",
            "#8BC34A",
        ]
        for hex_ in default_colors:
            btn = PushButton("", self.class_frame)
            btn.setProperty("class", "colorPaletteButton")
            btn.setFixedSize(22, 22)
            btn.setCheckable(True)
            btn.setProperty("color_hex", hex_)
            btn.setStyleSheet(f"background-color:{hex_};")
            btn.clicked.connect(self._on_class_color_selected)
            self.class_color_palette_layout.addWidget(btn)
            self.class_color_buttons.append(btn)
        self.class_color_palette_layout.addStretch()
        self.tree_layout.addLayout(self.class_color_palette_layout)
        self.create_class_button = PrimaryPushButton(
            FluentIcon.ADD, "Create Class", self.class_frame
        )
        self.create_class_button.setFixedHeight(35)
        self.create_class_button.clicked.connect(self._handle_create_class)
        self.tree_layout.addWidget(self.create_class_button)

        #  Class Hierarchy Tree
        self.class_tree_view = TreeWidget(self)
        self.class_tree_view.setObjectName("classHierarchyTree")
        self.class_tree_view.setHeaderHidden(True)
        self.class_tree_view.setDragDropMode(QAbstractItemView.InternalMove)
        self.class_tree_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        ch_tree_lbl = QLabel("Class Hierarchy", self.class_frame)
        ch_tree_lbl.setObjectName("classHierarchyTreeLabel")
        ch_tree_lbl.setAlignment(Qt.AlignCenter)
        self.tree_layout.addWidget(ch_tree_lbl)
        self.tree_layout.addWidget(self.class_tree_view, 1)  # Tree takes stretch factor

        #  Linear Probe / Classifier Controls
        self.classifier_controls_frame = QFrame(
            self
        )  # New frame for classifier controls
        self.classifier_controls_frame.setObjectName("classifierControlsFrame")
        shadow_classifier = QGraphicsDropShadowEffect(self)
        shadow_classifier.setBlurRadius(8)
        shadow_classifier.setOffset(0, 0)
        shadow_classifier.setColor(QColor(0, 0, 0, 15))
        self.classifier_controls_frame.setGraphicsEffect(shadow_classifier)

        classifier_layout = QVBoxLayout(self.classifier_controls_frame)
        classifier_layout.setContentsMargins(10, 10, 10, 10)
        classifier_layout.setSpacing(10)

        self.train_classifier_button = PushButton(
            "Train Classifier", self.classifier_controls_frame
        )
        self.train_classifier_button.setFixedHeight(35)
        classifier_layout.addWidget(self.train_classifier_button)

        trained_classifiers_layout = QHBoxLayout()
        self.trained_classifiers_combo = ComboBox(self.classifier_controls_frame)
        self.trained_classifiers_combo.addItem("No Classifier Trained")  # Placeholder
        self.trained_classifiers_combo.setEnabled(
            False
        )  # Enable when classifiers are available
        trained_classifiers_layout.addWidget(
            self.trained_classifiers_combo, 1
        )  # Combobox takes stretch
        classifier_layout.addLayout(trained_classifiers_layout)

        self.run_classification_button = PrimaryPushButton(
            FluentIcon.PLAY, "Run Classification", self.classifier_controls_frame
        )
        self.run_classification_button.setFixedHeight(35)
        self.run_classification_button.setEnabled(
            False
        )  # Enable when a classifier is selected
        classifier_layout.addWidget(self.run_classification_button)

        # Add classifier controls frame to the tree_layout (below the tree)
        self.tree_layout.addWidget(self.classifier_controls_frame)

        self.class_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.class_frame.setMinimumWidth(280)  # Adjusted width for controls

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(20)
        self.main_layout.addWidget(self.class_gallery, 3)
        self.main_layout.addWidget(
            self.class_frame, 1
        )  # .class_frame now contains creation & classifier controls

        self.setLayout(self.main_layout)
        self.classes_presenter = None
        self._connect_classifier_signals()  # New method to connect these

    def _connect_classifier_signals(self):
        self.train_classifier_button.clicked.connect(
            self.open_train_classifier_dialog_requested
        )
        self.run_classification_button.clicked.connect(self._handle_run_classification)
        self.trained_classifiers_combo.currentTextChanged.connect(
            self._on_classifier_selected
        )

    def _on_classifier_selected(self, classifier_name: str):
        self.run_classification_button.setEnabled(
            classifier_name != "No Classifier Trained" and classifier_name != ""
        )

    def _handle_run_classification(self):
        selected_classifier = self.trained_classifiers_combo.currentText()
        if selected_classifier != "No Classifier Trained" and selected_classifier != "":
            self.run_classification_requested.emit(selected_classifier)
        else:
            QMessageBox.information(
                self, "No Classifier", "Please select a trained classifier to run."
            )

    def update_trained_classifiers_list(self, classifier_names: list):
        self.trained_classifiers_combo.blockSignals(True)
        self.trained_classifiers_combo.clear()
        if classifier_names:
            self.trained_classifiers_combo.addItems(classifier_names)
            self.trained_classifiers_combo.setCurrentIndex(0)  # Select first one
            self.trained_classifiers_combo.setEnabled(True)
            self.run_classification_button.setEnabled(True)
        else:
            self.trained_classifiers_combo.addItem("No Classifier Trained")
            self.trained_classifiers_combo.setEnabled(False)
            self.run_classification_button.setEnabled(False)
        self.trained_classifiers_combo.blockSignals(False)
        self._on_classifier_selected(
            self.trained_classifiers_combo.currentText()
        )  # Update button state

    def _on_class_color_selected(self):
        clicked_button = self.sender()
        if not isinstance(clicked_button, PushButton):
            return
        self.selected_class_color_hex = clicked_button.property("color_hex")
        for btn in self.class_color_buttons:
            if btn != clicked_button:
                btn.setChecked(False)
        if not clicked_button.isChecked():
            clicked_button.setChecked(True)
        logging.debug(f"Class color selected: {self.selected_class_color_hex}")

    def _handle_create_class(self):
        class_name = self.new_class_name_edit.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Input Error", "Class name cannot be empty.")
            self.new_class_name_edit.setFocus()
            return
        if not self.selected_class_color_hex:
            if self.class_color_buttons:  # Auto-select first color if none selected
                self.class_color_buttons[0].setChecked(True)
                self.selected_class_color_hex = self.class_color_buttons[0].property(
                    "color_hex"
                )
            else:  # Absolute fallback (should not happen)
                self.selected_class_color_hex = "#{:06x}".format(
                    random.randint(0, 0xFFFFFF)
                )
            logging.warning(
                f"No color explicitly selected, using: {self.selected_class_color_hex}"
            )
        if self.classes_presenter and self.classes_presenter.data_manager:
            if self.classes_presenter.data_manager.get_class_by_name(class_name):
                QMessageBox.warning(
                    self, "Duplicate Name", "A class with this name already exists."
                )
                return
        logging.info(
            f"Requesting creation of class: '{class_name}' with color {self.selected_class_color_hex}"
        )
        self.create_class_requested.emit(class_name, self.selected_class_color_hex)
        self.new_class_name_edit.clear()
        if self.class_color_buttons:  # Reset color selection to first
            self.class_color_buttons[0].setChecked(True)
            self.selected_class_color_hex = self.class_color_buttons[0].property(
                "color_hex"
            )
        self._randomize_colors()

    def _randomize_colors(self):
        for button in self.class_color_buttons:
            random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            button.setProperty("color_hex", random_color)
            # Re-apply dynamic stylesheet for the button
            button.setStyleSheet(
                f"""
                QPushButton[class="colorPaletteButton"] {{ 
                    background-color: {random_color}; 
                    border-radius: 3px; 
                    border: 1px solid transparent; 
                }}
                QPushButton[class="colorPaletteButton"]:hover {{ 
                    border: 1px solid #707070; 
                }}
                QPushButton[class="colorPaletteButton"]:checked {{ 
                    border: 2px solid black; 
                }}
            """
            )
        if (
            self.class_color_buttons
        ):  # If any buttons exist, set the first one as selected
            self.class_color_buttons[0].setChecked(True)
            self.selected_class_color_hex = self.class_color_buttons[0].property(
                "color_hex"
            )

    def create_class_card(self, class_name, class_id, class_color, preview_image_path):
        if not preview_image_path or not os.path.exists(preview_image_path):
            logging.warning(
                f"Preview image path invalid for class {class_id}: {preview_image_path}. Card not created."
            )
            # Optionally create a placeholder card or show an error indicator
            return None

        card = ClassCard(
            iconPath=preview_image_path,
            classes_presenter=self.classes_presenter,
            class_id=class_id,
            parent=self,
        )  # Parent to self for layout management by FlowGallery
        self.classes.append(card)
        self.class_gallery.flow_layout.addWidget(card)
        return card

    def delete_class_card(self, class_id):
        card_to_delete = next(
            (card for card in self.classes if card.class_id == class_id), None
        )
        if card_to_delete:
            self.class_gallery.flow_layout.removeWidget(card_to_delete)
            self.classes.remove(card_to_delete)
            card_to_delete.deleteLater()

    def update_class_card(
        self, class_id, preview_image_path, new_name=None, new_color_hex=None
    ):
        card_to_update = next(
            (card for card in self.classes if card.class_id == class_id), None
        )
        if card_to_update:
            effective_name = new_name
            effective_color = new_color_hex
            if self.classes_presenter and self.classes_presenter.data_manager:
                class_obj = self.classes_presenter.data_manager.get_class(class_id)
                if class_obj:
                    if new_name is None:
                        effective_name = class_obj.name
                    if new_color_hex is None:
                        effective_color = class_obj.color

            # Check if preview_image_path is valid before trying to load
            if preview_image_path and os.path.exists(preview_image_path):
                card_to_update.update_display(
                    preview_image_path, effective_name, effective_color
                )
            else:
                logging.warning(
                    f"Invalid preview path for class card update {class_id}: {preview_image_path}. Updating text only."
                )
                card_to_update.label.setText(
                    effective_name or card_to_update.class_name
                )
        else:
            logging.warning(f"Could not find class card with ID {class_id} to update.")

    def set_presenter(self, presenter):
        self.classes_presenter = presenter
        # Connect new signals to presenter slots here if they exist
        self.open_train_classifier_dialog_requested.connect(
            self.classes_presenter.handle_open_train_classifier_dialog
        )
        self.run_classification_requested.connect(
            self.classes_presenter.handle_run_classification
        )

    def clear_classes(self) -> None:
        logging.info("Clearing all classes from ClassesViewWidget.")
        for card in list(self.classes):  # Iterate over a copy
            self.class_gallery.flow_layout.removeWidget(card)
            card.deleteLater()
        self.classes.clear()
        self.class_tree_view.clear()
        logging.info("All classes cleared from ClassesViewWidget.")
