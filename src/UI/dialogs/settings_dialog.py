# UI/dialogs/settings_dialog.py
import os

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from qfluentwidgets import (
    ComboBox,
    LineEdit,
    MessageBoxBase,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    Slider,
    SubtitleLabel,
)

from backend.config import (  # Import get_available_models
    get_available_models,
    load_settings,
    save_settings,
)
from UI.common.style_sheet import EidocellStyleSheet


#  Helper Dialog for Adding/Editing Models
class AddEditModelDialog(QDialog):
    def __init__(self, existing_names, model_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            "Add/Edit ONNX Model" if model_data is None else "Edit ONNX Model"
        )
        self.existing_names = existing_names
        self.model_data = model_data or {}  # Store existing data if editing

        layout = QFormLayout(self)

        self.name_edit = LineEdit(self)
        self.name_edit.setPlaceholderText("Unique name for the model")
        layout.addRow("Model Name:", self.name_edit)

        self.path_layout = QHBoxLayout()
        self.path_edit = LineEdit(self)
        self.path_edit.setPlaceholderText("Path to .onnx file")
        self.path_edit.setReadOnly(True)  # Path selected via button
        self.browse_button = PushButton("Browse...", self)
        self.browse_button.clicked.connect(self.browse_for_model)
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.browse_button)
        layout.addRow("Model Path:", self.path_layout)

        self.dimension_spinbox = QSpinBox(self)
        self.dimension_spinbox.setRange(1, 10000)  # Adjust range as needed
        self.dimension_spinbox.setSuffix(" dimensions")
        layout.addRow("Output Dimension:", self.dimension_spinbox)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.ok_button = PrimaryPushButton("OK", self)
        self.cancel_button = PushButton("Cancel", self)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        layout.addRow(self.button_layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Populate fields if editing
        if model_data:
            self.name_edit.setText(model_data.get("name", ""))
            self.path_edit.setText(model_data.get("path", ""))
            self.dimension_spinbox.setValue(model_data.get("dimension", 1))

    def browse_for_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model File", "", "ONNX Models (*.onnx)"
        )
        if file_path:
            self.path_edit.setText(file_path)

    def get_model_data(self):
        name = self.name_edit.text().strip()
        path = self.path_edit.text().strip()
        dimension = self.dimension_spinbox.value()

        #  Validation
        is_editing = bool(self.model_data)
        original_name = self.model_data.get("name") if is_editing else None

        if not name:
            QMessageBox.warning(self, "Input Error", "Model name cannot be empty.")
            return None
        if name != original_name and name in self.existing_names:
            QMessageBox.warning(
                self, "Input Error", f"Model name '{name}' already exists."
            )
            return None
        if not path:
            QMessageBox.warning(self, "Input Error", "Model path cannot be empty.")
            return None
        if not os.path.exists(path):
            QMessageBox.warning(
                self, "Input Error", f"Model path does not exist:\n{path}"
            )
            return None
        if dimension <= 0:
            QMessageBox.warning(
                self, "Input Error", "Output dimension must be greater than 0."
            )
            return None
        #  End Validation

        return {"name": name, "path": path, "dimension": dimension}


#  Main Settings Dialog
class SettingsDialog(MessageBoxBase):
    """Dialog for configuring application settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.settings = load_settings()  # Load initial settings

        #  General Settings
        self.generalSettingsGroup = QGroupBox("General Settings", self)
        generalLayout = QVBoxLayout(self.generalSettingsGroup)

        self.themeLabel = QLabel("Theme:", self)
        self.themeSelector = SegmentedWidget(self)
        self.themeSelector.setFixedHeight(30)
        self.themeSelector.addItem(routeKey="light", text="Light")
        self.themeSelector.addItem(routeKey="dark", text="Dark")
        self.themeSelector.setCurrentItem(self.settings.get("theme", "light"))

        self.providerLabel = QLabel("ONNX Execution Provider:", self)
        self.providerComboBox = ComboBox(self)
        self.providerComboBox.addItems(
            ["CPUExecutionProvider", "CUDAExecutionProvider"]
        )  # Add others if needed
        self.providerComboBox.setCurrentText(
            self.settings.get("provider", "CPUExecutionProvider")
        )

        generalLayout.addWidget(self.themeLabel)
        generalLayout.addWidget(self.themeSelector)
        generalLayout.addWidget(self.providerLabel)
        generalLayout.addWidget(self.providerComboBox)

        #  Display Settings
        self.displaySettingsGroup = QGroupBox("Display Settings", self)
        displayLayout = QFormLayout(
            self.displaySettingsGroup
        )  # Use QFormLayout for label/widget pairs

        self.thumbnailQualitySlider = Slider(Qt.Horizontal, self)
        self.thumbnailQualitySlider.setRange(1, 100)
        self.thumbnailQualityValueLabel = QLabel(
            str(self.settings.get("thumbnail_quality", 75)), self
        )
        self.thumbnailQualitySlider.setValue(self.settings.get("thumbnail_quality", 75))
        self.thumbnailQualitySlider.valueChanged.connect(
            lambda v: self.thumbnailQualityValueLabel.setText(str(v))
        )
        thumbLayout = QHBoxLayout()
        thumbLayout.addWidget(self.thumbnailQualitySlider)
        thumbLayout.addWidget(self.thumbnailQualityValueLabel)
        displayLayout.addRow("Thumbnail Quality:", thumbLayout)

        self.collageImagesSlider = Slider(Qt.Horizontal, self)
        self.collageImagesSlider.setRange(1, 50)  # Increased range
        self.collageImagesValueLabel = QLabel(
            str(self.settings.get("images_per_collage", 25)), self
        )
        self.collageImagesSlider.setValue(self.settings.get("images_per_collage", 25))
        self.collageImagesSlider.valueChanged.connect(
            lambda v: self.collageImagesValueLabel.setText(str(v))
        )
        collageLayout = QHBoxLayout()
        collageLayout.addWidget(self.collageImagesSlider)
        collageLayout.addWidget(self.collageImagesValueLabel)
        displayLayout.addRow("Images per Collage:", collageLayout)

        #  Model Management
        self.modelManagementGroup = QGroupBox(
            "Feature Extractor Model Management", self
        )
        modelMgmtLayout = QVBoxLayout(self.modelManagementGroup)

        self.selectedModelLabel = QLabel("Selected Model:", self)
        self.modelComboBox = ComboBox(self)  # ComboBox to SELECT the active model
        modelMgmtLayout.addWidget(self.selectedModelLabel)
        modelMgmtLayout.addWidget(self.modelComboBox)
        modelMgmtLayout.addSpacing(15)

        self.customModelsLabel = QLabel("Custom Models:", self)
        self.customModelsList = QListWidget(
            self
        )  # QListWidget to display custom models
        self.customModelsList.setObjectName("customModelsList")
        self.customModelsList = QListWidget(self)
        self.customModelsList.setAlternatingRowColors(True)
        modelMgmtLayout.addWidget(self.customModelsLabel)
        modelMgmtLayout.addWidget(self.customModelsList)

        modelButtonsLayout = QHBoxLayout()
        self.addModelButton = PushButton("Add Model...", self)
        self.removeModelButton = PushButton("Remove Selected", self)
        self.removeModelButton.setEnabled(False)  # Disabled initially
        modelButtonsLayout.addStretch()
        modelButtonsLayout.addWidget(self.addModelButton)
        modelButtonsLayout.addWidget(self.removeModelButton)
        modelMgmtLayout.addLayout(modelButtonsLayout)

        self.__initWidget()

    def __initWidget(self):
        """Initialize the layout and widgets within the dialog."""
        # Add groups to the main layout
        self.viewLayout.addWidget(self.generalSettingsGroup)
        self.viewLayout.addWidget(self.displaySettingsGroup)
        self.viewLayout.addWidget(self.modelManagementGroup)

        self.yesButton.setText("Apply")
        self.cancelButton.setText("Cancel")
        self.widget.setMinimumWidth(550)  # Increased width
        self.widget.setMinimumHeight(600)  # Increased height

        # Populate model selection combo box
        self.populate_model_combobox()

        # Populate custom models list
        self.populate_custom_models_list()

        # Connect signals
        self.addModelButton.clicked.connect(self.add_model)
        self.removeModelButton.clicked.connect(self.remove_model)
        self.customModelsList.currentItemChanged.connect(
            lambda current, previous: self.removeModelButton.setEnabled(
                current is not None
            )
        )

    def populate_model_combobox(self):
        """Populates the model selection ComboBox."""
        self.modelComboBox.clear()
        available_models = get_available_models()
        current_selection = self.settings.get("selected_model", "mobilenetv3s")

        for name in available_models.keys():
            self.modelComboBox.addItem(name)

        if current_selection in available_models:
            self.modelComboBox.setCurrentText(current_selection)
        elif available_models:
            # Fallback if saved selection is invalid
            first_model_name = list(available_models.keys())[0]
            self.modelComboBox.setCurrentText(first_model_name)
            self.settings["selected_model"] = first_model_name  # Update setting

    def populate_custom_models_list(self):
        """Populates the QListWidget with custom models from settings."""
        self.customModelsList.clear()
        for model_data in self.settings.get("custom_models", []):
            name = model_data.get("name", "N/A")
            path = model_data.get("path", "N/A")
            dim = model_data.get("dimension", "N/A")
            item_text = f"{name}  |  Dim: {dim}  |  Path: {path}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model_data)  # Store full data with the item
            self.customModelsList.addItem(item)

    @Slot()
    def add_model(self):
        """Handles adding a new custom model."""
        existing_names = [m["name"] for m in self.settings.get("custom_models", [])]
        dialog = AddEditModelDialog(existing_names, parent=self)
        if dialog.exec():
            new_model_data = dialog.get_model_data()
            if new_model_data:
                self.settings.setdefault("custom_models", []).append(new_model_data)
                self.populate_custom_models_list()
                self.populate_model_combobox()  # Update selector in case name was added/changed

    @Slot()
    def remove_model(self):
        """Handles removing the selected custom model."""
        selected_item = self.customModelsList.currentItem()
        if not selected_item:
            return

        model_data_to_remove = selected_item.data(Qt.UserRole)
        if not model_data_to_remove:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove the model '{model_data_to_remove.get('name')}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if confirm == QMessageBox.Yes:
            custom_models = self.settings.get("custom_models", [])
            # Find and remove the model dictionary
            self.settings["custom_models"] = [
                m for m in custom_models if m != model_data_to_remove
            ]
            self.populate_custom_models_list()
            self.populate_model_combobox()  # Update selector
            self.removeModelButton.setEnabled(False)  # Disable button after removal

    def accept(self):
        """Save settings when the dialog is accepted."""
        self.settings["theme"] = self.themeSelector.currentRouteKey()
        self.settings["selected_model"] = self.modelComboBox.currentText()
        self.settings["provider"] = self.providerComboBox.currentText()
        self.settings["thumbnail_quality"] = self.thumbnailQualitySlider.value()
        self.settings["images_per_collage"] = self.collageImagesSlider.value()
        # Custom models list is already updated by add/remove actions

        save_settings(self.settings)
        super().accept()
