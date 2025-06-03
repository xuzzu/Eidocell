# UI/dialogs/train_classifier_dialog.py
import logging

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import ComboBox
from qfluentwidgets import (
    DoubleSpinBox as FluentDoubleSpinBox,
)  # Use Fluent versions if preferred
from qfluentwidgets import LineEdit, MessageBoxBase, PrimaryPushButton, PushButton
from qfluentwidgets import SpinBox as FluentSpinBox
from qfluentwidgets import SubtitleLabel

from backend.config import get_available_models  # To populate model selector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TrainClassifierDialog(MessageBoxBase):
    # (feature_model_name, epochs, learning_rate, batch_size, early_stopping_patience)
    start_training_requested = Signal(str, int, float, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Linear Classifier")

        #  Main Layout (using self.viewLayout from MessageBoxBase)
        self.viewLayout.setSpacing(15)

        #  Title
        title = SubtitleLabel("Classifier Training Configuration", self)
        self.viewLayout.addWidget(title, 0, Qt.AlignCenter)

        #  Model Selection Group
        model_group = QGroupBox("Feature Model", self)
        model_layout = QFormLayout(
            model_group
        )  # Use QFormLayout for label-widget pairs
        self.feature_model_combo = ComboBox(model_group)
        self._populate_feature_models()
        model_layout.addRow("Base Feature Model:", self.feature_model_combo)
        self.viewLayout.addWidget(model_group)

        #  Training Parameters Group
        training_params_group = QGroupBox("Training Parameters", self)
        tp_layout = QFormLayout(training_params_group)

        self.epochs_spinbox = FluentSpinBox(
            training_params_group
        )  # Using FluentSpinBox
        self.epochs_spinbox.setRange(10, 1000)
        self.epochs_spinbox.setValue(100)
        self.epochs_spinbox.setSuffix(" epochs")
        tp_layout.addRow("Number of Epochs:", self.epochs_spinbox)

        self.lr_spinbox = FluentDoubleSpinBox(
            training_params_group
        )  # FluentDoubleSpinBox
        self.lr_spinbox.setRange(1e-6, 1e-1)
        self.lr_spinbox.setValue(0.001)
        self.lr_spinbox.setDecimals(5)
        self.lr_spinbox.setSingleStep(0.0001)
        tp_layout.addRow("Learning Rate:", self.lr_spinbox)

        self.batch_size_spinbox = FluentSpinBox(training_params_group)
        self.batch_size_spinbox.setRange(4, 256)
        self.batch_size_spinbox.setValue(32)
        self.batch_size_spinbox.setSuffix(" batch size")
        tp_layout.addRow("Batch Size:", self.batch_size_spinbox)

        # Early Stopping (Optional)
        self.early_stopping_checkbox = QCheckBox(
            "Enable Early Stopping", training_params_group
        )
        tp_layout.addRow(self.early_stopping_checkbox)
        self.early_stopping_patience_spinbox = FluentSpinBox(training_params_group)
        self.early_stopping_patience_spinbox.setRange(1, 50)
        self.early_stopping_patience_spinbox.setValue(10)
        self.early_stopping_patience_spinbox.setSuffix(" epochs patience")
        self.early_stopping_patience_spinbox.setEnabled(
            False
        )  # Enable if checkbox is checked
        tp_layout.addRow("Patience:", self.early_stopping_patience_spinbox)
        self.early_stopping_checkbox.toggled.connect(
            self.early_stopping_patience_spinbox.setEnabled
        )

        self.viewLayout.addWidget(training_params_group)

        #  Buttons
        self.yesButton.setText("Start Training")
        self.cancelButton.setText("Cancel")
        self.yesButton.clicked.connect(self._on_start_training_clicked)

        #  Dialog Size
        self.widget.setMinimumWidth(450)
        self.widget.setMinimumHeight(400)  # Adjust as needed

    def _populate_feature_models(self):
        self.feature_model_combo.clear()
        available_models = get_available_models()
        # Filter for PyTorch models as linear probing is currently tied to PyTorchFeatureExtractor
        pytorch_models = {
            name: info
            for name, info in available_models.items()
            if info.get("type") == "pytorch"
        }

        if pytorch_models:
            self.feature_model_combo.addItems(list(pytorch_models.keys()))
            self.feature_model_combo.setCurrentIndex(0)
        else:
            self.feature_model_combo.addItem("No PyTorch models available")
            self.feature_model_combo.setEnabled(False)
            self.yesButton.setEnabled(False)  # Cannot train without a model

    @Slot()
    def _on_start_training_clicked(self):
        selected_model = self.feature_model_combo.currentText()
        if selected_model == "No PyTorch models available" or not selected_model:
            QMessageBox.warning(
                self, "Model Error", "Please select a valid PyTorch feature model."
            )
            return

        epochs = self.epochs_spinbox.value()
        learning_rate = self.lr_spinbox.value()
        batch_size = self.batch_size_spinbox.value()
        early_stopping_patience = (
            self.early_stopping_patience_spinbox.value()
            if self.early_stopping_checkbox.isChecked()
            else -1
        )  # -1 to indicate disabled

        logging.info(
            f"Requesting classifier training: Model={selected_model}, Epochs={epochs}, LR={learning_rate}, Batch={batch_size}, Patience={early_stopping_patience}"
        )
        self.start_training_requested.emit(
            selected_model, epochs, learning_rate, batch_size, early_stopping_patience
        )
        self.accept()  # Close the dialog after emitting signal
