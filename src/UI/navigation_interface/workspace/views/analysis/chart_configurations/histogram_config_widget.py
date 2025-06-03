# UI/navigation_interface/workspace/views/analysis/chart_configurations/histogram_config_widget.py

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CheckBox,
    ComboBox,
    CompactDoubleSpinBox,
    CompactSpinBox,
    RadioButton,
)

from UI.common.style_sheet import EidocellStyleSheet
from UI.navigation_interface.workspace.views.analysis.chart_configurations.parameter_holders import (
    HistogramParameters,
)


class HistogramConfigWidget(QWidget):
    params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Apply Stylesheet
        EidocellStyleSheet.HISTOGRAM_CONFIG_WIDGET.apply(self)

        # Store properties
        self.properties = [
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
        self.default_params = HistogramParameters(
            x_variable=self.properties[0],
            num_bins=10,
            show_mean=True,
            relative_frequency=False,
        )

        # Create UI elements
        self.create_x_variable_selector()
        self.create_num_bins_selector()
        self.create_additional_params()

        # Arrange layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.x_variable_group)
        main_layout.addWidget(self.num_bins_group)
        main_layout.addWidget(self.additional_params_group)
        main_layout.addStretch()

        self.setLayout(main_layout)

        # Connect signals to handlers
        self.x_variable_combobox.currentTextChanged.connect(self.on_x_variable_changed)
        self.num_bins_spinbox.valueChanged.connect(self.on_num_bins_changed)
        self.show_mean_checkbox.stateChanged.connect(self.on_show_mean_changed)
        self.relative_freq_checkbox.stateChanged.connect(self.on_relative_freq_changed)

    def create_x_variable_selector(self):
        self.x_variable_group = QGroupBox("Select X Variable")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        self.x_variable_combobox = ComboBox(self.x_variable_group)
        self.x_variable_combobox.addItems(self.properties)
        self.x_variable_combobox.setCurrentText(self.default_params.x_variable)
        layout.addWidget(QLabel("X-axis:"))
        layout.addWidget(self.x_variable_combobox)
        self.x_variable_group.setLayout(layout)

    def create_num_bins_selector(self):
        self.num_bins_group = QGroupBox("Number of Bins")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        self.num_bins_spinbox = CompactDoubleSpinBox(self.num_bins_group)
        self.num_bins_spinbox.setRange(10, 1000)
        layout.addWidget(QLabel("Bins:"))
        layout.addWidget(self.num_bins_spinbox)
        self.num_bins_group.setLayout(layout)

    def create_additional_params(self):
        self.additional_params_group = QGroupBox("Additional Parameters")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        self.show_mean_checkbox = CheckBox("Show Global Mean")
        self.show_mean_checkbox.setChecked(self.default_params.show_mean)
        self.relative_freq_checkbox = CheckBox("Relative Frequency")
        self.relative_freq_checkbox.setChecked(self.default_params.relative_frequency)
        layout.addWidget(self.show_mean_checkbox)
        layout.addWidget(self.relative_freq_checkbox)
        self.additional_params_group.setLayout(layout)

    def on_x_variable_changed(self, text):
        self.default_params.x_variable = text
        self.emit_parameters_changed()

    def on_num_bins_changed(self, value):
        self.default_params.num_bins = value
        self.emit_parameters_changed()

    def on_show_mean_changed(self, state):
        self.default_params.show_mean = state == Qt.Checked
        self.emit_parameters_changed()

    def on_relative_freq_changed(self, state):
        self.default_params.relative_frequency = state == Qt.Checked
        self.emit_parameters_changed()

    def emit_parameters_changed(self):
        self.params_changed.emit()

    def get_parameters(self):
        """Returns the current parameter settings for the histogram."""
        return {
            "x_variable": self.x_variable_combobox.currentText(),
            "num_bins": int(self.num_bins_spinbox.value()),
            "show_mean": self.show_mean_checkbox.isChecked(),
            "relative_frequency": self.relative_freq_checkbox.isChecked(),
        }
