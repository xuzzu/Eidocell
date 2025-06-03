# UI/navigation_interface/workspace/views/analysis/analysis_controls.py (NEW FILE)

import logging

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import Slider
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    ComboBox,
    FluentIcon,
    PrimaryPushButton,
    SubtitleLabel,
)

from UI.common.style_sheet import EidocellStyleSheet

from .chart_configurations.histogram_config_widget import HistogramConfigWidget
from .chart_configurations.scatter_config_widget import ScatterConfigWidget


class AnalysisControlPanel(QWidget):
    """Control panel for selecting chart type and configuring its parameters."""

    create_chart_requested = Signal(str, dict)  # chart_type, parameters_dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent  # Reference to AnalysisViewWidget

        # Apply Stylesheet to the AnalysisControlPanel itself
        EidocellStyleSheet.ANALYSIS_CONTROL_PANEL.apply(self)

        self.main_styled_frame = QFrame(self)
        self.main_styled_frame.setObjectName("analysisControlPanelStyledFrame")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(1)
        shadow.setColor(QColor(0, 0, 0, 25))
        self.main_styled_frame.setGraphicsEffect(shadow)

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(self.main_styled_frame)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(outer_layout)

        self.vBoxLayout = QVBoxLayout(self.main_styled_frame)
        self.vBoxLayout.setContentsMargins(20, 20, 20, 20)
        self.vBoxLayout.setSpacing(15)

        title_label = QLabel("Chart Parameters", self.main_styled_frame)
        title_label.setObjectName("panelTitleLabel")  # Set object name for styling
        title_label.setAlignment(Qt.AlignCenter)
        self.vBoxLayout.addWidget(title_label)

        chart_type_group = QGroupBox("Chart Type", self.main_styled_frame)
        ct_layout = QVBoxLayout(chart_type_group)
        ct_layout.setContentsMargins(0, 10, 0, 0)  # Add top margin for group content
        self.chart_type_selector = ComboBox(chart_type_group)
        self.chart_type_selector.addItems(["Histogram", "Scatter Plot"])
        ct_layout.addWidget(self.chart_type_selector)
        self.vBoxLayout.addWidget(chart_type_group)

        self.chart_parameters_group = QGroupBox("Parameters", self.main_styled_frame)
        cp_layout = QVBoxLayout(self.chart_parameters_group)
        cp_layout.setContentsMargins(0, 10, 0, 0)  # Less top margin for stacked widget

        self.chart_params_stacked_widget = QStackedWidget(self.chart_parameters_group)
        self.histogram_config_widget = HistogramConfigWidget(
            self.chart_params_stacked_widget
        )
        self.scatter_config_widget = ScatterConfigWidget(
            self.chart_params_stacked_widget
        )

        self.chart_params_stacked_widget.addWidget(self.histogram_config_widget)
        self.chart_params_stacked_widget.addWidget(self.scatter_config_widget)
        cp_layout.addWidget(self.chart_params_stacked_widget)
        self.vBoxLayout.addWidget(self.chart_parameters_group)

        self.vBoxLayout.addStretch(1)  # Push button to bottom

        self.create_chart_button = PrimaryPushButton(
            FluentIcon.ADD_TO, "Create Chart", self.main_styled_frame
        )
        self.vBoxLayout.addWidget(self.create_chart_button)

        self.setFixedWidth(320)

        #  Connections
        self.chart_type_selector.currentTextChanged.connect(self._on_chart_type_changed)
        self.create_chart_button.clicked.connect(self._on_create_chart_clicked)

        # Initialize
        self._on_chart_type_changed(self.chart_type_selector.currentText())

    @Slot(str)
    def _on_chart_type_changed(self, chart_type_text: str):
        if chart_type_text == "Histogram":
            self.chart_params_stacked_widget.setCurrentWidget(
                self.histogram_config_widget
            )
            self.chart_parameters_group.setTitle("Histogram Parameters")
        elif chart_type_text == "Scatter Plot":
            self.chart_params_stacked_widget.setCurrentWidget(
                self.scatter_config_widget
            )
            self.chart_parameters_group.setTitle("Scatter Plot Parameters")
        # Adjust size of stacked widget based on current widget's size hint
        current_params_widget = self.chart_params_stacked_widget.currentWidget()
        if current_params_widget:
            self.chart_params_stacked_widget.setFixedHeight(
                current_params_widget.sizeHint().height() + 10
            )

    @Slot()
    def _on_create_chart_clicked(self):
        selected_chart_type_text = self.chart_type_selector.currentText()
        params_dict = {}

        if selected_chart_type_text == "Histogram":
            chart_type_key = "histogram"
            params_dict = (
                self.histogram_config_widget.get_parameters()
            )  # This returns HistogramParameters dataclass
        elif selected_chart_type_text == "Scatter Plot":
            chart_type_key = "scatter"
            params_dict = (
                self.scatter_config_widget.get_parameters()
            )  # This returns ScatterParameters dataclass
        else:
            logging.error(f"Unknown chart type selected: {selected_chart_type_text}")
            return

        print(f"Creating {selected_chart_type_text} with parameters: {params_dict}")
        self.create_chart_requested.emit(chart_type_key, params_dict)

    def set_data_manager_for_configs(self, data_manager):
        """
        Passes the data_manager to the config widgets if they need it
        (e.g., to populate ComboBoxes with available metadata columns).
        For now, HistogramConfigWidget and ScatterConfigWidget don't directly use it
        as they use a predefined list of 'properties'.
        If they were to populate choices from DataManager, this would be the place.
        """
        pass
