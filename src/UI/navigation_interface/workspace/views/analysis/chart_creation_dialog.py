# UI/navigation_interface/workspace/views/analysis/chart_configurations/chart_creation_dialog.py

from PySide6.QtWidgets import QStackedWidget, QWidget
from qfluentwidgets import MessageBoxBase, SegmentedWidget, SubtitleLabel

from backend.data_manager import DataManager
from backend.plot_generator import PlotGenerator
from UI.navigation_interface.workspace.views.analysis.chart_configurations.histogram_config_widget import (
    HistogramConfigWidget,
)
from UI.navigation_interface.workspace.views.analysis.chart_configurations.scatter_config_widget import (
    ScatterConfigWidget,
)


class ChartCreationDialog(MessageBoxBase):
    """Dialog for creating custom charts."""

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Chart")
        self.data_manager = data_manager
        self.plot_generator = PlotGenerator(self.data_manager)

        self.titleLabel = SubtitleLabel("Select Chart Type", self)
        self.chartTypeSelector = SegmentedWidget(self)
        self.chartTypeSelector.setFixedHeight(30)
        self.chartOptionsStackedWidget = QStackedWidget(self)

        # Initialize plot options widgets
        self.histogramConfig = HistogramConfigWidget(self)
        self.scatterConfig = ScatterConfigWidget(self)

        self.__initWidget()

    def __initWidget(self):
        """Initialize the layout and widgets within the dialog."""
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.chartTypeSelector)
        self.viewLayout.addWidget(self.chartOptionsStackedWidget)

        self.yesButton.setText("Create")
        self.cancelButton.setText("Cancel")
        self.widget.setMinimumWidth(800)

        # Add chart type options to the segmented widget and stacked widget
        self._addChartType("histogram", "Histogram", self.histogramConfig)
        self._addChartType("scatter", "Scatter Plot", self.scatterConfig)

        # Set initial chart type
        self.chartOptionsStackedWidget.setCurrentWidget(self.histogramConfig)
        self.chartTypeSelector.setCurrentItem("histogram")

        # Connect the chart type selector to switch between options
        self.chartTypeSelector.currentItemChanged.connect(
            lambda k: self.chartOptionsStackedWidget.setCurrentWidget(
                self._find_chart_options_widget(k)
            )
        )

        # Connect the Create button to plot generation
        self.yesButton.clicked.connect(self.create_plot)

    def _addChartType(self, route_key: str, label: str, widget: QWidget):
        """Helper function to add chart types to the segmented widget and stacked widget."""
        widget.setObjectName(route_key + "Options")
        self.chartOptionsStackedWidget.addWidget(widget)
        self.chartTypeSelector.addItem(routeKey=route_key, text=label)

    def _find_chart_options_widget(self, route_key: str) -> QWidget:
        """Finds the chart options widget based on the route key."""
        for i in range(self.chartOptionsStackedWidget.count()):
            widget = self.chartOptionsStackedWidget.widget(i)
            if widget.objectName() == f"{route_key}Options":
                return widget
        return self.histogramConfig  # Default fallback

    def create_plot(self):
        """Handles the Create button click to generate and display the plot."""
        pass

    def get_chart_parameters(self, selected_chart_type: str):
        """Extracts parameters based on the selected chart type."""
        if selected_chart_type == "histogram":
            parameters = (
                self.histogramConfig.get_parameters()
            )  # Assuming histogramOptions is your HistogramConfigWidget
            return parameters
        elif selected_chart_type == "scatter":
            parameters = self.scatterConfig.get_parameters()
            return parameters
