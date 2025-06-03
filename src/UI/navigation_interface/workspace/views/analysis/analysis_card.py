from pathlib import Path

# UI/navigation_interface/workspace/views/analysis/analysis_card.py
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QMenu, QSizePolicy, QVBoxLayout
from qfluentwidgets import CaptionLabel, CardWidget, isDarkTheme
from qframelesswindow.webengine import FramelessWebEngineView

from backend.config import (  # Keep if used for default sizing
    ANALYSIS_CARD_HEIGHT,
    ANALYSIS_CARD_WIDTH,
)
from UI.navigation_interface.workspace.views.analysis.plot_widget.interactive_chart_widget import (
    InteractiveChartWidget,
)


class AnalysisCard(CardWidget):
    """Card for displaying an InteractiveChartWidget."""

    deleteRequested = Signal(str)  # plot_id for deletion

    def __init__(
        self,
        interactive_chart_widget_instance: InteractiveChartWidget,
        plot_name: str,
        plot_id: str,
        parent=None,
    ):
        super().__init__(parent)
        self._borderRadius = 6  # Example, adjust as needed
        self.plot_id = plot_id  # Store the plot_id

        self.interactive_chart_widget = interactive_chart_widget_instance
        self.titleLabel = CaptionLabel(plot_name, self)  # Display plot name
        self.titleLabel.setAlignment(Qt.AlignCenter)

        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(8, 8, 8, 8)  # Card padding
        self.vBoxLayout.setSpacing(5)

        # Ensure ICW takes available space
        self.interactive_chart_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.titleLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.interactive_chart_widget, 1)

        self.setMinimumSize(350, 300)

        self.setStyleSheet(
            f"""
            AnalysisCard {{
                background-color: {'#2C2C2C' if isDarkTheme() else '#FFFFFF'};
                border-radius: {self._borderRadius}px;
                /* Add border for testing if needed */
                /* border: 1px solid {'#404040' if isDarkTheme() else '#E0E0E0'}; */
            }}
            CaptionLabel {{
                color: {'white' if isDarkTheme() else 'black'};
                padding-bottom: 3px;
            }}
        """
        )

    def _normalBackgroundColor(self):
        return QColor(255, 255, 255, 0 if isDarkTheme() else 255)

    def _hoverBackgroundColor(self):
        return QColor(255, 255, 255, 10 if isDarkTheme() else 245)

    def _pressedBackgroundColor(self):
        return QColor(255, 255, 255, 7 if isDarkTheme() else 235)

    def paintEvent(self, e):
        super().paintEvent(e)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Plot")
        action = menu.exec(event.globalPos())
        if action == delete_action:
            self.deleteRequested.emit(self.plot_id)
