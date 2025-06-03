# scatter_chart_container.py

import sys

from delegates import ImageDelegate
from models import StringListModel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListView,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .chart_frame import ChartFrame  # Import ChartFrame
from .scatter_chart import ScatterChart


class ScatterChartContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize ScatterChart and wrap it in ChartFrame
        self.actual_scatter_chart = ScatterChart()
        self.actual_scatter_chart.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self.scatter_chart_frame = ChartFrame(self.actual_scatter_chart)
        self.scatter_chart_frame.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        # Initialize Inspector (QListView)
        self.list_view = QListView()
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setFixedWidth(200)  # Adjust as needed

        # Initialize Model and Delegate for QListView
        self.list_model = StringListModel()
        self.list_view.setModel(self.list_model)
        self.image_delegate = ImageDelegate()
        self.list_view.setItemDelegate(self.image_delegate)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Connect signals from the actual chart instance
        self.actual_scatter_chart.selectionChanged.connect(self.update_inspector)

        # Setup Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)  # Margins for the container itself
        layout.setSpacing(10)
        layout.addWidget(self.scatter_chart_frame, 3)  # Add framed chart
        layout.addWidget(self.list_view, 1)  # Smaller proportion for inspector
        self.setLayout(layout)

        # Give ScatterChartContainer a slight off-white background to see frames better
        self.setStyleSheet("background-color: #f8f8f8;")

    def set_data(self, data):
        self.actual_scatter_chart.set_data(data)  # Use actual_scatter_chart

        # Populate the inspector model with all IDs and associated images
        ids = [
            point[4] for point in self.actual_scatter_chart.data if len(point) == 5
        ]  # Use actual_scatter_chart
        self.list_model.setStringList(ids)
        self.list_model.image_data.clear()

        # Associate images with IDs
        for id_ in ids:
            pixmap = QPixmap(150, 150)
            pixmap.fill(QColor("lightgray"))  # Placeholder: Replace with actual images
            self.list_model.setImageData(id_, pixmap)

    def update_inspector(self, selected_ids):
        """Update the inspector view based on selected IDs."""
        if selected_ids:
            # Show only selected items
            self.list_model.setStringList(selected_ids)
        else:
            # Show all items
            all_ids = [
                point[4] for point in self.actual_scatter_chart.data if len(point) == 5
            ]  # Use actual_scatter_chart
            self.list_model.setStringList(all_ids)


# Example usage
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Sample data: (x, y, size_variable, color, id)
    sample_data = [
        (10, 20, 5, QColor(Qt.green), "ID1"),
        (30, 40, 15, QColor(Qt.blue), "ID2"),
        (50, 60, 10, QColor(Qt.red), "ID3"),
        (70, 80, 20, QColor(Qt.yellow), "ID4"),
    ]

    container = ScatterChartContainer()
    container.set_data(sample_data)
    container.resize(900, 700)  # Increased size for better visibility of frame/shadow
    container.setWindowTitle("Scatter Chart with Inspector in Frame")
    container.show()

    sys.exit(app.exec())
