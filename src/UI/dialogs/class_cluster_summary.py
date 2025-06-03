from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ClassClusterSummary(QWidget):
    """Widget for displaying a summary of a class or cluster."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window)

        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # Title label
        self.title_label = QLabel(title, self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        # Number of samples label
        self.samples_label = QLabel("Number of samples: 0", self)
        self.samples_label.setAlignment(Qt.AlignCenter)
        self.samples_label.setStyleSheet("font-size: 12pt; color: gray;")
        self.layout.addWidget(self.samples_label)

        # Table widget setup
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Parameter", "Average", "Std Dev"])
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.setSelectionMode(QTableWidget.NoSelection)
        self.table_widget.setStyleSheet(
            """
            QTableWidget {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                padding: 8px;
                font-weight: bold;
                border: 1px solid #CCCCCC;
            }
        """
        )
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table_widget)

        # Optional: Add a footer or additional information if needed
        # self.footer_label = QLabel("Additional Info", self)
        # self.footer_label.setAlignment(Qt.AlignCenter)
        # self.layout.addWidget(self.footer_label)

        # Set minimum size for better appearance
        self.setMinimumSize(400, 300)

    def set_summary_data(self, name, num_samples, parameter_data):
        """Sets the summary data for the window.

        Args:
            name (str): Name of the class/cluster.
            num_samples (int): Number of images in the class/cluster.
            parameter_data (dict): Dictionary containing parameter names, averages, and standard deviations.
                                   Example: {"area": (1234.5, 100.2), "circularity": (0.85, 0.05), ...}
        """
        self.title_label.setText(f"{name} Summary")
        self.samples_label.setText(f"Number of samples: {num_samples}")

        self.table_widget.setRowCount(len(parameter_data))
        for row, (param, (avg, std)) in enumerate(parameter_data.items()):
            param_item = QTableWidgetItem(param.capitalize())
            avg_item = QTableWidgetItem(f"{avg:.2f}")
            std_item = QTableWidgetItem(f"{std:.2f}")

            # Center align the text
            param_item.setTextAlignment(Qt.AlignCenter)
            avg_item.setTextAlignment(Qt.AlignCenter)
            std_item.setTextAlignment(Qt.AlignCenter)

            self.table_widget.setItem(row, 0, param_item)
            self.table_widget.setItem(row, 1, avg_item)
            self.table_widget.setItem(row, 2, std_item)
