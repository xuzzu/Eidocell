from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotViewerWidget(QWidget):
    """Widget for viewing Plotly plots in a web engine."""

    def __init__(self, html_file_path, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)  # Set window flags to make it a separate window
        self.setWindowTitle("Plot Viewer")
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #FFF; color: #000;")
        layout = QVBoxLayout(self)
        self.web_view = QWebEngineView(self)
        layout.addWidget(self.web_view)
        self.load_plot(html_file_path)

    def load_plot(self, html_file_path):
        """Loads the specified HTML file into the web view."""
        url = QUrl.fromLocalFile(html_file_path)
        self.web_view.load(url)
        print(url)
