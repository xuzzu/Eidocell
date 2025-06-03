# flow_widget.py
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import FlowLayout

from UI.common.style_sheet import EidocellStyleSheet


class FlowGallery(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set object name for styling
        self.setObjectName("FlowGalleryFrame")

        # Apply Stylesheet
        EidocellStyleSheet.FLOW_GALLERY.apply(self)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)

        # Set up the scrolling area inside the frame
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("FlowGalleryScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Create a container widget for the layout
        self.container = QWidget()
        self.container.setObjectName("FlowGalleryContainer")
        self.scroll_area.setWidget(self.container)

        # Use FlowLayout from Fluent Widgets
        self.flow_layout = FlowLayout(self.container, needAni=False, isTight=False)
        self.flow_layout.setSpacing(10)  # Adjust spacing as needed
        self.flow_layout.setAlignment(
            Qt.AlignHCenter | Qt.AlignTop
        )  # Center tiles horizontally

        # Layout for the frame
        self.frame_layout = QVBoxLayout(self)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)  # Inner margins
        self.frame_layout.setSpacing(0)
        self.frame_layout.addWidget(self.scroll_area)

        # Set the layout to the frame
        self.setLayout(self.frame_layout)

        # Enable clipping to ensure rounded corners
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def add_item(self, widget):
        """Convenience method to add a widget to the flow layout."""
        self.flow_layout.addWidget(widget)

    def remove_item(self, widget):
        """Convenience method to remove a widget from the flow layout."""
        self.flow_layout.removeWidget(widget)
