import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QLabel,
    QSizePolicy,
)


class PreviewGrid(QFrame):
    """A widget displaying the preview grid of segmented images with enhanced design."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set object name for styling
        self.setObjectName("previewGridFrame")

        # Style the frame with rounded corners and background color
        self.setStyleSheet(
            """
            QFrame#previewGridFrame {
                background-color: #ffffff;
                border-radius: 10px;
            }
        """
        )

        # Apply a subtle shadow effect to the frame
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)  # Controls the softness of the shadow
        shadow.setXOffset(0)  # Horizontal offset
        shadow.setYOffset(0)  # Vertical offset
        shadow.setColor(QColor(0, 0, 0, 30))  # Shadow color with transparency
        self.setGraphicsEffect(shadow)

        # Set size policy to forbit expansion
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Set up the grid layout
        self.layout = QGridLayout(self)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.preview_widgets = []

        # Create a 2x4 grid of preview images
        self.setup_preview_grid()

    def setup_preview_grid(self):
        """Initialize empty preview grid"""
        rows = 2
        cols = 4
        for i in range(rows):
            for j in range(cols):
                preview = QLabel()
                preview.setFixedSize(140, 140)
                preview.setAlignment(Qt.AlignCenter)
                preview.setStyleSheet(
                    """
                    QLabel { 
                        background-color: #f0f0f0; 
                        border: 1px solid #cccccc;
                        border-radius: 4px;
                    }
                """
                )
                self.layout.addWidget(preview, i, j)
                self.preview_widgets.append(preview)

    def update_previews(self, image_paths):
        """Update preview images with segmented results"""
        for widget, path in zip(self.preview_widgets, image_paths):
            if path:
                # Load masked image from path
                masked_image = cv2.imread(path)
                if masked_image is not None:
                    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                    pixmap = self.convert_to_pixmap(masked_image)
                    widget.setPixmap(
                        pixmap.scaled(
                            widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
                else:
                    widget.setText("Image not found")
            else:
                widget.setText("No Image")

    @staticmethod
    def convert_to_pixmap(cv_image):
        """Convert OpenCV image to QPixmap"""
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        return QPixmap.fromImage(q_image)
