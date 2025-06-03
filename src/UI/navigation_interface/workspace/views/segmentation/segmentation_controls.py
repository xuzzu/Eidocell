# frontend/views/segmentation_controls.py

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)
from qfluentwidgets import ComboBox, PushButton, Slider

from UI.common.style_sheet import EidocellStyleSheet


class SegmentationControls(QFrame):
    """Segmentation controls with Fluent Widgets styling and enhanced design."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set object name for styling
        self.setObjectName("segmentationControlsFrame")

        # Apply Stylesheet
        EidocellStyleSheet.SEGMENTATION_CONTROLS.apply(self)

        # Apply a subtle shadow effect to the frame
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)  # Controls the softness of the shadow
        shadow.setXOffset(0)  # Horizontal offset
        shadow.setYOffset(0)  # Vertical offset
        shadow.setColor(QColor(0, 0, 0, 30))  # Shadow color with transparency
        self.setGraphicsEffect(shadow)

        # Set size policy to be fixed
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedWidth(300)
        self.setFixedHeight(335)

        # Initialize widgets
        self.method_label = QLabel("Base segmentation method")
        self.method_label.setObjectName("methodLabel")
        self.parameter_label = QLabel("Distance from center (%)")
        self.parameter_label.setObjectName("parameterLabel")

        self.method_selector = ComboBox()
        self.method_selector.addItems(
            ["Otsu's Thresholding", "Watershed", "Adaptive Thresholding"]
        )
        self.method_selector.setCurrentIndex(0)
        self.method_selector.setFixedWidth(250)

        self.parameter_slider = Slider(Qt.Horizontal)
        self.parameter_slider.setFixedWidth(250)
        self.parameter_slider_2 = Slider(Qt.Horizontal)  # Second slider
        self.parameter_slider_2.setFixedWidth(250)
        self.parameter_slider_2.setVisible(True)  # Initially visible
        self.parameter_label_2 = QLabel("")  # Second label
        self.parameter_label_2.setObjectName("parameterLabel2")
        self.parameter_label_2.setVisible(True)

        # Use PushButton for buttons
        button_layout = QVBoxLayout()
        self.apply_button = PushButton("Apply to all")
        self.resample_button = PushButton("Resample")
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.resample_button)
        # push to the bottom
        button_layout.addStretch()

        # Main layout for controls
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Add widgets to the layout
        layout.addWidget(self.method_label)
        layout.addWidget(self.method_selector)
        layout.addWidget(self.parameter_label)
        layout.addWidget(self.parameter_slider)
        layout.addWidget(self.parameter_label_2)
        layout.addWidget(self.parameter_slider_2)
        layout.addLayout(button_layout)
        layout.addStretch()

        # Connect signals
        self.method_selector.currentIndexChanged.connect(self.update_parameter_settings)

        # Initialize parameter settings based on default selection
        self.update_parameter_settings()

    def update_parameter_settings(self):
        """Update slider settings based on selected method"""
        method = self.method_selector.currentText()
        self.parameter_slider_2.setVisible(False)
        self.parameter_label_2.setVisible(False)

        if method == "Adaptive Thresholding":
            self.parameter_label.setText("Block size")
            self.parameter_slider.setRange(3, 101)
            self.parameter_slider.setSingleStep(2)
            self.parameter_slider.setValue(35)  # Default Block Size
            self.parameter_slider.setTickInterval(2)
            self.parameter_slider.setEnabled(True)

            self.parameter_label_2.setText("C:")
            self.parameter_label_2.setVisible(True)
            self.parameter_slider_2.setRange(-100, 100)
            self.parameter_slider_2.setValue(2)
            self.parameter_slider_2.setSingleStep(1)
            self.parameter_slider_2.setTickInterval(1)
            self.parameter_slider_2.setVisible(True)
            self.parameter_slider_2.setEnabled(True)

        elif method == "Otsu's Thresholding":
            self.parameter_label.setText("Distance from center (%)")
            self.parameter_slider.setRange(0, 100)
            self.parameter_slider.setValue(30)
            self.parameter_slider.setTickInterval(1)
            self.parameter_slider.setSingleStep(1)
            self.parameter_slider.setEnabled(True)

            self.parameter_label_2.setText("Minimum object size")
            self.parameter_label_2.setVisible(True)
            self.parameter_slider_2.setRange(10, 100)
            self.parameter_slider_2.setValue(15)
            self.parameter_slider_2.setSingleStep(1)
            self.parameter_slider_2.setTickInterval(1)
            self.parameter_slider_2.setVisible(True)
            self.parameter_slider_2.setEnabled(True)

        elif method == "Watershed":
            self.parameter_label.setText("Foreground threshold (%)")
            self.parameter_slider.setRange(0, 100)
            self.parameter_slider.setValue(70)
            self.parameter_slider.setTickInterval(1)
            self.parameter_slider.setSingleStep(1)
            self.parameter_slider.setEnabled(True)

            self.parameter_label_2.setText("Morph kernel size")
            self.parameter_label_2.setVisible(True)
            self.parameter_slider_2.setRange(1, 11)  # Odd numbers only (for kernels)
            self.parameter_slider_2.setValue(3)
            self.parameter_slider_2.setTickInterval(2)
            self.parameter_slider_2.setSingleStep(2)  # Odd numbers only
            self.parameter_slider_2.setVisible(True)
            self.parameter_slider_2.setEnabled(True)
