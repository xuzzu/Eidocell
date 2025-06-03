from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import ComboBox
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import PushButton, Slider, TogglePushButton, ToggleToolButton


class GalleryControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gallery_presenter = None

        # Main Frame: This will contain all controls and be centered.
        self.main_frame = QFrame(self)
        self.main_frame.setObjectName("galleryControlsFrame")
        self.main_frame.setStyleSheet(
            """
            QFrame#galleryControlsFrame {
                background-color: #ffffff; /* or themeColor() */
                border-radius: 10px;
                /* Optionally set a max-width if you don't want it to stretch too much */
                /* max-width: 900px; */
            }
        """
        )
        # Apply a subtle shadow effect to the frame
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 25))  # Slightly more visible shadow
        self.main_frame.setGraphicsEffect(shadow)

        # Layout for the content INSIDE the main_frame
        # This QHBoxLayout will hold the vertical sections for Scale, Actions, Mask View
        frame_content_layout = QHBoxLayout(self.main_frame)
        frame_content_layout.setContentsMargins(
            20, 15, 20, 15
        )  # Padding inside the frame
        frame_content_layout.setSpacing(
            30
        )  # Spacing BETWEEN the main vertical sections

        #  Section 1: Scale
        scale_section_layout = QVBoxLayout()
        scale_section_layout.setSpacing(8)  # Spacing within this section
        self.scaleLabel = QLabel("Scale", self.main_frame)
        self.scaleLabel.setStyleSheet("font-size: 13pt; color: #333;")  # Adjusted style
        self.scaleLabel.setAlignment(Qt.AlignCenter)
        self.scale_slider = Slider(Qt.Horizontal, self.main_frame)
        self.scale_slider.setRange(50, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.setFixedWidth(220)  # Adjusted width
        scale_section_layout.addWidget(self.scaleLabel)
        scale_section_layout.addWidget(self.scale_slider, 0, Qt.AlignCenter)
        scale_section_layout.addStretch(1)  # Pushes content to top if section is taller

        #  Section 2: Actions (Sort & Filter)
        actions_section_layout = QVBoxLayout()
        actions_section_layout.setSpacing(8)
        self.actionsLabel = QLabel("Actions", self.main_frame)
        self.actionsLabel.setStyleSheet("font-size: 13pt; color: #333;")
        self.actionsLabel.setAlignment(Qt.AlignCenter)

        sort_filter_controls_layout = (
            QHBoxLayout()
        )  # Horizontal layout for sort and filter
        sort_filter_controls_layout.setSpacing(10)

        # Sorting controls
        sort_group_layout = QHBoxLayout()
        sort_group_layout.setSpacing(5)
        self.sortAscButton = ToggleToolButton(FIF.UP, self.main_frame)
        self.sortAscButton.setToolTip("Sort Ascending")
        self.sortDescButton = ToggleToolButton(FIF.DOWN, self.main_frame)
        self.sortDescButton.setToolTip("Sort Descending")
        self.parameterComboBox = ComboBox(self.main_frame)
        self.parameterComboBox.setFixedWidth(150)
        self.parameterComboBox.addItems(
            [
                "Area",
                "Perimeter",
                "Eccentricity",
                "Solidity",
                "Aspect Ratio",
                "Circularity",
                "Major Axis Length",
                "Minor Axis Length",
                "Mean Intensity",
                "Compactness",
                "Convexity",
                "Curl",
                "Volume",
            ]
        )
        # sort_group_layout.addWidget(sort_label)
        sort_group_layout.addWidget(self.sortAscButton)
        sort_group_layout.addWidget(self.sortDescButton)
        sort_group_layout.addWidget(self.parameterComboBox)
        sort_filter_controls_layout.addLayout(sort_group_layout)

        # Scale button
        self.inspectButton = PushButton(FIF.ZOOM_IN, "Inspect", self.main_frame)
        self.inspectButton.setToolTip("Open scale adjustment")
        sort_filter_controls_layout.addWidget(self.inspectButton, 0, Qt.AlignVCenter)

        actions_section_layout.addWidget(self.actionsLabel)
        actions_section_layout.addLayout(sort_filter_controls_layout)
        actions_section_layout.addStretch(1)

        #  Section 3: View Options (Mask Toggle)
        view_options_section_layout = QVBoxLayout()
        view_options_section_layout.setSpacing(8)
        # Optional: Add a label for this section if desired
        self.viewOptionsLabel = QLabel("View", self.main_frame)
        self.viewOptionsLabel.setStyleSheet("font-size: 13pt; color: #333;")
        self.viewOptionsLabel.setAlignment(Qt.AlignCenter)
        self.mask_toggle = TogglePushButton(FIF.VIEW, "Mask view", self.main_frame)
        self.mask_toggle.setFixedWidth(130)  # Adjusted width

        view_options_section_layout.addWidget(self.viewOptionsLabel)
        view_options_section_layout.addWidget(self.mask_toggle, 0, Qt.AlignCenter)
        view_options_section_layout.addStretch(1)

        # Add sections to the frame_content_layout
        frame_content_layout.addLayout(scale_section_layout)
        frame_content_layout.addLayout(actions_section_layout)
        frame_content_layout.addLayout(view_options_section_layout)

        # Outer layout for GalleryControls widget, responsible for centering main_frame
        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)  # Margins around the frame
        outer_layout.addStretch(1)  # Push frame to center
        outer_layout.addWidget(self.main_frame)
        outer_layout.addStretch(1)  # Push frame to center
        self.setLayout(outer_layout)

        # self.main_frame.setFixedHeight(120)  # Set fixed height for the main frame
