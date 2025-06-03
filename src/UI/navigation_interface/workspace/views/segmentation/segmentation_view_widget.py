# UI/navigation_interface/workspace/views/segmentation/segmentation_view_widget.py

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon, SegmentedWidget

from UI.navigation_interface.workspace.views.segmentation.advanced_segmentation_widget import (
    AdvancedSegmentationWidget,
)
from UI.navigation_interface.workspace.views.segmentation.preview_grid import (
    PreviewGrid,
)
from UI.navigation_interface.workspace.views.segmentation.segmentation_controls import (
    SegmentationControls,
)


class BaseSegmentationWidget(QWidget):
    """
    Container widget for the original 'base' segmentation controls and preview.
    """

    def __init__(self, main_window_reference, parent=None):
        super().__init__(parent)
        self.setObjectName("baseSegmentationWidget")

        self.controls = SegmentationControls(self)
        self.preview = PreviewGrid(self)

        content_layout = QHBoxLayout(self)
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(20)

        # Wrappers for vertical alignment (optional, but can help)
        left_wrapper = QVBoxLayout()
        left_wrapper.addWidget(self.controls)
        left_wrapper.addStretch(1)

        right_wrapper = QVBoxLayout()
        right_wrapper.addWidget(self.preview)
        right_wrapper.addStretch(1)

        content_layout.addLayout(left_wrapper)
        content_layout.addLayout(right_wrapper, 1)

        self.setLayout(content_layout)


class SegmentationViewWidget(QWidget):
    """Main widget for the segmentation interface, now with sub-views."""

    def __init__(self, main_window_reference, parent=None):
        super().__init__(parent)
        self.setObjectName("segmentationView")
        self.main_window_reference = main_window_reference
        self.segmentation_presenter = None  # Will be set by presenter

        self.main_v_layout = QVBoxLayout(self)
        self.main_v_layout.setContentsMargins(20, 20, 20, 20)
        self.main_v_layout.setSpacing(15)
        self.main_v_layout.setAlignment(Qt.AlignTop)

        self.view_switcher = SegmentedWidget(self)
        self.view_switcher.setFixedHeight(32)

        self.stacked_widget = QStackedWidget(self)

        self.base_segmentation_pane = BaseSegmentationWidget(
            main_window_reference, self
        )
        self.advanced_segmentation_pane = AdvancedSegmentationWidget(self)

        self._add_sub_interface(
            self.base_segmentation_pane, "baseSegmentation", "Base Segmentation"
        )
        self._add_sub_interface(
            self.advanced_segmentation_pane,
            "advancedSegmentation",
            "Advanced Segmentation",
        )

        self.main_v_layout.addWidget(self.view_switcher, 0, Qt.AlignHCenter)
        self.main_v_layout.addWidget(self.stacked_widget, 1)

        self.stacked_widget.setCurrentWidget(self.base_segmentation_pane)
        self.view_switcher.setCurrentItem(self.base_segmentation_pane.objectName())
        self.view_switcher.setEnabled(False)
        self.view_switcher.setToolTip(
            "Advanced Segmentation is currently under development."
        )

        self.view_switcher.currentItemChanged.connect(
            lambda route_key: self.stacked_widget.setCurrentWidget(
                self.findChild(QWidget, route_key)
            )
        )

    def _add_sub_interface(
        self, widget: QWidget, object_name: str, text: str, icon=None
    ):
        """Helper to add sub-interface to both switcher and stacked widget."""
        widget.setObjectName(object_name)
        self.stacked_widget.addWidget(widget)

        if icon:
            self.view_switcher.addItem(routeKey=object_name, text=text, icon=icon)
        else:
            self.view_switcher.addItem(routeKey=object_name, text=text)

    def set_presenter(self, presenter):
        """Set the presenter and initialize the view"""
        self.segmentation_presenter = presenter
        if self.segmentation_presenter and hasattr(
            self.segmentation_presenter, "connect_base_controls"
        ):
            self.segmentation_presenter.connect_base_controls(
                self.base_segmentation_pane.controls
            )

    @property
    def controls(self) -> SegmentationControls:
        return self.base_segmentation_pane.controls

    @property
    def preview(self) -> PreviewGrid:
        return self.base_segmentation_pane.preview

    def update_previews(self):
        """Update preview images with current settings (delegates to Base or Advanced)"""
        current_widget = self.stacked_widget.currentWidget()
        if current_widget == self.base_segmentation_pane:
            if self.segmentation_presenter:
                # This logic might need to be in the presenter, checking the active view
                # For now, assume presenter's update_selected_samples works with base controls
                self.segmentation_presenter.update_selected_samples(
                    self.segmentation_presenter.selected_samples
                )
        # elif current_widget == self.advanced_segmentation_pane:
        # Handle advanced preview update if/when implemented
        # pass
        else:
            logging.warning(
                "update_previews called on an unknown segmentation sub-view."
            )

    def apply_segmentation(self):
        """Apply segmentation to all images (delegates to Base or Advanced)"""
        current_widget = self.stacked_widget.currentWidget()
        if current_widget == self.base_segmentation_pane:
            if self.segmentation_presenter:
                self.segmentation_presenter.segment_all()
        else:
            logging.warning(
                "apply_segmentation called on an unknown segmentation sub-view."
            )

    def resample_preview_images(self):
        """Resample preview images with new samples (delegates to Base or Advanced)"""
        current_widget = self.stacked_widget.currentWidget()
        if current_widget == self.base_segmentation_pane:
            if self.segmentation_presenter:
                self.segmentation_presenter.resample_samples()
        else:
            logging.warning(
                "resample_preview_images called on an unknown segmentation sub-view."
            )
