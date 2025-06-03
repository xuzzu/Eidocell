# UI/dialogs/gallery_filter_dialog.py
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import MessageBoxBase, PushButton, ScrollArea, SubtitleLabel

from UI.dialogs.filter_condition_widget import FilterConditionWidget


class GalleryFilterDialog(MessageBoxBase):
    """Dialog for defining gallery filter conditions."""

    filters_applied = Signal(list)  # Emits a list of condition dicts

    def __init__(self, available_features: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Gallery")
        self.available_features = available_features
        self.condition_widgets = []  # To keep track of added condition widgets

        # Main layout is self.viewLayout from MessageBoxBase

        # Title
        self.titleLabel = SubtitleLabel("Define Filter Conditions", self)
        self.viewLayout.addWidget(self.titleLabel)

        # "Add Condition" Button
        self.add_condition_button = PushButton(FIF.ADD, "Add Condition", self)
        self.add_condition_button.clicked.connect(self._add_condition_row)
        self.viewLayout.addWidget(self.add_condition_button, alignment=Qt.AlignLeft)

        # ScrollArea for conditions
        self.conditions_scroll_area = ScrollArea(self)  # Use qfluentwidgets ScrollArea
        self.conditions_scroll_area.setWidgetResizable(True)
        self.conditions_scroll_area.setFixedHeight(200)  # Adjust as needed
        self.conditions_scroll_area.setFrameShape(QFrame.NoFrame)  # Make it blend

        self.conditions_container_widget = (
            QWidget()
        )  # Widget to hold the layout for conditions
        self.conditions_layout = QVBoxLayout(self.conditions_container_widget)
        self.conditions_layout.setAlignment(Qt.AlignTop)  # Conditions added from top
        self.conditions_layout.setContentsMargins(0, 5, 0, 5)
        self.conditions_layout.setSpacing(5)

        self.conditions_scroll_area.setWidget(self.conditions_container_widget)
        self.viewLayout.addWidget(self.conditions_scroll_area)

        # Buttons (yesButton and cancelButton are from MessageBoxBase)
        self.yesButton.setText("Apply Filters")
        self.cancelButton.setText("Cancel")
        self.yesButton.clicked.connect(self._apply_filters)

        # Adjust dialog size
        self.widget.setMinimumWidth(600)
        self.widget.setMinimumHeight(400)

        # Add an initial condition row
        self._add_condition_row()

    def _add_condition_row(self):
        """Adds a new FilterConditionWidget to the layout."""
        condition_widget = FilterConditionWidget(self.available_features, self)
        condition_widget.remove_requested.connect(
            lambda w=condition_widget: self._remove_condition_row(w)
        )
        self.conditions_layout.addWidget(condition_widget)
        self.condition_widgets.append(condition_widget)
        # Ensure scroll area resizes if content exceeds its height
        self.conditions_container_widget.adjustSize()

    def _remove_condition_row(self, condition_widget: FilterConditionWidget):
        """Removes the specified FilterConditionWidget."""
        if condition_widget in self.condition_widgets:
            self.condition_widgets.remove(condition_widget)
            self.conditions_layout.removeWidget(condition_widget)
            condition_widget.deleteLater()
            self.conditions_container_widget.adjustSize()

    def _apply_filters(self):
        """Gathers all conditions and emits the filters_applied signal."""
        current_filters = []
        for widget in self.condition_widgets:
            condition = widget.get_condition()
            # Basic validation for "Between"
            if (
                condition["operator"] == "Between"
                and condition.get("value2", condition["value1"]) < condition["value1"]
            ):
                # Show some error or swap values
                logging.warning(
                    f"Filter condition for '{condition['feature']}' has 'Between' with value2 < value1. Swapping."
                )
                condition["value1"], condition["value2"] = (
                    condition["value2"],
                    condition["value1"],
                )
            current_filters.append(condition)

        self.filters_applied.emit(current_filters)
        self.accept()  # Close dialog

    def get_filters(self) -> list:
        """Returns the list of filter conditions."""
        # This might be called if the dialog is accepted without explicit apply button click (if yesButton is used)
        filters = []
        for widget in self.condition_widgets:
            filters.append(widget.get_condition())
        return filters
