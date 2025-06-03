# UI/dialogs/filter_condition_widget.py

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import ComboBox, DoubleSpinBox
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import LineEdit, TransparentToolButton


class FilterConditionWidget(QWidget):
    """Widget for defining a single filter condition."""

    remove_requested = Signal()  # Signal to tell parent to remove this widget

    def __init__(self, available_features: list, parent=None):
        super().__init__(parent)
        self.available_features = available_features

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)

        # Feature ComboBox
        self.feature_combo = ComboBox(self)
        self.feature_combo.addItems(self.available_features)
        self.feature_combo.setMinimumWidth(150)
        main_layout.addWidget(self.feature_combo)

        # Operator ComboBox
        self.operator_combo = ComboBox(self)
        self.operator_combo.addItems([">", ">=", "<", "<=", "==", "!=", "Between"])
        self.operator_combo.setMinimumWidth(80)
        self.operator_combo.currentTextChanged.connect(self._update_value_widgets)
        main_layout.addWidget(self.operator_combo)

        # Value Input(s) - use a QStackedWidget or hide/show
        self.value_widget_container = QWidget(self)
        self.value_layout = QHBoxLayout(self.value_widget_container)
        self.value_layout.setContentsMargins(0, 0, 0, 0)
        self.value_layout.setSpacing(5)

        self.value1_spinbox = DoubleSpinBox(self)
        self.value1_spinbox.setRange(-1e9, 1e9)  # Wide range
        self.value1_spinbox.setDecimals(3)
        self.value_layout.addWidget(self.value1_spinbox)

        self.value_separator_label = QLabel("and", self)  # For "Between"
        self.value_layout.addWidget(self.value_separator_label)

        self.value2_spinbox = DoubleSpinBox(self)
        self.value2_spinbox.setRange(-1e9, 1e9)
        self.value2_spinbox.setDecimals(3)
        self.value_layout.addWidget(self.value2_spinbox)

        main_layout.addWidget(self.value_widget_container)
        main_layout.addSpacerItem(
            QSpacerItem(10, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        # Remove Button
        self.remove_button = TransparentToolButton(FIF.CLOSE, self)
        self.remove_button.setToolTip("Remove this condition")
        self.remove_button.clicked.connect(self.remove_requested.emit)
        main_layout.addWidget(self.remove_button)

        self._update_value_widgets(self.operator_combo.currentText())  # Initial setup

    def _update_value_widgets(self, operator_text: str):
        """Shows/hides the second value spinbox based on the operator."""
        if operator_text == "Between":
            self.value_separator_label.show()
            self.value2_spinbox.show()
        else:
            self.value_separator_label.hide()
            self.value2_spinbox.hide()

    def get_condition(self) -> dict:
        """Returns the condition defined by this widget as a dictionary."""
        feature = self.feature_combo.currentText()
        operator = self.operator_combo.currentText()
        value1 = self.value1_spinbox.value()
        condition = {"feature": feature, "operator": operator, "value1": value1}
        if operator == "Between":
            condition["value2"] = self.value2_spinbox.value()
        return condition
