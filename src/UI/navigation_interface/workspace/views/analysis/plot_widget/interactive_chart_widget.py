### interactive_chart_widget.py
import logging
import uuid
from typing import Dict, List, Optional

from PySide6.QtCore import QModelIndex, QSize, Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QSizePolicy  # Removed QLabel as BodyLabel is used
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Fluent Widget imports
from qfluentwidgets import BodyLabel, CardWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import PushButton, SegmentedToggleToolWidget, TableWidget

from .chart_frame import ChartFrame
from .enums import InteractionMode
from .gate import BaseGate


class InteractiveChartWidget(QWidget):
    _ASPECT_W, _ASPECT_H = 3, 5

    new_gate_defined = Signal(str, object)  # plot_id, gate_obj

    # Emitted if user edits gate properties (e.g., name, color) in the table (for future use).
    gate_update_requested = Signal(object)  # gate_obj (updated)

    # Emitted when user clicks the "Delete Gate" button for a gate in this chart's table.
    gate_delete_requested = Signal(str)  # gate_id

    def __init__(self, chart_widget, parent=None):
        super().__init__(parent)
        self.chart = chart_widget
        self.chart_frame = ChartFrame(self.chart)
        self.plot_id: Optional[str] = None  # Will be set by AnalysisPresenter

        self._row_to_gate_id_map = {}
        self._is_syncing_selection = False

        # Define mode mapping for SegmentedToggleToolWidget
        self.mode_map = {
            # "select_points": (
            #     InteractionMode.SELECT_POINTS,
            #     FIF.FINGERPRINT,
            #     self.tr("Mode: Select individual data points (Not implemented)"),
            # ),
            "rect_gate": (
                InteractionMode.RECT_GATE,
                FIF.LAYOUT,
                self.tr("Mode: Draw a rectangular gate"),
            ),
            "polygon_gate": (
                InteractionMode.POLYGON_GATE,
                FIF.BRUSH,
                self.tr(
                    "Mode: Draw a polygon gate (Click to add points, Double-click or Enter to finish, Esc to cancel)"
                ),
            ),
        }

        self._init_ui_and_pass_controls()
        self._connect_signals()

        self._set_initial_mode()  # Call this after UI is initialized and connections are made

        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sp.setHeightForWidth(True)
        sp.setWidthForHeight(True)
        self.setSizePolicy(sp)

    def _init_ui_and_pass_controls(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.chart_frame, 1)

        self.control_panel = CardWidget(self)
        control_panel_layout = QVBoxLayout(self.control_panel)
        control_panel_layout.setSpacing(10)
        control_panel_layout.setContentsMargins(
            12, 12, 12, 12
        )  # Standard CardWidget margins

        #  Mode Selection
        mode_control_layout = QHBoxLayout()
        mode_label = BodyLabel(self.tr("Mode:"), self.control_panel)
        mode_control_layout.addWidget(mode_label)

        self.mode_toggle_widget = SegmentedToggleToolWidget(self.control_panel)
        self.mode_toggle_widget.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        # self.mode_toggle_widget.addItem(
        #     "select_points", icon=self.mode_map["select_points"][1]
        # )
        self.mode_toggle_widget.addItem("rect_gate", icon=self.mode_map["rect_gate"][1])
        self.mode_toggle_widget.addItem(
            "polygon_gate", icon=self.mode_map["polygon_gate"][1]
        )

        # tt_select, dis_select = self.mode_map["select_points"][2], True

        tt_rect, dis_rect = self.mode_map["rect_gate"][2], False
        if self.chart.chart_type not in ["ScatterChart", "HistogramChart"]:
            dis_rect = True
            tt_rect = self.tr(
                "Rectangular gates are supported on Scatter and Histogram charts."
            )

        tt_poly, dis_poly = self.mode_map["polygon_gate"][2], False
        if self.chart.chart_type != "ScatterChart":
            dis_poly = True
            tt_poly = self.tr("Polygon gates are only supported on Scatter charts.")

        mode_control_layout.addWidget(self.mode_toggle_widget)
        mode_control_layout.addStretch()
        control_panel_layout.addLayout(mode_control_layout)

        #  Action Buttons
        action_buttons_layout = QHBoxLayout()
        self.delete_gate_button = PushButton(
            self.tr("Delete Selected Gate"), self.control_panel, icon=FIF.DELETE
        )
        self.delete_gate_button.setEnabled(False)
        self.delete_gate_button.setToolTip(
            self.tr("Delete the gate currently selected in the table below")
        )
        action_buttons_layout.addWidget(self.delete_gate_button)
        action_buttons_layout.addStretch()
        control_panel_layout.addLayout(action_buttons_layout)

        #  Gate Table
        gate_table_label = BodyLabel(self.tr("Defined Gates:"), self.control_panel)
        control_panel_layout.addWidget(gate_table_label)

        self.gate_table = TableWidget(self.control_panel)
        self.gate_table.setWordWrap(False)
        self.gate_table.setColumnCount(4)
        self.gate_table.setHorizontalHeaderLabels(
            [self.tr("Name"), self.tr("Type"), self.tr("Events"), self.tr("Color")]
        )
        self.gate_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.gate_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.gate_table.verticalHeader().setVisible(False)

        self.gate_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.gate_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.gate_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.gate_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )

        table_min_height = (
            self.gate_table.horizontalHeader().height()
            + self.gate_table.fontMetrics().height() * 3
            + 20
        )
        self.gate_table.setMinimumHeight(table_min_height)
        self.gate_table.setMaximumHeight(
            table_min_height + self.gate_table.fontMetrics().height() * 2
        )

        control_panel_layout.addWidget(self.gate_table)
        self.chart_frame.set_control_widget(self.control_panel)

    def _set_initial_mode(self):
        initial_mode_key = None
        # Determine the first enabled mode key in preferred order
        preferred_order = [
            "rect_gate",
            "polygon_gate",
        ]  # "select_points" is not used in this version
        for key_candidate in preferred_order:
            # Check if the item (button) for this key exists and is enabled
            if self.mode_toggle_widget.widget(key_candidate):
                initial_mode_key = key_candidate
                break

        if initial_mode_key:
            self.mode_toggle_widget.setCurrentItem(initial_mode_key)

            if self.chart and initial_mode_key in self.mode_map:
                interaction_mode_enum = self.mode_map[initial_mode_key][0]
                self.chart.set_interaction_mode(interaction_mode_enum)

    def _connect_signals(self):
        self.mode_toggle_widget.currentItemChanged.connect(
            self.on_mode_segment_changed_by_key
        )

        self.gate_table.itemSelectionChanged.connect(
            self.on_gate_table_selection_changed
        )

        self.delete_gate_button.clicked.connect(self.on_delete_gate_clicked)

        if self.chart:
            self.chart.gateAdded.connect(self._handle_chart_gate_added)
            self.chart.gatesUpdated.connect(self.update_gate_table)
            self.chart.selectedGateChanged.connect(self.on_chart_selected_gate_changed)

    @Slot(object)  # gate_obj from BaseChart
    def _handle_chart_gate_added(self, gate_obj: BaseGate):
        """Emits the ICW's new_gate_defined signal."""
        if self.plot_id:
            self.new_gate_defined.emit(self.plot_id, gate_obj)

    @Slot(str, str)
    def on_mode_segment_changed_by_key(self, newItemKey: str):
        if newItemKey in self.mode_map:
            mode_enum, _, _ = self.mode_map[newItemKey]
            if self.chart:
                self.chart.set_interaction_mode(mode_enum)

    def add_or_update_gate_in_ui(self, gate_obj: BaseGate):
        """
        Adds a new gate to the chart's internal list and updates the UI table,
        or updates an existing gate if one with the same ID is found.
        Called by AnalysisPresenter when a gate is loaded or received from DataManager.
        """
        if not self.chart:
            logging.error(
                f"ICW ({self.plot_id}): Chart not available to add/update gate {gate_obj.id}"
            )
            return

        existing_gate = next((g for g in self.chart.gates if g.id == gate_obj.id), None)

        if existing_gate:
            try:
                idx = self.chart.gates.index(existing_gate)
                self.chart.gates[idx] = gate_obj  # Replace with the new object
            except ValueError:  # Should not happen if existing_gate was found
                self.chart.gates.append(gate_obj)  # Fallback: append if index not found
            logging.debug(
                f"ICW ({self.plot_id}): Updated gate {gate_obj.id} ('{gate_obj.name}') in chart's internal list."
            )
        else:
            self.chart.gates.append(gate_obj)
            logging.debug(
                f"ICW ({self.plot_id}): Added new gate {gate_obj.id} ('{gate_obj.name}') to chart's internal list."
            )

        self.update_gate_table(self.chart.gates)
        self.chart.update()

    @Slot(list)
    def update_gate_table(self, gates_list):
        if self._is_syncing_selection:
            return

        self._is_syncing_selection = True

        current_selected_gate_id_in_table = None
        selected_items = self.gate_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            current_selected_gate_id_in_table = self._row_to_gate_id_map.get(row)

        self.gate_table.setRowCount(0)
        self._row_to_gate_id_map.clear()

        new_row_to_select = -1

        for row_idx, gate in enumerate(gates_list):
            self.gate_table.insertRow(row_idx)
            self._row_to_gate_id_map[row_idx] = gate.id

            self.gate_table.setItem(row_idx, 0, QTableWidgetItem(gate.name))
            self.gate_table.setItem(
                row_idx, 1, QTableWidgetItem(gate.__class__.__name__)
            )
            self.gate_table.setItem(row_idx, 2, QTableWidgetItem(str(gate.event_count)))

            swatch = QPixmap(16, 16)
            swatch.fill(gate.color)
            color_item = QTableWidgetItem()
            color_item.setIcon(swatch)
            color_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            color_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.gate_table.setItem(row_idx, 3, color_item)

            chart_selected_gate = self.chart.get_selected_gate() if self.chart else None
            if (
                current_selected_gate_id_in_table
                and gate.id == current_selected_gate_id_in_table
            ):
                new_row_to_select = row_idx
            elif (
                not current_selected_gate_id_in_table
                and chart_selected_gate
                and chart_selected_gate.id == gate.id
            ):
                new_row_to_select = row_idx

        if new_row_to_select != -1:
            self.gate_table.selectRow(new_row_to_select)
        else:
            self.gate_table.clearSelection()
            if self.chart and self.chart.get_selected_gate() is None:
                self.delete_gate_button.setEnabled(False)

        chart_selected_gate = self.chart.get_selected_gate() if self.chart else None
        if not self.gate_table.selectedItems() and not chart_selected_gate:
            self.delete_gate_button.setEnabled(False)
        elif chart_selected_gate or new_row_to_select != -1:
            self.delete_gate_button.setEnabled(True)

        self._is_syncing_selection = False

    @Slot()
    def on_gate_table_selection_changed(self):
        if self._is_syncing_selection or not self.chart:
            return
        self._is_syncing_selection = True

        selected_rows = self.gate_table.selectionModel().selectedRows()

        if selected_rows:
            selected_row_index = selected_rows[0].row()
            gate_id_from_table = self._row_to_gate_id_map.get(selected_row_index)

            current_chart_gate = self.chart.get_selected_gate()
            if not current_chart_gate or current_chart_gate.id != gate_id_from_table:
                if gate_id_from_table:
                    self.chart.select_gate(gate_id_from_table)
            self.delete_gate_button.setEnabled(True)
        else:
            if self.chart.get_selected_gate() is not None:
                self.chart.select_gate(None)
            self.delete_gate_button.setEnabled(False)

        self._is_syncing_selection = False

    @Slot(object)
    def on_chart_selected_gate_changed(self, selected_gate_obj):
        if self._is_syncing_selection:
            return
        self._is_syncing_selection = True

        self.delete_gate_button.setEnabled(selected_gate_obj is not None)

        if selected_gate_obj:
            found_and_selected_in_table = False
            for r, gid in self._row_to_gate_id_map.items():
                if gid == selected_gate_obj.id:
                    is_row_already_selected = False
                    selected_table_rows = (
                        self.gate_table.selectionModel().selectedRows()
                    )
                    if selected_table_rows and selected_table_rows[0].row() == r:
                        is_row_already_selected = True

                    if not is_row_already_selected:
                        self.gate_table.selectRow(r)
                    found_and_selected_in_table = True
                    break
            if not found_and_selected_in_table:
                self.gate_table.clearSelection()
        else:
            self.gate_table.clearSelection()

        self._is_syncing_selection = False

    @Slot()
    def on_delete_gate_clicked(self):
        if self._is_syncing_selection or not self.chart:
            return

        selected_gate = self.chart.get_selected_gate()
        if selected_gate:
            self.chart.remove_gate(selected_gate.id)
            self.gate_delete_requested.emit(selected_gate.id)

    @Slot()
    def on_delete_gate_clicked(self):
        if self._is_syncing_selection or not self.chart:
            return

        selected_gate = (
            self.chart.get_selected_gate()
        )  # This is a BaseGate object or None

        if (
            selected_gate
            and hasattr(selected_gate, "id")
            and selected_gate.id is not None
        ):
            gate_id_str = str(selected_gate.id)  # Convert UUID to string for the signal
            logging.debug(
                f"ICW ({self.plot_id}): Delete button clicked for gate ID {gate_id_str} ('{selected_gate.name}'). Emitting request."
            )
            self.gate_delete_requested.emit(gate_id_str)
        else:
            logging.warning(
                f"ICW ({self.plot_id}): Delete gate button clicked, but no gate is selected in the chart or selected gate has no ID."
            )

    @Slot(str)  # gate_id_str
    def remove_gate_from_ui(self, gate_id_to_remove_str: str):
        """
        Removes a gate from the chart's internal list and updates the UI table and chart drawing.
        Called by AnalysisPresenter after DataManager confirms the gate deletion.
        """
        if not self.chart:
            logging.error(
                f"ICW ({self.plot_id}): Chart not available to remove gate {gate_id_to_remove_str}"
            )
            return

        try:
            gate_id_uuid = uuid.UUID(
                gate_id_to_remove_str
            )  # Convert string ID to UUID for comparison
        except ValueError:
            logging.error(
                f"ICW ({self.plot_id}): Invalid UUID string received for gate deletion: {gate_id_to_remove_str}"
            )
            return

        gate_to_remove = next(
            (g for g in self.chart.gates if g.id == gate_id_uuid), None
        )

        if gate_to_remove:
            self.chart.gates.remove(gate_to_remove)
            logging.debug(
                f"ICW ({self.plot_id}): Removed gate {gate_id_to_remove_str} ('{gate_to_remove.name}') from chart's internal list."
            )

            # If the deleted gate was the one selected in the chart, deselect it
            if self.chart.selected_gate_id == gate_id_uuid:
                self.chart.select_gate(
                    None
                )  # This will also update delete_gate_button state via signal

            # Refresh the UI table to reflect the removal
            self.update_gate_table(self.chart.gates)  # This also handles selection sync

            # Trigger a repaint of the chart to remove the gate's visual representation
            self.chart.update()
        else:
            logging.warning(
                f"ICW ({self.plot_id}): Gate ID {gate_id_to_remove_str} not found in chart's internal list for UI removal. It might have already been removed or never added to this specific chart instance."
            )

    def set_data(self, *args, **kwargs):
        if not self.chart:
            return
        self.chart.set_data(*args, **kwargs)
        self.update_gate_table(self.chart.gates if self.chart else [])

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return int(width * self._ASPECT_H / self._ASPECT_W)

    def widthForHeight(self, height: int) -> int:
        return int(height * self._ASPECT_W / self._ASPECT_H)

    def sizeHint(self) -> QSize:
        base_w = 400
        return QSize(base_w, self.heightForWidth(base_w))
