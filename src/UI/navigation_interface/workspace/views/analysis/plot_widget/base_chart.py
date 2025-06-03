import uuid
from typing import Optional

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, Qt
from PySide6.QtWidgets import QSizePolicy, QWidget  # Import QSizePolicy

from .enums import InteractionMode
from .gate import BaseGate, IntervalGate, PolygonGate, RectangularGate


class BaseChart(QWidget):
    selectionChanged = Signal(list)
    gateAdded = Signal(BaseGate)
    gatesUpdated = Signal(list)
    selectedGateChanged = Signal(object)

    def __init__(
        self,
        parent=None,
        origin_zero=True,
        x_axis_label="X",
        y_axis_label="Y",
        chart_aspect_ratio=(1, 1),
    ):  # Aspect ratio for the plot drawing area
        super().__init__(parent)
        self.data = []
        self.selected_data_points = []

        self.gates = []
        self.selected_gate_id = None
        self.interaction_mode = InteractionMode.RECT_GATE
        self.active_gate_construction_state = {}
        self.chart_type = "BaseChart"
        self._chart_aspect_w, self._chart_aspect_h = chart_aspect_ratio

        self.margins = {"left": 60, "right": 30, "top": 30, "bottom": 50}

        self.axis_color = QColor(Qt.black)
        self.origin_zero = origin_zero
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

        self.x_parameter_name = x_axis_label
        self.y_parameter_name = y_axis_label

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: transparent;")

        # Size policy for the chart itself to enforce aspect ratio
        sp = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sp.setHeightForWidth(True)

        self.setSizePolicy(sp)
        self.setMinimumSize(QSize(200, 200))  # Set initial minimum size
        self.setMinimumSize(
            self.minimumSizeHint()
        )  # Set initial minimum size based on hint

    def set_chart_aspect_ratio(self, w, h):
        if w > 0 and h > 0:
            self._chart_aspect_w = w
            self._chart_aspect_h = h
            self.updateGeometry()  # Important: tell layout system to re-evaluate

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        """Entire widget keeps _chart_aspect_w : _chart_aspect_h."""
        return int(width * self._chart_aspect_h / self._chart_aspect_w)

    def widthForHeight(self, height: int) -> int:
        """Helps when the parent layout drives the height first."""
        return int(height * self._chart_aspect_w / self._chart_aspect_h)

    def minimumSizeHint(self):
        min_plot_width = 80  # Absolute minimum for the plotting area itself
        min_plot_height = (
            int(min_plot_width * (self._chart_aspect_h / self._chart_aspect_w))
            if self._chart_aspect_w > 0
            else 60
        )

        return QSize(
            self.margins["left"] + self.margins["right"] + min_plot_width,
            self.margins["top"] + self.margins["bottom"] + min_plot_height,
        )

    def sizeHint(self):
        pref_plot_width = 300  # Preferred width for the plot area
        # Calculate preferred plot height based on aspect ratio and preferred plot width
        pref_plot_height = (
            int(pref_plot_width * (self._chart_aspect_h / self._chart_aspect_w))
            if self._chart_aspect_w > 0
            else 200
        )

        return QSize(
            self.margins["left"] + self.margins["right"] + pref_plot_width,
            self.margins["top"] + self.margins["bottom"] + pref_plot_height,
        )

    def _clamp_to_chart_area(self, point: QPoint) -> QPoint:
        """
        Return a QPoint snapped to the inside of the plot rectangle
        (the area between the axes, i.e. margins removed).
        """
        x_min = self.margins["left"]
        x_max = self.width() - self.margins["right"]
        y_min = self.margins["top"]
        y_max = self.height() - self.margins["bottom"]

        clamped_x = max(x_min, min(point.x(), x_max))
        clamped_y = max(y_min, min(point.y(), y_max))
        return QPoint(clamped_x, clamped_y)

    def set_interaction_mode(self, mode: InteractionMode):
        if self.interaction_mode != mode:
            self.interaction_mode = mode
            self.clear_temporary_selection_state()
            self.update()

    def clear_temporary_selection_state(self):
        self.active_gate_construction_state = {}
        if hasattr(self, "current_selection_rect_pixels"):
            self.current_selection_rect_pixels = None
        if hasattr(self, "is_drawing_rect_gate"):  # Added for scatter
            self.is_drawing_rect_gate = False
        if hasattr(self, "is_drawing_interval_gate"):  # Added for histogram
            self.is_drawing_interval_gate = False
        if hasattr(self, "current_polygon_vertices_pixels"):
            self.current_polygon_vertices_pixels = []
        if hasattr(self, "current_mouse_pos_pixels"):
            self.current_mouse_pos_pixels = None
        if hasattr(self, "is_near_start_vertex_for_closing"):  # Added for scatter
            self.is_near_start_vertex_for_closing = False
            if hasattr(self, "unsetCursor"):
                self.unsetCursor()
        self.update()

    def set_data(self, data, x_param_name=None, y_param_name=None):
        if not isinstance(data, list):
            raise TypeError("Data must be a list.")
        self.data = data
        self.selected_data_points = []

        if x_param_name:
            self.x_parameter_name = x_param_name
            self.x_axis_label = x_param_name
        if y_param_name:
            self.y_parameter_name = y_param_name
            self.y_axis_label = y_param_name

        self.recalculate_all_gate_populations()
        self.updateGeometry()  # Data change might affect ideal size due to range
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        chart_area_width = self.width() - self.margins["left"] - self.margins["right"]
        chart_area_height = self.height() - self.margins["top"] - self.margins["bottom"]

        if (
            chart_area_width <= 1 or chart_area_height <= 1
        ):  # Use a small positive threshold
            return

        min_x_data, max_x_data, min_y_data, max_y_data = self.get_data_range()

        self.x_scale = (
            chart_area_width / (max_x_data - min_x_data)
            if max_x_data != min_x_data
            else 1
        )
        self.y_scale = (
            chart_area_height / (max_y_data - min_y_data)
            if max_y_data != min_y_data
            else 1
        )

        self.draw_axes(painter)
        self.draw_chart(painter)
        self.draw_gates(painter)
        self.draw_gate_construction_visuals(painter)

    def draw_gate_construction_visuals(self, painter):
        pass

    def scale_point(self, point_data_coords):
        min_x, max_x, min_y, max_y = self.get_data_range()
        chart_width = self.width() - self.margins["left"] - self.margins["right"]
        chart_height = self.height() - self.margins["top"] - self.margins["bottom"]

        if chart_width <= 0 or chart_height <= 0:
            return float(self.margins["left"]), float(
                self.height() - self.margins["bottom"]
            )

        data_x, data_y = point_data_coords

        current_x_scale = (
            self.x_scale if self.x_scale != 0 and (max_x != min_x) else 1.0
        )
        current_y_scale = (
            self.y_scale if self.y_scale != 0 and (max_y != min_y) else 1.0
        )

        if max_x == min_x:
            x_scaled = self.margins["left"] + chart_width / 2.0
        else:
            x_scaled = self.margins["left"] + (data_x - min_x) * current_x_scale

        if max_y == min_y:
            y_scaled = self.height() - self.margins["bottom"] - chart_height / 2.0
        else:
            y_scaled = (
                self.height()
                - self.margins["bottom"]
                - (data_y - min_y) * current_y_scale
            )

        return float(x_scaled), float(y_scaled)

    def unscale_point(self, pixel_x, pixel_y):
        min_x, max_x, min_y, max_y = self.get_data_range()
        chart_width = self.width() - self.margins["left"] - self.margins["right"]
        chart_height = self.height() - self.margins["top"] - self.margins["bottom"]

        if chart_width <= 0 or chart_height <= 0:
            return float(min_x), float(min_y)

        current_x_scale = (
            self.x_scale if self.x_scale != 0 and (max_x != min_x) else 1.0
        )
        current_y_scale = (
            self.y_scale if self.y_scale != 0 and (max_y != min_y) else 1.0
        )

        if max_x == min_x:
            data_x = min_x
        else:
            data_x = min_x + ((pixel_x - self.margins["left"]) / current_x_scale)

        if max_y == min_y:
            data_y = min_y
        else:
            data_y = min_y + (
                (self.height() - self.margins["bottom"] - pixel_y) / current_y_scale
            )

        return float(data_x), float(data_y)

    def draw_chart(self, painter):
        raise NotImplementedError("Subclasses must implement draw_chart")

    def draw_gates(self, painter):
        for gate in self.gates:
            original_pen = painter.pen()
            original_brush = painter.brush()

            if self.selected_gate_id == gate.id:
                sel_pen = QPen(gate.color.lighter(150), 3)
                painter.setPen(sel_pen)
            else:
                painter.setPen(QPen(gate.color, 1))

            if isinstance(gate, IntervalGate) and hasattr(
                self, "draw_interval_gate_visualization"
            ):
                self.draw_interval_gate_visualization(painter, gate)
            elif hasattr(gate, "draw"):
                gate.draw(painter, self.scale_point)

            painter.setPen(original_pen)
            painter.setBrush(original_brush)

    def add_gate(self, gate_obj: BaseGate):
        """
        Adds a gate object to this chart's local list and emits gateAdded.
        Called by derived charts (ScatterChart, HistogramChart) when user finishes drawing a gate.
        The gate_obj.parameters should be set by the derived chart before calling this.
        """
        if not isinstance(gate_obj, BaseGate):
            raise TypeError("Can only add objects of type BaseGate or its subclasses.")

        # Ensure gate.parameters are set based on current chart axes
        if isinstance(gate_obj, (RectangularGate, PolygonGate)):
            expected_params = (self.x_parameter_name, self.y_parameter_name)
            if gate_obj.parameters != expected_params:
                gate_obj.parameters = expected_params
        elif isinstance(gate_obj, IntervalGate):
            expected_param = (self.x_parameter_name,)  # Interval gates are 1D on X-axis
            if gate_obj.parameters != expected_param:
                gate_obj.parameters = expected_param

        self.gates.append(gate_obj)
        self._update_local_gate_population(
            gate_obj
        )  # Update count based on current chart data
        self.gateAdded.emit(gate_obj)  # Signal to InteractiveChartWidget
        self.gatesUpdated.emit(self.gates)  # For table refresh
        self.select_gate(gate_obj.id)
        self.update()

    def remove_gate(self, gate_id_to_remove: uuid.UUID):
        gate_to_remove = next(
            (g for g in self.gates if g.id == gate_id_to_remove), None
        )
        if gate_to_remove:
            self.gates.remove(gate_to_remove)
            if self.selected_gate_id == gate_id_to_remove:
                self.select_gate(None)
            self.gatesUpdated.emit(self.gates)
            self.update()

    def select_gate(self, gate_id: Optional[uuid.UUID]):
        if self.selected_gate_id != gate_id:
            self.selected_gate_id = gate_id
            selected_gate_obj = self.get_selected_gate()
            self.selectedGateChanged.emit(selected_gate_obj)
            self.update()

    def get_selected_gate(self):
        if self.selected_gate_id is None:
            return None
        return next((g for g in self.gates if g.id == self.selected_gate_id), None)

    def clear_gates(self):
        self.gates = []
        self.select_gate(None)
        self.gatesUpdated.emit(self.gates)
        self.update()

    def recalculate_gate_population(self, gate):
        if not self.data or not self.data[0]:
            gate.ids_in_gate = []
            gate.event_count = 0
            return

        scatter_id_idx = None
        if self.data and self.data[0] is not None:
            last_el_idx = len(self.data[0]) - 1
            if (
                last_el_idx >= 0
                and isinstance(self.data[0][last_el_idx], (str, int, float))
                and not isinstance(self.data[0][last_el_idx], QColor)
            ):
                if len(self.data[0]) == 5:
                    scatter_id_idx = 4
                elif len(self.data[0]) == 3:
                    scatter_id_idx = 2

        hist_id_idx = 1 if len(self.data[0]) > 1 else None
        if hist_id_idx is not None and (
            hist_id_idx >= len(self.data[0])
            or not isinstance(self.data[0][hist_id_idx], (str, int, float))
        ):
            hist_id_idx = None

        if (
            self.chart_type == "ScatterChart"
            and isinstance(gate, RectangularGate)
            and hasattr(self, "_xs_np")
        ):
            import numpy as np

            xs = self._xs_np
            ys = self._ys_np

            if hasattr(self, "_ids_np"):
                ids = self._ids_np
            else:
                ids = np.arange(len(self.data), dtype=np.int32)

            mask = (
                (xs >= gate.rect.left())
                & (xs <= gate.rect.right())
                & (ys >= gate.rect.top())
                & (ys <= gate.rect.bottom())
            )

            gate.ids_in_gate = ids[mask].tolist()
            gate.event_count = int(mask.sum())
            return

        if self.chart_type == "ScatterChart" and isinstance(
            gate, (RectangularGate, PolygonGate)
        ):
            gate.update_contained_ids(
                self.data, x_param_idx=0, y_param_idx=1, id_param_idx=scatter_id_idx
            )

        elif self.chart_type == "HistogramChart" and isinstance(gate, IntervalGate):
            gate.update_contained_ids(
                self.data, value_param_idx=0, id_param_idx=hist_id_idx
            )

        else:  # Fallback for any other chart / gate type
            id_idx_guess = None
            last_el_idx = len(self.data[0]) - 1
            if isinstance(
                self.data[0][last_el_idx], (str, int, float)
            ) and not isinstance(self.data[0][last_el_idx], QColor):
                id_idx_guess = last_el_idx

            if isinstance(gate, (RectangularGate, PolygonGate)):
                gate.update_contained_ids(
                    self.data, x_param_idx=0, y_param_idx=1, id_param_idx=id_idx_guess
                )
            elif isinstance(gate, IntervalGate):
                gate.update_contained_ids(
                    self.data, value_param_idx=0, id_param_idx=id_idx_guess
                )

    def recalculate_all_gate_populations(self):
        if not self.data:
            for gate in self.gates:
                gate.ids_in_gate = []
                gate.event_count = 0
            if self.gates:
                self.gatesUpdated.emit(self.gates)
            return

        for gate in self.gates:
            self.recalculate_gate_population(gate)

        if self.gates:
            self.gatesUpdated.emit(self.gates)

    def _update_local_gate_population(self, gate: BaseGate):
        """
        Updates the gate's event count based on the data *currently plotted on this chart*.
        This provides immediate feedback to the user. The authoritative, session-wide
        count is managed by DataManager.
        """
        if not self.data:
            gate.event_count = 0  # Local count for this chart
            return

        if self.chart_type == "ScatterChart":
            # Scatter data: x=idx 0, y=idx 1. ID might be at idx 4.
            id_idx = 4 if self.data and len(self.data[0]) > 4 else None
            gate.update_contained_ids(
                self.data, x_param_idx=0, y_param_idx=1, id_param_idx=id_idx
            )
        elif self.chart_type == "HistogramChart":
            # Histogram data: value=idx 0. ID might be at idx 1.
            id_idx = 1 if self.data and len(self.data[0]) > 1 else None
            gate.update_contained_ids(self.data, value_param_idx=0, id_param_idx=id_idx)
        else:
            # Fallback for unknown chart type or if BaseChart is used directly
            gate.update_contained_ids(self.data)

        # print(f"BaseChart ({self.x_parameter_name}): Local pop for gate {gate.name}: {gate.event_count}")

    def _recalculate_local_gate_populations(self):
        """Recalculates populations for all gates based on current chart data."""
        if not self.data:
            for gate in self.gates:
                gate.event_count = 0
            if self.gates:
                self.gatesUpdated.emit(self.gates)
            return

        for gate in self.gates:
            self._update_local_gate_population(gate)

        if self.gates:
            self.gatesUpdated.emit(self.gates)  # For table update

    def get_selected_data_points(self):
        return self.selected_data_points

    def clear_selection(self):
        self.selected_data_points = []
        self.selectionChanged.emit(self.selected_data_points)
        self.update()

    def generate_ticks(self, min_val, max_val, num_ticks=5):
        if num_ticks < 2:
            num_ticks = 2
        if min_val == max_val:
            return (
                [min_val - 0.5, min_val, min_val + 0.5] if num_ticks > 1 else [min_val]
            )
        if num_ticks - 1 == 0:
            return [min_val]
        step = (max_val - min_val) / (num_ticks - 1)
        return [min_val + i * step for i in range(num_ticks)]

    def draw_axes(self, painter):
        width = self.width()
        height = self.height()
        painter.setPen(self.axis_color)
        m = self.margins
        chart_x0 = m["left"]
        chart_y0 = m["top"]
        chart_x1 = width - m["right"]
        chart_y1 = height - m["bottom"]
        if chart_x1 - chart_x0 <= 0 or chart_y1 - chart_y0 <= 0:
            return

        painter.drawLine(chart_x0, chart_y1, chart_x1, chart_y1)  # x-axis
        painter.drawLine(chart_x0, chart_y0, chart_x0, chart_y1)  # y-axis

        metrics = QFontMetrics(painter.font())
        tick_h = metrics.height()

        x_label = self.x_axis_label or self.x_parameter_name
        x_lbl_w = metrics.horizontalAdvance(x_label)
        x_lbl_x = chart_x0 + (chart_x1 - chart_x0 - x_lbl_w) / 2
        x_lbl_y = chart_y1 + tick_h + 20  # PATCHED
        painter.drawText(int(x_lbl_x), int(x_lbl_y), x_label)

        y_label = self.y_axis_label or self.y_parameter_name
        rot_w = metrics.horizontalAdvance(y_label)
        y_lbl_x = chart_x0 - tick_h - 30  # PATCHED
        y_lbl_y = chart_y0 + (chart_y1 - chart_y0 + rot_w) / 2

        painter.save()
        painter.translate(y_lbl_x, y_lbl_y)
        painter.rotate(-90)
        painter.drawText(0, 0, y_label)
        painter.restore()

        min_x, max_x, min_y, max_y = self.get_data_range()
        self.draw_x_ticks(painter, self.generate_x_ticks(min_x, max_x))
        self.draw_y_ticks(painter, self.generate_y_ticks(min_y, max_y))

    def generate_x_ticks(self, min_val, max_val):
        return self.generate_ticks(min_val, max_val)

    def generate_y_ticks(self, min_val, max_val):
        return self.generate_ticks(min_val, max_val)

    def get_data_range(self):
        min_x_final, max_x_final = 0.0, 1.0
        min_y_final, max_y_final = 0.0, 1.0
        data_present = bool(self.data and self.data[0] is not None)
        gates_present = bool(self.gates)

        if not data_present and not gates_present:
            return 0.0, 100.0, 0.0, 100.0

        if data_present:
            first_x_val = self.data[0][0]
            if self.chart_type == "BarChart":
                min_x_data = 0.0
                max_x_data = float(len(self.data))
            elif self.chart_type == "HistogramChart":
                min_x_data = getattr(self, "data_min_val", 0.0)
                max_x_data = getattr(self, "data_max_val", 1.0)
            elif isinstance(first_x_val, (int, float)):
                all_x_data = [
                    p[0]
                    for p in self.data
                    if p is not None and len(p) > 0 and isinstance(p[0], (int, float))
                ]  # check len(p)>0
                if not all_x_data:
                    min_x_data, max_x_data = 0.0, 1.0
                else:
                    min_x_data, max_x_data = min(all_x_data), max(all_x_data)
            else:
                min_x_data = 0.0
                max_x_data = float(len(self.data))

            if self.chart_type == "HistogramChart":
                min_y_data = 0.0
                # Ensure frequencies is not empty before calling max
                frequencies = getattr(self, "frequencies", [])
                max_y_data = max(frequencies) if frequencies else 1.0
            else:
                valid_y_points = [
                    p[1]
                    for p in self.data
                    if p is not None and len(p) > 1 and isinstance(p[1], (int, float))
                ]
                if valid_y_points:
                    min_y_data = min(valid_y_points)
                    max_y_data = max(valid_y_points)
                else:
                    min_y_data, max_y_data = 0.0, 1.0

            min_x_final, max_x_final = min_x_data, max_x_data
            min_y_final, max_y_final = min_y_data, max_y_data

        if gates_present:
            gate_min_x_overall, gate_max_x_overall = float("inf"), float("-inf")
            gate_min_y_overall, gate_max_y_overall = float("inf"), float("-inf")
            any_gate_has_extent = False
            for gate in self.gates:
                g_rect = None
                current_gate_min_x, current_gate_max_x = None, None
                current_gate_min_y, current_gate_max_y = None, None
                if isinstance(gate, RectangularGate):
                    g_rect = gate.rect
                elif isinstance(gate, PolygonGate):
                    g_rect = gate.polygon.boundingRect()
                elif isinstance(gate, IntervalGate):
                    current_gate_min_x, current_gate_max_x = gate.min_val, gate.max_val
                if g_rect and g_rect.isValid():
                    current_gate_min_x, current_gate_max_x = (
                        g_rect.left(),
                        g_rect.right(),
                    )
                    current_gate_min_y, current_gate_max_y = (
                        g_rect.top(),
                        g_rect.bottom(),
                    )
                if current_gate_min_x is not None:
                    any_gate_has_extent = True
                    gate_min_x_overall = min(gate_min_x_overall, current_gate_min_x)
                    gate_max_x_overall = max(gate_max_x_overall, current_gate_max_x)
                if current_gate_min_y is not None:
                    gate_min_y_overall = min(gate_min_y_overall, current_gate_min_y)
                    gate_max_y_overall = max(gate_max_y_overall, current_gate_max_y)
            if any_gate_has_extent:
                min_x_final = (
                    min(min_x_final, gate_min_x_overall)
                    if data_present and min_x_final != float("inf")
                    else gate_min_x_overall
                )
                max_x_final = (
                    max(max_x_final, gate_max_x_overall)
                    if data_present and max_x_final != float("-inf")
                    else gate_max_x_overall
                )
                if gate_min_y_overall != float("inf"):
                    min_y_final = (
                        min(min_y_final, gate_min_y_overall)
                        if data_present and min_y_final != float("inf")
                        else gate_min_y_overall
                    )
                    max_y_final = (
                        max(max_y_final, gate_max_y_overall)
                        if data_present and max_y_final != float("-inf")
                        else gate_max_y_overall
                    )
                elif not data_present:
                    min_y_final, max_y_final = (
                        (0.0, 1.0)
                        if min_y_final == float("inf")
                        else (min_y_final, max_y_final)
                    )  # Ensure Y is initialized if only interval gates

        if self.origin_zero:
            min_x_final = min(0.0, min_x_final) if min_x_final != float("inf") else 0.0
            min_y_final = min(0.0, min_y_final) if min_y_final != float("inf") else 0.0

        if min_x_final == float("inf") or min_x_final == float("-inf"):
            min_x_final = 0.0
        if max_x_final == float("inf") or max_x_final == float("-inf"):
            max_x_final = 1.0 if min_x_final == 0.0 else min_x_final + 1.0
        if min_y_final == float("inf") or min_y_final == float("-inf"):
            min_y_final = 0.0
        if max_y_final == float("inf") or max_y_final == float("-inf"):
            max_y_final = 1.0 if min_y_final == 0.0 else min_y_final + 1.0

        if min_x_final == max_x_final:
            delta = 0.5 if abs(min_x_final) < 1e-6 else abs(min_x_final * 0.5)
            delta = max(delta, 0.5)
            min_x_final -= delta
            max_x_final += delta
        if min_y_final == max_y_final:
            delta = 0.5 if abs(min_y_final) < 1e-6 else abs(min_y_final * 0.5)
            delta = max(delta, 0.5)
            min_y_final -= delta
            max_y_final += delta
        return min_x_final, max_x_final, min_y_final, max_y_final

    def draw_x_ticks(self, painter, ticks):
        if not ticks:
            return
        valid_ticks = [
            t for t in ticks if t is not None and isinstance(t, (int, float))
        ]
        if not valid_ticks:
            return
        _, _, min_y_data_for_scaling, _ = self.get_data_range()
        chart_bottom_y = self.height() - self.margins["bottom"]
        for tick_val in valid_ticks:
            x_pixel, _ = self.scale_point((tick_val, min_y_data_for_scaling))
            tick_start_y = chart_bottom_y
            tick_end_y = chart_bottom_y + 5
            painter.drawLine(
                int(x_pixel), int(tick_start_y), int(x_pixel), int(tick_end_y)
            )
            label = f"{tick_val:.2g}"
            text_width = painter.fontMetrics().horizontalAdvance(label)
            text_x = int(x_pixel - text_width / 2)
            text_y = int(tick_end_y + painter.fontMetrics().ascent() + 2)
            painter.drawText(text_x, text_y, label)

    def draw_y_ticks(self, painter, ticks):
        if not ticks:
            return
        valid_ticks = [
            t for t in ticks if t is not None and isinstance(t, (int, float))
        ]
        if not valid_ticks:
            return
        min_x_data_for_scaling, _, _, _ = self.get_data_range()
        chart_left_x = self.margins["left"]
        for tick_val in valid_ticks:
            _, y_pixel = self.scale_point((min_x_data_for_scaling, tick_val))
            tick_start_x = chart_left_x
            tick_end_x = chart_left_x - 5
            painter.drawLine(
                int(tick_start_x), int(y_pixel), int(tick_end_x), int(y_pixel)
            )
            label = f"{tick_val:.2g}"
            text_width = painter.fontMetrics().horizontalAdvance(label)
            text_x = int(tick_end_x - text_width - 5)
            text_y = int(
                y_pixel
                + painter.fontMetrics().ascent() / 2
                - painter.fontMetrics().descent() / 2
            )
            painter.drawText(text_x, text_y, label)
