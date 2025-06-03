import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, qRgb

from .base_chart import BaseChart
from .enums import InteractionMode
from .gate import IntervalGate


class HistogramChart(BaseChart):
    def __init__(
        self,
        parent=None,
        *,
        num_bins: int = 10,
        origin_zero: bool = True,
        x_axis_label="Value",
        y_axis_label="Count",
    ):
        super().__init__(parent, origin_zero, x_axis_label, y_axis_label)

        # basic settings
        self.num_bins = num_bins
        self.chart_type = "HistogramChart"
        self.value_parameter_name = x_axis_label

        # data containers
        self.bin_width = 0
        self.data_min_val = 0
        self.data_max_val = 1
        self.frequencies = []  # will store counts or densities
        self.bin_id_map = {}

        # display options
        self._relative_freq = False
        self._show_mean = False
        self._global_mean = None  # keeps numeric mean when needed

        # performance cache
        self._density_threshold = 1
        self._hist_img_cache = None

        # gating helpers
        self.current_selection_rect_pixels = None
        self.is_drawing_interval_gate = False

    def clear_temporary_selection_state(self):
        """Overrides BaseChart method."""
        super().clear_temporary_selection_state()
        self.current_selection_rect_pixels = None
        self.is_drawing_interval_gate = False
        self.update()

    def set_relative_frequency(self, flag: bool):
        if self._relative_freq != flag:
            self._relative_freq = flag
            self.y_axis_label = "Density" if flag else "Count"
            self.calculate_frequencies()  # recalculates to density/absolute
            self.update()

    def set_show_mean(self, flag: bool):
        if self._show_mean != flag:
            self._show_mean = flag
            # mean might require recomputation if values changed
            if flag and self.data:
                self._global_mean = np.mean([v[0] for v in self.data])
            self.update()

    def set_data(self, data, value_param_name=None):
        if not isinstance(data, list):
            raise TypeError("Data must be a list.")
        for i, item in enumerate(data):
            if not (isinstance(item, (tuple, list)) and len(item) >= 1):
                raise TypeError(f"Item {i}: needs (value, ...).")
            if not isinstance(item[0], (int, float)):
                raise TypeError(f"Item {i}: value must be numeric.")

        if value_param_name:
            self.value_parameter_name = value_param_name
            self.x_axis_label = value_param_name

        super().set_data(
            data, x_param_name=self.value_parameter_name, y_param_name=self.y_axis_label
        )

        self.calculate_frequencies()
        self.update()

    def calculate_frequencies(self):
        if not self.data:
            self.frequencies = []
            self.bin_id_map = {}
            self.bin_width = 0
            self.data_min_val = 0
            self.data_max_val = 1
            self._global_mean = None
            self._hist_img_cache = None
            return

        values = np.fromiter(
            (item[0] for item in self.data), dtype=np.float64, count=len(self.data)
        )
        self.data_min_val, self.data_max_val = values.min(), values.max()
        if self.data_min_val == self.data_max_val:
            self.data_min_val -= 0.5
            self.data_max_val += 0.5

        n_bins = max(1, self.num_bins)
        data_range = self.data_max_val - self.data_min_val
        self.bin_width = data_range / n_bins

        # numpy histogram (counts)
        freq, _ = np.histogram(
            values, bins=n_bins, range=(self.data_min_val, self.data_max_val)
        )
        self.frequencies = freq.tolist()

        # id map for gating (unchanged logic)
        self.bin_id_map = {i: [] for i in range(n_bins)}
        for idx, v in enumerate(values):
            bin_idx = min(n_bins - 1, int((v - self.data_min_val) / self.bin_width))
            self.bin_id_map[bin_idx].append(idx)

        # relative frequency?
        if self._relative_freq:
            total = values.size
            self.frequencies = [f / total for f in self.frequencies]

        # mean (for mean marker)
        self._global_mean = float(values.mean()) if self._show_mean else None
        self._hist_img_cache = None  # invalidate

    def _render_histogram_image(self, chart_w: int, chart_h: int):
        highest = max(self.frequencies) or 1
        img8 = np.zeros((chart_h, self.num_bins), dtype=np.uint8)
        for i, f in enumerate(self.frequencies):
            h_px = int(f / highest * chart_h)
            if h_px:
                img8[chart_h - h_px :, i] = 255

        img8 = np.repeat(img8, int(np.ceil(chart_w / self.num_bins)), axis=1)
        img8 = img8[:, :chart_w]
        if not img8.flags["C_CONTIGUOUS"]:
            img8 = np.ascontiguousarray(img8)

        qimg = QImage(
            img8.data,
            img8.shape[1],
            img8.shape[0],
            img8.strides[0],
            QImage.Format_Grayscale8,
        )
        qimg.setColorTable([qRgb(i, i, i) for i in range(256)])
        return qimg.copy()

    def draw_chart(self, painter):
        if not self.frequencies:
            return

        plot_w = self.width() - self.margins["left"] - self.margins["right"]
        plot_h = self.height() - self.margins["top"] - self.margins["bottom"]
        if plot_w <= 0 or plot_h <= 0:
            return

        # fast path
        if len(self.data) > self._density_threshold:
            vp_key = (
                plot_w,
                plot_h,
                self.num_bins,
                self.data_min_val,
                self.data_max_val,
                max(self.frequencies),
            )

            if self._hist_img_cache is None or self._hist_img_cache[0] != vp_key:
                img = self._render_histogram_image(plot_w, plot_h)
                self._hist_img_cache = (vp_key, img)
            else:
                img = self._hist_img_cache[1]

            painter.drawImage(self.margins["left"], self.margins["top"], img)
        else:
            # per-bar drawing
            bar_w = plot_w / self.num_bins
            for i, f in enumerate(self.frequencies):
                if f == 0:
                    continue
                start_val = self.data_min_val + i * self.bin_width
                x_pix, _ = self.scale_point((start_val, 0))
                _, y_top = self.scale_point((start_val, f))
                _, y_base = self.scale_point((start_val, 0))
                bar_h_px = abs(y_base - y_top)

                sel_gate = self.get_selected_gate()
                painter.setBrush(QColor(Qt.blue))
                if sel_gate and isinstance(sel_gate, IntervalGate):
                    mid = start_val + self.bin_width / 2
                    if sel_gate.is_point_inside(mid):
                        painter.setBrush(sel_gate.color.lighter(120))
                painter.setPen(Qt.NoPen)
                painter.drawRect(int(x_pix), int(y_top), int(bar_w - 1), int(bar_h_px))

        #  mean marker -
        if self._show_mean and self._global_mean is not None:
            mean_x, _ = self.scale_point((self._global_mean, 0))
            painter.setPen(QPen(Qt.red, 1, Qt.DashLine))
            painter.drawLine(
                int(mean_x),
                self.margins["top"],
                int(mean_x),
                self.height() - self.margins["bottom"],
            )

    def draw_gate_construction_visuals(self, painter):
        """Overrides BaseChart method."""
        # Draw temporary rectangle for IntervalGate mode (which uses RECT_GATE interaction)
        if (
            self.interaction_mode == InteractionMode.RECT_GATE
            and self.current_selection_rect_pixels
        ):
            pen = QPen(QColor(0, 120, 215, 150), 1, Qt.DashLine)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 120, 215, 50))
            painter.setBrush(brush)
            painter.drawRect(self.current_selection_rect_pixels)

    def draw_interval_gate_visualization(self, painter, gate):
        if not isinstance(gate, IntervalGate) or not self.data:
            return

        # Pen is set by BaseChart.draw_gates based on selection state
        # painter.setPen(gate.color)
        painter.setBrush(
            QColor(gate.color.red(), gate.color.green(), gate.color.blue(), 50)
        )

        gate_min_x_pixel, _ = self.scale_point((gate.min_val, 0))
        gate_max_x_pixel, _ = self.scale_point((gate.max_val, 0))

        top_y_pixel = self.margins["top"]
        bottom_y_pixel = self.height() - self.margins["bottom"]

        rect_to_draw = QRectF(
            QPointF(gate_min_x_pixel, top_y_pixel),
            QPointF(gate_max_x_pixel, bottom_y_pixel),
        ).normalized()

        painter.drawRect(rect_to_draw)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if (
                self.interaction_mode == InteractionMode.RECT_GATE
            ):  # Histogram uses RECT_GATE mode for IntervalGate
                self.is_drawing_interval_gate = True
                self.current_selection_rect_pixels = QRect(event.pos(), event.pos())
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self.interaction_mode == InteractionMode.RECT_GATE
            and self.is_drawing_interval_gate
        ):
            start_y = self.margins["top"]
            end_y = self.height() - self.margins["bottom"]

            clamped_x = self._clamp_to_chart_area(event.pos()).x()
            self.current_selection_rect_pixels = QRect(
                QPoint(self.current_selection_rect_pixels.left(), start_y),
                QPoint(clamped_x, end_y),
            ).normalized()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.interaction_mode == InteractionMode.RECT_GATE
            and self.is_drawing_interval_gate
        ):

            self.is_drawing_interval_gate = False

            if (
                self.current_selection_rect_pixels
                and self.current_selection_rect_pixels.width() > 3
            ):

                rect_px = self.current_selection_rect_pixels
                rect_px = QRect(
                    self._clamp_to_chart_area(rect_px.topLeft()),
                    self._clamp_to_chart_area(rect_px.bottomRight()),
                ).normalized()

                mid_y = (
                    self.margins["top"]
                    + (self.height() - self.margins["top"] - self.margins["bottom"]) / 2
                )
                gate_min_val, _ = self.unscale_point(rect_px.left(), mid_y)
                gate_max_val, _ = self.unscale_point(rect_px.right(), mid_y)

                if gate_min_val > gate_max_val:
                    gate_min_val, gate_max_val = gate_max_val, gate_min_val

                gate_name = f"Interval-{len(self.gates) + 1}"
                new_gate = IntervalGate(
                    min_val=gate_min_val,
                    max_val=gate_max_val,
                    name=gate_name,
                    parameter_name=self.value_parameter_name,
                )
                self.add_gate(new_gate)

                self.current_selection_rect_pixels = None
                self.update()
        super().mouseReleaseEvent(event)

    def get_data_range(self):
        return super().get_data_range()

    def generate_x_ticks(self, min_val_data, max_val_data):
        if not self.data and not self.gates:
            return [0, 1]

        num_major_ticks = 5
        if max_val_data == min_val_data:
            return [min_val_data]

        # Ensure num_major_ticks is at least 2 for step calculation if range exists
        actual_num_ticks = max(2, num_major_ticks)
        step = (max_val_data - min_val_data) / (actual_num_ticks - 1)

        ticks = [min_val_data + i * step for i in range(actual_num_ticks)]
        return ticks
