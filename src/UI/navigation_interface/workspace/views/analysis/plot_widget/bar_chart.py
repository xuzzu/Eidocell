from base_chart import BaseChart
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter

# from enums import InteractionMode # Not strictly needed if BarChart doesn't implement new modes


class BarChart(BaseChart):
    def __init__(
        self,
        parent=None,
        origin_zero=True,
        x_axis_label="Category",
        y_axis_label="Value",
    ):
        super().__init__(parent, origin_zero, x_axis_label, y_axis_label)
        self.bar_width_ratio = 0.8  # Ratio of space per category
        # self.bar_spacing = 5 # Replaced by ratio logic
        self.chart_type = "BarChart"  # Important for BaseChart logic
        # Bar chart typically doesn't have interactive gating like scatter/histogram
        # It might have bar selection, which is different.

    def set_data(self, data):
        if not isinstance(data, list) or not all(
            isinstance(item, (tuple, list)) and len(item) == 2 for item in data
        ):
            raise TypeError("Data must be a list of (category, value) pairs.")
        if not all(isinstance(item[1], (int, float)) for item in data):
            raise TypeError("Values must be numeric.")
        # For BarChart, the 'x' in BaseChart's data model is an index, 'y' is the value.
        # Category names are stored separately or used directly for labels.
        # We pass data as is, BaseChart.get_data_range will use chart_type.
        super().set_data(
            data, x_param_name=self.x_axis_label, y_param_name=self.y_axis_label
        )

    def draw_chart(self, painter):
        if not self.data:
            return

        chart_draw_width = self.width() - self.margins["left"] - self.margins["right"]
        num_categories = len(self.data)
        if num_categories == 0 or chart_draw_width <= 0:
            return

        # Width available for each category (bar + spacing)
        width_per_category = chart_draw_width / num_categories
        actual_bar_width = width_per_category * self.bar_width_ratio
        spacing = width_per_category * (1 - self.bar_width_ratio) / 2  # Centered bar

        current_x_pixel = self.margins["left"] + spacing  # Start of first bar

        for i, (category, value) in enumerate(self.data):
            # For BarChart, scale_point's x_data_coord isn't directly used for bar x-pos,
            # but for y_scaling it expects a valid x from data_range.
            # We use category index 'i' as the x_data_coord for scaling context.
            _, bar_top_y_pixel = self.scale_point((i, value))
            _, baseline_y_pixel = self.scale_point((i, 0))
            bar_height_pixel = abs(baseline_y_pixel - bar_top_y_pixel)

            # Simple selection highlighting (not gate-based for BarChart yet)
            # if (category, value) in self.selected_data_points: # Using selected_data_points
            #     painter.setBrush(Qt.red)
            # else:
            #     painter.setBrush(Qt.blue)

            # For now, no interactive selection on bar chart, just draw
            painter.setBrush(Qt.blue)
            painter.setPen(Qt.NoPen)

            painter.drawRect(
                int(current_x_pixel),
                int(bar_top_y_pixel),
                int(actual_bar_width),
                int(bar_height_pixel),
            )
            current_x_pixel += width_per_category

    def mousePressEvent(self, event):
        # Bar chart selection is not implemented with gating system yet.
        # If simple bar selection is needed, it would be implemented here
        # by checking self.interaction_mode == InteractionMode.SELECT_POINTS.
        # For now, pass to base.
        super().mousePressEvent(event)

    def draw_axes(self, painter):
        super().draw_axes(painter)

        # Draw Category Labels on X-axis
        if self.data:
            chart_draw_width = (
                self.width() - self.margins["left"] - self.margins["right"]
            )
            num_categories = len(self.data)
            if num_categories == 0 or chart_draw_width <= 0:
                return

            width_per_category = chart_draw_width / num_categories

            current_x_label_center = self.margins["left"] + width_per_category / 2
            y_label_pos = (
                self.height()
                - self.margins["bottom"]
                + painter.fontMetrics().ascent()
                + 5
            )

            for category, _ in self.data:
                label = str(category)
                text_width = painter.fontMetrics().horizontalAdvance(label)
                painter.drawText(
                    int(current_x_label_center - text_width / 2),
                    int(y_label_pos),
                    label,
                )
                current_x_label_center += width_per_category

    def generate_x_ticks(self, min_val, max_val):
        # For BarChart, min_val=0, max_val=num_categories. Ticks are category centers.
        # However, we draw labels directly in draw_axes. So, no numerical ticks needed.
        return None

    # Y-ticks are handled by BaseChart
