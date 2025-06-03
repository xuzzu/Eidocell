# UI/widgets/scale_widgets.py
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class BaseScaleWidget(QWidget):
    """Base class for horizontal and vertical scale widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0  # units per pixel
        self.units = "pixels"
        self.image_dimension_pixels = 0
        self.major_tick_color = QColor(Qt.black)
        self.minor_tick_color = QColor(Qt.darkGray)  # Slightly darker minor ticks
        self.text_color = QColor(Qt.black)
        self.line_color = QColor(Qt.black)
        self.tick_font = QFont("Arial", 10)  # Increased from 8
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Default policy

    def setScale(self, scale_factor: float, units: str, image_dimension_pixels: int):
        if scale_factor <= 0 or image_dimension_pixels <= 0:
            self.scale_factor = 1.0
            self.units = "pixels"
            self.image_dimension_pixels = 0
        else:
            self.scale_factor = scale_factor
            self.units = units
            self.image_dimension_pixels = image_dimension_pixels
        self.update()  # Trigger repaint

    def _format_label(self, value: float) -> str:
        """
        Format the tick label with e.g. 2 or 3 significant digits
        consistently in either decimal or scientific notation.
        """
        if value == 0:
            return "0"
        abs_val = abs(value)
        if abs_val < 1e-3 or abs_val >= 1e5:
            return f"{value:.3g}"
        else:
            return f"{value:.3g}"

    def _calculate_ticks(self, widget_length_pixels):
        """Calculates major and minor tick positions and labels."""
        if self.image_dimension_pixels == 0 or self.scale_factor == 0:
            return [], []

        total_real_length = self.image_dimension_pixels * self.scale_factor
        if total_real_length <= 1e-9:
            return [], []  # Avoid issues with near-zero length

        # Determine a reasonable number of major ticks (e.g., 5-10)
        target_ticks = 6
        exponent = (
            math.floor(math.log10(total_real_length / target_ticks))
            if total_real_length > 0
            else 0
        )
        base_interval = 10**exponent
        possible_intervals = [base_interval * x for x in [1, 2, 5, 10]]
        min_sensible_interval = total_real_length / 20
        possible_intervals = [
            i for i in possible_intervals if i > min_sensible_interval
        ]
        if not possible_intervals:
            possible_intervals = [base_interval * x for x in [1, 2, 5, 10]]
        best_interval = min(
            possible_intervals,
            key=lambda x: (
                abs(total_real_length / x - target_ticks) if x > 0 else float("inf")
            ),
        )
        major_tick_interval_real = best_interval
        major_tick_interval_pixels = major_tick_interval_real / self.scale_factor
        minor_ticks_per_major = 5
        minor_tick_interval_pixels = major_tick_interval_pixels / minor_ticks_per_major
        major_ticks = []
        minor_ticks = []
        pixel_mapping_ratio = (
            widget_length_pixels / self.image_dimension_pixels
            if self.image_dimension_pixels
            else 0
        )
        current_pixel_pos_image = 0
        while current_pixel_pos_image <= self.image_dimension_pixels + 1e-9:
            widget_pos = current_pixel_pos_image * pixel_mapping_ratio
            real_value = current_pixel_pos_image * self.scale_factor
            label = self._format_label(real_value)
            major_ticks.append((widget_pos, label))
            next_major_pixel_pos_image = (
                current_pixel_pos_image + major_tick_interval_pixels
            )
            for i in range(1, minor_ticks_per_major):
                minor_pixel_pos_image = (
                    current_pixel_pos_image + i * minor_tick_interval_pixels
                )
                if minor_pixel_pos_image < next_major_pixel_pos_image - 1e-9:
                    minor_widget_pos = minor_pixel_pos_image * pixel_mapping_ratio
                    if minor_widget_pos < widget_length_pixels:
                        minor_ticks.append(minor_widget_pos)
            current_pixel_pos_image += major_tick_interval_pixels
            if major_tick_interval_pixels <= 1e-9:
                break
        return major_ticks, minor_ticks


class HorizontalScaleWidget(BaseScaleWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(45)  # Increased from 30
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        fm = QFontMetrics(self.tick_font)
        widget_width = self.width()
        widget_height = self.height()

        line_y = widget_height * 0.35  # Line higher up
        minor_tick_height = 6  # Increased minor tick length
        major_tick_height = 10  # Increased major tick length
        text_y = (
            line_y + major_tick_height + fm.height() + 2
        )  # More space below ticks for text

        # Draw main horizontal line
        painter.setPen(QPen(self.line_color, 1.5))  # Slightly thicker line
        painter.drawLine(0, line_y, widget_width, line_y)

        major_ticks, minor_ticks = self._calculate_ticks(widget_width)

        # Draw minor ticks
        painter.setPen(QPen(self.minor_tick_color, 1))
        for x_pos in minor_ticks:
            painter.drawLine(int(x_pos), line_y, int(x_pos), line_y + minor_tick_height)

        # Draw major ticks and labels
        painter.setPen(QPen(self.major_tick_color, 1.5))  # Thicker major ticks
        painter.setFont(self.tick_font)
        painter.setPen(self.text_color)  # Set text color

        last_label_rect = QRectF()  # Keep track of last label position

        for i, (x_pos, label) in enumerate(major_ticks):
            painter.setPen(QPen(self.major_tick_color, 1.5))  # Reset pen for tick
            painter.drawLine(int(x_pos), line_y, int(x_pos), line_y + major_tick_height)

            painter.setPen(self.text_color)  # Set pen for text
            display_label = label
            # Add units only to the last label
            if i == len(major_ticks) - 1:
                display_label += f" {self.units}"

            text_width = fm.horizontalAdvance(display_label)
            text_rect = QRectF(
                x_pos - text_width / 2, text_y - fm.height(), text_width, fm.height()
            )

            # Adjust position to prevent overlap and going out of bounds
            if text_rect.right() > widget_width - 5:
                text_rect.moveRight(widget_width - 5)
            if text_rect.left() < 5:
                text_rect.moveLeft(5)

            # Prevent label overlap
            if i > 0 and text_rect.intersects(last_label_rect):
                continue  # Skip drawing this label if it overlaps the previous one

            painter.drawText(text_rect, Qt.AlignCenter, display_label)
            last_label_rect = text_rect


class VerticalScaleWidget(BaseScaleWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(85)  # Increased from 50
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._label_gap = 6

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        fm = QFontMetrics(self.tick_font)
        widget_w = self.width()
        widget_h = self.height()

        line_x = widget_w * 0.35
        minor_len = 6
        major_len = 10
        text_start_x = line_x + major_len + self._label_gap

        # Main vertical line
        painter.setPen(QPen(self.line_color, 1.5))
        painter.drawLine(line_x, 0, line_x, widget_h)

        major, minor = self._calculate_ticks(widget_h)

        painter.setPen(QPen(self.minor_tick_color, 1))
        for y in minor:
            painter.drawLine(line_x, int(y), line_x + minor_len, int(y))

        painter.setFont(self.tick_font)
        last_label_rect = QRectF()

        for idx, (y, label) in enumerate(major):
            # tick
            painter.setPen(QPen(self.major_tick_color, 1.5))
            painter.drawLine(line_x, int(y), line_x + major_len, int(y))

            # full text (units on the last label)
            text = label + (f" {self.units}" if idx == len(major) - 1 else "")

            #  draw the text so that its *baseline* sits at y
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(text)
            ascent = fm.ascent()
            descent = fm.descent()

            baseline_x = text_start_x
            baseline_y = y + (ascent - descent) / 2

            # bounding-box we’ll use for overlap / boundary checks
            label_rect = QRectF(baseline_x, baseline_y - ascent, tw, ascent + descent)

            # keep inside the widget & avoid overlap
            if label_rect.bottom() > widget_h - 4:
                shift = label_rect.bottom() - (widget_h - 4)
                label_rect.translate(0, -shift)
                baseline_y -= shift
            if label_rect.intersects(last_label_rect):
                continue

            painter.setPen(self.text_color)
            painter.drawText(QPointF(baseline_x, baseline_y), text)
            last_label_rect = label_rect
