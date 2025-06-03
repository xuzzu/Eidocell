### UI\navigation_interface\workspace\views\analysis\plot_widget\scatter_chart.py
import math
import random
from collections import defaultdict

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    qRgb,
)

from .base_chart import BaseChart
from .enums import InteractionMode
from .gate import PolygonGate, RectangularGate

SNAP_DISTANCE_PIXELS = 10
MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK = 3


class ScatterChart(BaseChart):
    def __init__(
        self,
        parent=None,
        *,
        x_axis_label="X",
        y_axis_label="Y",
        point_size=8,
        point_opacity=1.0,
        origin_zero=True,
        density_threshold=1,
    ):  # Default density threshold for switching to heatmap
        super().__init__(parent, origin_zero, x_axis_label, y_axis_label)

        self.chart_type = "ScatterChart"
        self.point_size = point_size
        self.point_opacity = point_opacity

        # big-data performance
        self._density_threshold = density_threshold
        self._density_cache = None

        # gating helpers
        self.current_selection_rect_pixels = None
        self.is_drawing_rect_gate = False
        self.current_polygon_vertices_pixels = []
        self.is_near_start_vertex_for_closing = False

        # style controls
        self._color_by = None
        self._trendline = "none"
        self._marginal_x = False
        self._marginal_y = False

        # cached arrays for fast ops
        self._xs_np = self._ys_np = self._ids_np = None
        self._groups_np = None  # For color_by groups
        self._group_to_color = {}

        self.thread_pool = QThreadPool.globalInstance()
        self.active_density_worker_key = (
            None  # Stores the key of the currently running worker
        )

        self.setMouseTracking(True)  # Important for polygon gate preview
        self.setFocusPolicy(Qt.StrongFocus)  # For key events (Esc, Enter for polygon)

    def set_style(
        self, *, color_by=None, trendline="none", marginal_x=False, marginal_y=False
    ):
        self._color_by = color_by or None
        self._trendline = trendline  # "none", "global", "per group"
        self._marginal_x = bool(marginal_x)
        self._marginal_y = bool(marginal_y)
        self._build_color_map()  # Update color map based on new data/grouping
        self.update()

    def clear_temporary_selection_state(self):
        """Overrides BaseChart method."""
        super().clear_temporary_selection_state()
        self.update()

    def set_data(self, data, x_param_name=None, y_param_name=None):
        """
        Sets the data for the scatter plot.
        Data format: List of tuples. Each tuple should be:
        (x_value, y_value, optional_size_value, optional_group_value_for_color, optional_id_string)
        """
        if not isinstance(data, list):
            raise TypeError("Data must be a list.")

        # Basic validation for each point
        for point_idx, point in enumerate(data):
            if not isinstance(point, (tuple, list)):
                raise TypeError(f"Data point {point_idx} must be a tuple or list.")
            if (
                len(point) < 2 or len(point) > 5
            ):  # x, y are mandatory. size, group, id are optional.
                raise TypeError(
                    f"Data point {point_idx} needs 2 to 5 elements (x, y, [size, group, id]), got {len(point)}."
                )
            if not all(
                isinstance(coord, (int, float)) for coord in point[:2]
            ):  # x, y must be numeric
                raise TypeError(
                    f"Data point {point_idx}: x and y coordinates must be numeric."
                )
            # Optional: Validate size if present (index 2)
            if (
                len(point) >= 3
                and point[2] is not None
                and not isinstance(point[2], (int, float))
            ):
                raise TypeError(
                    f"Data point {point_idx}: size (at index 2) must be numeric or None."
                )

        super().set_data(data, x_param_name, y_param_name)  # Calls BaseChart.set_data

        #  Prepare NumPy arrays for faster operations
        self.normalized_size_variable = None
        if self.data and len(self.data[0]) >= 3:
            sizes = [
                p[2] if len(p) >= 3 and isinstance(p[2], (int, float)) else None
                for p in self.data
            ]
            if any(s is not None for s in sizes):
                valid_sizes = [s for s in sizes if s is not None]
                if valid_sizes:
                    min_s, max_s = min(valid_sizes), max(valid_sizes)
                    range_s = max_s - min_s if max_s > min_s else 1.0
                    self.normalized_size_variable = [
                        ((s - min_s) / range_s if s is not None else 0.5) for s in sizes
                    ]
                else:
                    self.normalized_size_variable = [0.5] * len(
                        self.data
                    )  # Default if no valid sizes

        if self.data:
            self._xs_np = np.fromiter(
                (p[0] for p in self.data), dtype=np.float32, count=len(self.data)
            )
            self._ys_np = np.fromiter(
                (p[1] for p in self.data), dtype=np.float32, count=len(self.data)
            )
            # Assuming ID is the last element (index 4) if present
            self._ids_np = np.array(
                [p[4] if len(p) == 5 else i for i, p in enumerate(self.data)],
                dtype=object,
            )  # Store original IDs or indices

            # Extract groups for coloring if _color_by is set and data has group info (index 3)
            if self._color_by and len(self.data[0]) >= 4:
                self._groups_np = np.array(
                    [p[3] if len(p) >= 4 else None for p in self.data], dtype=object
                )
            else:
                self._groups_np = None
        else:
            self._xs_np = np.array([], dtype=np.float32)
            self._ys_np = np.array([], dtype=np.float32)
            self._ids_np = np.array([], dtype=object)
            self._groups_np = None

        self._build_color_map()  # Rebuild color map if data/groups change
        self._density_cache = None  # Invalidate density cache
        self.update()  # Trigger repaint

    def _build_color_map(self):
        if not self._color_by or self._groups_np is None or len(self._groups_np) == 0:
            self._group_to_color = {}
            return

        # Define a palette
        palette = [
            QColor("#1f77b4"),
            QColor("#ff7f0e"),
            QColor("#2ca02c"),
            QColor("#d62728"),
            QColor("#9467bd"),
            QColor("#8c564b"),
            QColor("#e377c2"),
            QColor("#7f7f7f"),
            QColor("#bcbd22"),
            QColor("#17becf"),
        ]

        unique_groups = np.unique(
            self._groups_np[self._groups_np != np.array(None)]
        )  # Get unique non-None group values
        self._group_to_color = {
            group: palette[i % len(palette)] for i, group in enumerate(unique_groups)
        }

    def _make_density_image(self, plot_w, plot_h, min_x, max_x, min_y, max_y):
        if self._xs_np is None or self._ys_np is None or len(self._xs_np) == 0:
            return QImage()

        xs_all = self._xs_np
        ys_all = self._ys_np

        # Filter points that are within the current view's data range
        inside_mask = (
            (xs_all >= min_x)
            & (xs_all <= max_x)
            & (ys_all >= min_y)
            & (ys_all <= max_y)
        )

        xs_visible = xs_all[inside_mask]
        ys_visible = ys_all[inside_mask]

        if xs_visible.size == 0:  # No points visible in the current zoom/pan
            return QImage()

        # Create 2D histogram
        # Number of bins for density plot matches pixel dimensions for direct mapping
        x_bins = np.linspace(min_x, max_x, plot_w + 1, dtype=np.float32)
        y_bins = np.linspace(min_y, max_y, plot_h + 1, dtype=np.float32)

        hist2d, _, _ = np.histogram2d(ys_visible, xs_visible, bins=(y_bins, x_bins))

        # Apply log scale for better visibility of sparse regions, then normalize
        hist2d = np.log1p(hist2d)
        max_val = hist2d.max()
        if max_val > 0:
            hist2d /= max_val  # Normalize to 0-1

        img8_raw = (hist2d * 255).astype(np.uint8)

        img8 = np.flipud(img8_raw)

        if not img8.flags["C_CONTIGUOUS"]:  # Ensure C-contiguous for QImage
            img8 = np.ascontiguousarray(img8)

        qimg = QImage(
            img8.data,
            img8.shape[1],
            img8.shape[0],
            img8.strides[0],
            QImage.Format_Grayscale8,
        )
        qimg.setColorTable([qRgb(i, i, i) for i in range(256)])  # Grayscale palette
        return qimg.copy()  # Return a copy to avoid issues if img8 is modified

    def draw_chart(self, painter):
        if not self.data:
            return

        # Get current chart drawing area dimensions
        plot_w = self.width() - self.margins["left"] - self.margins["right"]
        plot_h = self.height() - self.margins["top"] - self.margins["bottom"]
        if plot_w <= 0 or plot_h <= 0:
            return

        # Get current data range based on zoom/pan (from BaseChart)
        min_x, max_x, min_y, max_y = self.get_data_range()

        if len(self.data) > self._density_threshold:
            current_view_key = (
                plot_w,
                plot_h,
                min_x,
                max_x,
                min_y,
                max_y,
                id(self._xs_np),
            )

            if (
                self._density_cache is None
                or self._density_cache[0] != current_view_key
            ):
                # Cache is invalid or doesn't exist, regenerate density image
                density_image = self._make_density_image(
                    plot_w, plot_h, min_x, max_x, min_y, max_y
                )
                self._density_cache = (current_view_key, density_image)
            else:
                # Use cached image
                density_image = self._density_cache[1]

            if not density_image.isNull():
                painter.drawImage(
                    self.margins["left"], self.margins["top"], density_image
                )

        if (
            self._trendline != "none"
            and len(self.data) >= 2
            and self._xs_np is not None
            and len(self._xs_np) >= 2
        ):  # Trendline needs at least 2 points
            self._draw_trendlines(painter, min_x, max_x)

        if (
            (self._marginal_x or self._marginal_y)
            and self._xs_np is not None
            and len(self._xs_np) > 0
        ):
            self._draw_marginals(painter, plot_w, plot_h, min_x, max_x, min_y, max_y)

    def _draw_trendlines(
        self, painter, min_x_range, max_x_range
    ):  # Pass current view's x-range
        painter.setRenderHint(QPainter.Antialiasing, True)  # Enable for lines

        # Check if data for trendlines is available
        if self._xs_np is None or self._ys_np is None or len(self._xs_np) < 2:
            return

        if (
            self._trendline == "global" or self._groups_np is None or not self._color_by
        ):  # If no grouping or global trendline
            self._draw_single_trendline(
                painter,
                self._xs_np,
                self._ys_np,
                QColor(Qt.black),
                min_x_range,
                max_x_range,
            )
        elif self._trendline == "per group" and self._color_by:
            for group_val, color in self._group_to_color.items():
                mask = self._groups_np == group_val
                if np.sum(mask) >= 2:  # Need at least 2 points for a line
                    self._draw_single_trendline(
                        painter,
                        self._xs_np[mask],
                        self._ys_np[mask],
                        color,
                        min_x_range,
                        max_x_range,
                    )

        painter.setRenderHint(
            QPainter.Antialiasing, False
        )  # Reset if other parts should not be antialiased

    def _draw_single_trendline(
        self, painter, xs_data, ys_data, color, min_x_draw, max_x_draw
    ):
        if len(xs_data) < 2:
            return  # Cannot fit line

        try:
            coeffs = np.polyfit(xs_data, ys_data, 1)
            poly_fn = np.poly1d(coeffs)
        except (np.linalg.LinAlgError, ValueError) as e:
            # print(f"Could not fit trendline: {e}")
            return

        _, _, overall_min_y_data, overall_max_y_data = self.get_data_range()

        # Calculate start and end points of the trendline within the current drawing x-range
        y_start_trend = poly_fn(min_x_draw)
        y_end_trend = poly_fn(max_x_draw)

        # This ensures that the scaled points don't go wildly off-screen vertically.
        y_start_clipped = max(
            overall_min_y_data, min(y_start_trend, overall_max_y_data)
        )
        y_end_clipped = max(overall_min_y_data, min(y_end_trend, overall_max_y_data))

        # Scale the (potentially adjusted) start and end points to pixel coordinates
        pixel_start_x, pixel_start_y = self.scale_point(
            (min_x_draw, y_start_clipped)
        )  # Use min_x_draw with clipped_y
        pixel_end_x, pixel_end_y = self.scale_point(
            (max_x_draw, y_end_clipped)
        )  # Use max_x_draw with clipped_y

        painter.setPen(QPen(color, 1))  # Trendline pen
        painter.drawLine(
            QPointF(pixel_start_x, pixel_start_y), QPointF(pixel_end_x, pixel_end_y)
        )

    def _draw_marginals(
        self,
        painter,
        plot_w,
        plot_h,
        min_x_range,
        max_x_range,
        min_y_range,
        max_y_range,
    ):
        # Ensure data is available
        if self._xs_np is None or self._ys_np is None or len(self._xs_np) == 0:
            return

        num_bins_marginal = 30

        # Filter data to only what's currently visible in the main plot's x and y range
        visible_mask = (
            (self._xs_np >= min_x_range)
            & (self._xs_np <= max_x_range)
            & (self._ys_np >= min_y_range)
            & (self._ys_np <= max_y_range)
        )

        visible_xs = self._xs_np[visible_mask]
        visible_ys = self._ys_np[visible_mask]

        # Max thickness for marginal plots (leaving a small gap from axes)
        max_x_marginal_height = self.margins["top"] - 4
        max_y_marginal_width = self.margins["right"] - 4

        painter.setBrush(QColor(100, 100, 100, 100))  # Semi-transparent gray
        painter.setPen(Qt.NoPen)

        #  X-Marginal Histogram (at the top)
        if self._marginal_x and len(visible_xs) > 0 and max_x_marginal_height > 0:
            counts, bin_edges_x = np.histogram(
                visible_xs, bins=num_bins_marginal, range=(min_x_range, max_x_range)
            )
            max_count_x = counts.max() if counts.size > 0 else 1.0
            if max_count_x == 0:
                max_count_x = 1.0

            for i in range(len(counts)):
                if counts[i] == 0:
                    continue

                # Data coordinates for the current X-bin
                x_bin_left_data = bin_edges_x[i]
                x_bin_right_data = bin_edges_x[i + 1]

                # Convert data coordinates to pixel coordinates for X
                pixel_x_for_bin_left, _ = self.scale_point(
                    (x_bin_left_data, min_y_range)
                )  # Use dummy Y for scaling X
                pixel_x_for_bin_right, _ = self.scale_point(
                    (x_bin_right_data, min_y_range)
                )

                bar_pixel_width = abs(pixel_x_for_bin_right - pixel_x_for_bin_left)
                if bar_pixel_width < 1:
                    bar_pixel_width = 1  # Ensure minimum width

                bar_pixel_height = (counts[i] / max_count_x) * max_x_marginal_height
                if bar_pixel_height < 1:
                    bar_pixel_height = 1  # Ensure minimum height

                x_pixel_pos = pixel_x_for_bin_left
                y_pixel_pos = (
                    self.margins["top"] - bar_pixel_height - 2
                )  # Position above main plot, with a 2px gap

                painter.drawRect(
                    QRectF(
                        x_pixel_pos, y_pixel_pos, bar_pixel_width - 1, bar_pixel_height
                    )
                )  # -1 for small inter-bar gap

        #  Y-Marginal Histogram (at the right)
        if self._marginal_y and len(visible_ys) > 0 and max_y_marginal_width > 0:
            counts, bin_edges_y = np.histogram(
                visible_ys, bins=num_bins_marginal, range=(min_y_range, max_y_range)
            )
            max_count_y = counts.max() if counts.size > 0 else 1.0
            if max_count_y == 0:
                max_count_y = 1.0

            for i in range(len(counts)):
                if counts[i] == 0:
                    continue

                # Data coordinates for the current Y-bin
                y_bin_bottom_data = bin_edges_y[i]  # Lower value in data space
                y_bin_top_data = bin_edges_y[i + 1]  # Higher value in data space

                # Convert data coordinates to pixel coordinates for Y
                # scale_point inverts Y: higher data Y -> lower pixel Y
                _, pixel_y_for_bin_bottom = self.scale_point(
                    (min_x_range, y_bin_bottom_data)
                )  # Use dummy X for scaling Y
                _, pixel_y_for_bin_top = self.scale_point((min_x_range, y_bin_top_data))

                # Pixel coordinates for the bar
                # bar_top_pixel_y is the one with the smaller numerical value (closer to screen top)
                bar_top_pixel_y = min(pixel_y_for_bin_bottom, pixel_y_for_bin_top)
                bar_bottom_pixel_y = max(pixel_y_for_bin_bottom, pixel_y_for_bin_top)

                bar_pixel_height = bar_bottom_pixel_y - bar_top_pixel_y
                if bar_pixel_height < 1:
                    bar_pixel_height = 1  # Ensure minimum height

                bar_pixel_width = (counts[i] / max_count_y) * max_y_marginal_width
                if bar_pixel_width < 1:
                    bar_pixel_width = 1  # Ensure minimum width

                x_pixel_pos = (
                    self.width() - self.margins["right"] + 2
                )  # Position to the right of main plot, with 2px gap
                y_pixel_pos = bar_top_pixel_y

                painter.drawRect(
                    QRectF(
                        x_pixel_pos, y_pixel_pos, bar_pixel_width, bar_pixel_height - 1
                    )
                )  # -1 for small inter-bar gap

    def draw_gate_construction_visuals(self, painter):
        if (
            self.interaction_mode == InteractionMode.RECT_GATE
            and self.current_selection_rect_pixels
        ):
            pen = QPen(QColor(0, 120, 215, 150), 1, Qt.DashLine)  # Blue dashed line
            painter.setPen(pen)
            brush = QBrush(QColor(0, 120, 215, 50))  # Light blue fill
            painter.setBrush(brush)
            painter.drawRect(self.current_selection_rect_pixels)

        elif self.interaction_mode == InteractionMode.POLYGON_GATE:
            if (
                self.current_polygon_vertices_pixels
            ):  # If there are any vertices defined
                poly_pen_color = QColor(0, 150, 50, 200)  # Green for polygon
                poly_pen = QPen(poly_pen_color, 2, Qt.SolidLine)
                painter.setPen(poly_pen)

                poly_to_draw = QPolygonF(
                    [QPointF(p) for p in self.current_polygon_vertices_pixels]
                )
                painter.drawPolyline(poly_to_draw)

                vertex_brush_color = QColor(0, 150, 50, 150)
                painter.setBrush(vertex_brush_color)
                for i, vertex_px in enumerate(self.current_polygon_vertices_pixels):
                    radius = 3
                    current_pen = QPen(poly_pen_color, 1)  # Default pen for vertex
                    if (
                        i == 0
                        and self.is_near_start_vertex_for_closing
                        and len(self.current_polygon_vertices_pixels)
                        >= MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK - 1
                    ):
                        radius = 5  # Make it larger
                        current_pen.setColor(
                            poly_pen_color.lighter(150)
                        )  # Lighter color
                        current_pen.setWidth(2)
                    painter.setPen(current_pen)
                    painter.drawEllipse(QPointF(vertex_px), radius, radius)

                painter.setPen(poly_pen)
                if (
                    self.current_mouse_pos_pixels
                    and self.current_polygon_vertices_pixels
                ):
                    last_vertex_px = self.current_polygon_vertices_pixels[-1]

                    preview_target_pos = QPointF(self.current_mouse_pos_pixels)
                    # If near start vertex for closing, snap preview line to start vertex
                    if (
                        self.is_near_start_vertex_for_closing
                        and len(self.current_polygon_vertices_pixels)
                        >= MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK - 1
                    ):  # Need at least 2 points for a segment to close
                        preview_target_pos = QPointF(
                            self.current_polygon_vertices_pixels[0]
                        )

                    # Draw dashed line from last vertex to current mouse/snap point
                    dash_pen = QPen(poly_pen_color, 1, Qt.DashLine)
                    painter.setPen(dash_pen)
                    painter.drawLine(QPointF(last_vertex_px), preview_target_pos)

                    # Also draw dashed line from first vertex to current mouse if not closing
                    if len(self.current_polygon_vertices_pixels) > 1 and not (
                        self.is_near_start_vertex_for_closing
                        and len(self.current_polygon_vertices_pixels)
                        >= MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK - 1
                    ):
                        first_vertex_px = self.current_polygon_vertices_pixels[0]
                        painter.drawLine(
                            QPointF(first_vertex_px),
                            QPointF(self.current_mouse_pos_pixels),
                        )

    def _finalize_polygon_gate(self):
        if (
            len(self.current_polygon_vertices_pixels) >= 3
        ):  # Need at least 3 vertices for a polygon
            data_vertices = []
            for p_vertex_px in self.current_polygon_vertices_pixels:
                # Convert pixel coordinates back to data coordinates
                data_x, data_y = self.unscale_point(p_vertex_px.x(), p_vertex_px.y())
                data_vertices.append((data_x, data_y))

            # Ensure polygon is closed if it wasn't by snapping
            if self.is_near_start_vertex_for_closing or (
                data_vertices[0] != data_vertices[-1]
                and len(data_vertices) >= MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK - 1
            ):  # Ensure it's a valid polygon before closing
                # Check if already closed (e.g. by snap-click)
                if (
                    self.current_polygon_vertices_pixels[0]
                    != self.current_polygon_vertices_pixels[-1]
                ):  # Check pixel coords for exact match
                    if data_vertices[0] != data_vertices[-1]:  # Also check data coords
                        data_vertices.append(
                            data_vertices[0]
                        )  # Close the polygon in data space

            gate_name = f"PolyGate-{len(self.gates) + 1}"
            new_gate = PolygonGate(
                vertices=data_vertices,
                name=gate_name,
                parameters=(
                    self.x_parameter_name,
                    self.y_parameter_name,
                ),  # Set parameters for this chart
            )
            self.add_gate(new_gate)

        self.clear_temporary_selection_state()  # Reset drawing state

    def keyPressEvent(self, event):
        if self.interaction_mode == InteractionMode.POLYGON_GATE:
            if event.key() == Qt.Key_Escape:
                self.clear_temporary_selection_state()
                event.accept()
                return
            elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                if (
                    len(self.current_polygon_vertices_pixels) >= 3
                ):  # Finalize if enough points
                    self._finalize_polygon_gate()
                else:  # Not enough points, just clear
                    self.clear_temporary_selection_state()
                event.accept()
                return
        super().keyPressEvent(event)  # Pass to base if not handled

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Clamp click to chart area for gate definition
            clamped_pos = self._clamp_to_chart_area(event.pos())

            if self.interaction_mode == InteractionMode.RECT_GATE:
                self.start_point_pixels = clamped_pos  # Use clamped pos for start
                self.is_drawing_rect_gate = True
                # self.start_point_pixels = event.pos() # Original
                self.current_selection_rect_pixels = QRect(
                    self.start_point_pixels, self.start_point_pixels
                )
                self.update()

            elif self.interaction_mode == InteractionMode.POLYGON_GATE:
                # If near start vertex and enough points, finalize by closing
                if (
                    self.is_near_start_vertex_for_closing
                    and len(self.current_polygon_vertices_pixels)
                    >= MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK - 1
                ):  # Allow closing with 2 points + snap
                    # self.start_point_pixels = clamped_pos # Not used here in same way
                    # Ensure polygon is closed by adding first point if not already last
                    if (
                        self.current_polygon_vertices_pixels[0] != clamped_pos
                    ):  # Use clamped_pos for comparison
                        self.current_polygon_vertices_pixels.append(
                            self.current_polygon_vertices_pixels[0]
                        )
                    self._finalize_polygon_gate()
                else:
                    # Add new vertex
                    self.current_polygon_vertices_pixels.append(
                        clamped_pos
                    )  # Add clamped point
                    # self.current_polygon_vertices_pixels.append(event.pos()) # Original
                    self.current_mouse_pos_pixels = (
                        event.pos()
                    )  # Keep original mouse for visual feedback
                self.update()
            # else: # Other modes, or no specific action for this mode on press
            # self.select_gate(None) # Deselect gate if clicking for other modes (optional)
        # super().mousePressEvent(event) # If BaseChart has press handling

    def mouseMoveEvent(self, event):
        self.current_mouse_pos_pixels = (
            event.pos()
        )  # Store current mouse for polygon preview

        if (
            self.interaction_mode == InteractionMode.RECT_GATE
            and self.is_drawing_rect_gate
        ):
            if (
                hasattr(self, "start_point_pixels") and self.start_point_pixels
            ):  # Check if dragging started
                # Clamp the end point of the rectangle to the chart area
                end_pt = self._clamp_to_chart_area(event.pos())
                self.current_selection_rect_pixels = QRect(
                    self.start_point_pixels, end_pt
                ).normalized()
            self.update()

        elif self.interaction_mode == InteractionMode.POLYGON_GATE:
            if self.current_polygon_vertices_pixels:  # If polygon drawing has started
                # Check if mouse is near the first vertex to indicate closing possibility
                first_vertex_px = self.current_polygon_vertices_pixels[0]
                # Use manhattanLength for simple distance check, or QPointF.toPoint().manhattanLength() if QPointF
                dist_to_start = QPointF(
                    event.pos() - first_vertex_px
                ).manhattanLength()  # event.pos() is QPoint

                near_start = (
                    dist_to_start < SNAP_DISTANCE_PIXELS
                    and len(self.current_polygon_vertices_pixels)
                    >= MIN_POLYGON_VERTICES_FOR_CLOSE_CLICK - 1
                )  # Need at least 2 points to close to 3rd

                if near_start != self.is_near_start_vertex_for_closing:
                    self.is_near_start_vertex_for_closing = near_start
                    self.setCursor(
                        Qt.PointingHandCursor if near_start else Qt.ArrowCursor
                    )
                self.update()
        else:
            if self.is_near_start_vertex_for_closing:
                self.is_near_start_vertex_for_closing = False  # Reset flag
                self.unsetCursor()  # Revert to default cursor

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.interaction_mode == InteractionMode.RECT_GATE
            and self.is_drawing_rect_gate
        ):
            self.is_drawing_rect_gate = False  # Stop drawing mode

            if self.current_selection_rect_pixels and (
                self.current_selection_rect_pixels.width() > 3
                or self.current_selection_rect_pixels.height() > 3
            ):
                # Clamp rect to chart area before unscaling
                rect_px = QRect(
                    self._clamp_to_chart_area(
                        self.current_selection_rect_pixels.topLeft()
                    ),
                    self._clamp_to_chart_area(
                        self.current_selection_rect_pixels.bottomRight()
                    ),
                ).normalized()

                # Convert pixel coordinates back to data coordinates
                data_x1, data_y1 = self.unscale_point(rect_px.left(), rect_px.top())
                data_x2, data_y2 = self.unscale_point(rect_px.right(), rect_px.bottom())

                # Create RectangularGate (x,y,w,h in data coordinates)
                gate_x = min(data_x1, data_x2)
                gate_y = min(
                    data_y1, data_y2
                )  # Note: Y-axis is often inverted in data vs screen pixel
                gate_w = abs(data_x1 - data_x2)
                gate_h = abs(data_y1 - data_y2)

                gate_name = f"RectGate-{len(self.gates) + 1}"
                new_gate = RectangularGate(
                    x=gate_x,
                    y=gate_y,
                    width=gate_w,
                    height=gate_h,
                    name=gate_name,
                    parameters=(
                        self.x_parameter_name,
                        self.y_parameter_name,
                    ),  # From BaseChart
                )
                self.add_gate(new_gate)  # BaseChart method to add and emit

            self.current_selection_rect_pixels = None  # Clear temporary rectangle
            if hasattr(self, "start_point_pixels"):
                del self.start_point_pixels  # Clean up start point
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.interaction_mode == InteractionMode.POLYGON_GATE:
                # Finalize polygon on double click if enough vertices
                if len(self.current_polygon_vertices_pixels) >= 3:
                    self._finalize_polygon_gate()
                else:  # Not enough points, clear current attempt
                    self.clear_temporary_selection_state()
