import random
import uuid

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPolygonF


def _random_bright_colour():
    """Return a random QColor with decent brightness and saturation."""
    h = random.randint(0, 359)  # full hue wheel
    s = random.randint(140, 220)  # avoid very pale colours
    l = random.randint(120, 180)  # avoid too dark
    c = QColor()
    c.setHsl(h, s, l)
    return c


class BaseGate:
    """
    Base class for all gate types.
    """

    def __init__(self, name=None, color=None, parameters=None):
        self.id = uuid.uuid4()  # Unique identifier for the gate
        self.name = name if name else f"Gate {str(self.id)[:4]}"
        self.color = color if color else _random_bright_colour()
        self.parameters = (
            parameters if parameters else tuple()
        )  # e.g., ('FSC-A', 'SSC-A') or ('FSC-A',)
        self.ids_in_gate = []  # List of data point IDs that fall within this gate
        self.event_count = 0
        self.percentage_of_total = 0.0
        self.percentage_of_parent = 0.0  # If part of hierarchical gating

    def is_point_inside(self, *args):
        """
        Abstract method to check if a point is inside the gate.
        The number of arguments will depend on the gate type (e.g., x, y for 2D; value for 1D).
        """
        raise NotImplementedError("Subclasses must implement is_point_inside")

    def update_contained_ids(
        self,
        all_data_points,
        x_param_idx=0,
        y_param_idx=1,
        id_param_idx=None,
        value_param_idx=0,
    ):
        """
        Updates the list of IDs contained within this gate based on the full dataset.
        This is a generic updater; specific charts might optimize this.

        :param all_data_points: List of data points. Structure depends on the chart.
                                e.g., for Scatter: [(x, y, size, color, id), ...]
                                e.g., for Histogram: [(value, id), ...]
        :param x_param_idx: Index of X value in a data point (for 2D gates).
        :param y_param_idx: Index of Y value in a data point (for 2D gates).
        :param id_param_idx: Index of ID in a data point. If None, assumes no specific ID to store.
        :param value_param_idx: Index of value in a data point (for 1D gates).
        """
        self.ids_in_gate = []
        if not all_data_points:
            self.event_count = 0
            return

        for i, data_point in enumerate(all_data_points):
            try:
                point_id = (
                    data_point[id_param_idx]
                    if id_param_idx is not None and len(data_point) > id_param_idx
                    else i
                )

                if isinstance(self, (RectangularGate, PolygonGate)):  # 2D Gates
                    if len(data_point) > max(x_param_idx, y_param_idx):
                        x_val, y_val = data_point[x_param_idx], data_point[y_param_idx]
                        if self.is_point_inside(x_val, y_val):
                            self.ids_in_gate.append(point_id)
                elif isinstance(self, IntervalGate):  # 1D Gates
                    if len(data_point) > value_param_idx:
                        value = data_point[value_param_idx]
                        if self.is_point_inside(value):
                            self.ids_in_gate.append(point_id)
            except IndexError:
                print(f"Warning: Data point {i} has unexpected structure: {data_point}")
                continue
            except TypeError as e:
                print(
                    f"Warning: Type error processing data point {i} ({data_point}): {e}"
                )
                continue

        self.event_count = len(self.ids_in_gate)
        # Percentage calculations would typically be done by a gate manager
        # or the chart itself, as it knows the total/parent population sizes.

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' ({self.event_count} events)>"

    def draw(self, painter, chart_scaler_func, chart_unscaler_func):
        """
        Abstract method for drawing the gate on a QPainter.
        :param painter: QPainter instance.
        :param chart_scaler_func: A function from the chart to scale data coords to pixel coords.
        :param chart_unscaler_func: A function from the chart to unscale pixel coords to data coords.
        """
        raise NotImplementedError("Subclasses must implement draw")


class RectangularGate(BaseGate):
    def __init__(self, x, y, width, height, name=None, color=None, parameters=None):
        super().__init__(name, color, parameters)
        self.rect = QRectF(x, y, width, height)  # Stored in data coordinates

    def is_point_inside(self, x_val, y_val):
        return self.rect.contains(QPointF(x_val, y_val))

    def draw(self, painter, chart_scaler_func):
        painter.setPen(self.color)
        painter.setBrush(
            QColor(self.color.red(), self.color.green(), self.color.blue(), 30)
        )  # Semi-transparent fill

        # Scale gate's data coordinates to pixel coordinates for drawing
        # Top-left corner
        px1, py1 = chart_scaler_func((self.rect.left(), self.rect.top()))
        # Bottom-right corner
        px2, py2 = chart_scaler_func((self.rect.right(), self.rect.bottom()))

        # QPainter draws rectangles from top-left. Ensure correct y-coordinates due to inversion.
        # Scaler function handles y-inversion. px1,py1 is top-left in pixel space.
        # px2,py2 is bottom-right in pixel space.
        # However, if py1 > py2 (which it will be if top < bottom in data space),
        # QRectF needs (left, top, width, height) where top is the *smaller* y pixel value.

        pixel_rect = QRectF(QPointF(px1, py1), QPointF(px2, py2)).normalized()
        painter.drawRect(pixel_rect)

        # Optionally draw name/stats
        painter.drawText(
            pixel_rect.topLeft() - QPointF(0, 5), f"{self.name} ({self.event_count})"
        )


class PolygonGate(BaseGate):
    def __init__(self, vertices, name=None, color=None, parameters=None):
        # vertices: list of QPointF or list of (x,y) tuples in data coordinates
        super().__init__(name, color, parameters)

        # Convert vertices to QPointF if they are not already
        qpointf_vertices = []
        if vertices:
            for v in vertices:
                if isinstance(v, QPointF):
                    qpointf_vertices.append(v)
                elif isinstance(v, (tuple, list)) and len(v) == 2:
                    qpointf_vertices.append(QPointF(v[0], v[1]))
                else:
                    raise TypeError(f"Invalid vertex type in PolygonGate: {v}")

        self.polygon = QPolygonF(qpointf_vertices)

    def is_point_inside(self, x_val, y_val):
        # Ray Casting Algorithm or QPolygonF.containsPoint
        # QPolygonF.containsPoint requires Qt.FillRule (OddEvenFill is Ray Casting)
        return self.polygon.containsPoint(QPointF(x_val, y_val), Qt.OddEvenFill)

    def draw(self, painter, chart_scaler_func):
        painter.setPen(self.color)
        painter.setBrush(
            QColor(self.color.red(), self.color.green(), self.color.blue(), 30)
        )

        scaled_vertices = [
            QPointF(*chart_scaler_func(v.toTuple())) for v in self.polygon
        ]

        painter.drawPolygon(QPolygonF(scaled_vertices))

        # Optionally draw name/stats near the first vertex or centroid
        if scaled_vertices:
            painter.drawText(
                scaled_vertices[0] - QPointF(0, 5), f"{self.name} ({self.event_count})"
            )


class IntervalGate(BaseGate):
    def __init__(self, min_val, max_val, name=None, color=None, parameter_name=None):
        # parameter_name: string, e.g., 'FSC-A'
        super().__init__(
            name, color, parameters=(parameter_name,) if parameter_name else tuple()
        )
        self.min_val = min_val
        self.max_val = max_val

    def is_point_inside(self, value):
        return self.min_val <= value <= self.max_val

    def draw(self, painter, chart_scaler_func, chart_height, margins):
        """
        Draws the interval gate on a histogram-like chart.
        :param chart_scaler_func: Scales (value, y_coord_for_scaling) -> (pixel_x, pixel_y)
        :param chart_height: Total height of the chart widget.
        :param margins: Chart margins dictionary.
        """
        painter.setPen(self.color)
        painter.setBrush(
            QColor(self.color.red(), self.color.green(), self.color.blue(), 30)
        )

        # We need a dummy y-value for scaling the x-position of the interval
        # The y-position of the gate rectangle will span the chart area.
        # Let's use 0 for the dummy y-value for scaling.
        # The actual y-pixel coordinates for the rectangle will be from chart top to bottom margin.

        # X-pixel coordinates for the start and end of the interval
        # Using a dummy y value (e.g. 0) for scaling x. scale_point should handle it.
        # (This assumes scale_point can take a 1D value for a 1D chart, or it needs adaptation)
        # For now, let's assume scale_point takes (value, dummy_y_for_range)
        # Or, we need a dedicated 1D scaler if the chart is purely 1D.
        # BaseChart.scale_point is 2D. We'll use it with a fixed y (e.g., min_y of data range).

        # Get min_y from the chart's data range to pass to scale_point
        # This is a bit of a hack; ideally, a 1D chart would have a 1D scaler.
        # For now, we assume the chart using this gate can provide its min_y.
        # This will be called from HistogramChart, which can provide its data range.
        # We'll pass min_data_y to the draw method from the chart.

        # This part needs to be called from HistogramChart's draw_chart or a new draw_gates method
        # where min_data_y can be easily accessed.
        # For now, the drawing logic for IntervalGate will be more specific in HistogramChart.draw_gates
        # This base draw method here is more of a placeholder for IntervalGate.

        # The actual drawing will be implemented in HistogramChart's draw_gates method
        # as it needs specific context like bar width, y-scaling of frequencies etc.
        # This base draw method is illustrative.

        # Example: Highlight region
        # (px_min, _) = chart_scaler_func((self.min_val, 0)) # Using 0 as dummy y
        # (px_max, _) = chart_scaler_func((self.max_val, 0)) # Using 0 as dummy y
        # top_y = margins["top"]
        # bottom_y = chart_height - margins["bottom"]
        # painter.drawRect(QRectF(px_min, top_y, px_max - px_min, bottom_y - top_y))
        # painter.drawText(QPointF(px_min, top_y - 5), f"{self.name} ({self.event_count})")
        pass  # Actual drawing to be handled by the chart for IntervalGate
