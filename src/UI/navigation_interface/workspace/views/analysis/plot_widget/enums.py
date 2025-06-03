# enums.py
from enum import Enum, auto


class InteractionMode(Enum):
    """
    Defines the interaction mode for charts.
    """

    SELECT_POINTS = auto()  # For selecting individual data points (not creating a gate)
    RECT_GATE = auto()  # For drawing rectangular gates
    POLYGON_GATE = auto()  # For drawing polygon gates
    # Future modes:
    # EDIT_GATE = auto()    # For moving/resizing existing gates
    # PAN_ZOOM = auto()     # For panning and zooming the chart view
