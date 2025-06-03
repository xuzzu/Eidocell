from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ChartFrame(QFrame):
    # aspect_ratio_chart_area removed from __init__ as BaseChart now handles its own aspect
    def __init__(self, chart_widget: QWidget, parent=None):
        super().__init__(parent)
        self.chart_widget = chart_widget  # This is a BaseChart instance
        self.control_widget_ref = None

        self._internal_layout = QVBoxLayout(self)
        self._internal_layout.setContentsMargins(
            8, 8, 8, 8
        )  # Slightly more padding inside frame
        self._internal_layout.setSpacing(8)  # More spacing between chart and controls

        # Chart widget's size policy (Expanding, Preferred + H4W) will drive its height
        self._internal_layout.addWidget(
            self.chart_widget, 1
        )  # Chart takes expanding space

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            ChartFrame {
                background-color: white;
                border-radius: 8px;
            }
        """
        )

        # ChartFrame's size policy: Expanding horizontally, Preferred vertically
        # Its height will be determined by the sum of its children's (chart + controls) preferred heights.
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # ChartFrame itself doesn't need heightForWidth if its children manage their aspect/height.
        # However, if we want the ChartFrame to have a say for the *entire* block (chart + controls),
        # then it would need heightForWidth. For now, let BaseChart drive its part.
        self.setSizePolicy(sp)
        self.setMinimumSize(280, 200)  # Overall minimum for frame

    def set_control_widget(self, control_widget: QWidget):
        if self.control_widget_ref and self.control_widget_ref is control_widget:
            return
        if self.control_widget_ref:
            self._internal_layout.removeWidget(self.control_widget_ref)
            self.control_widget_ref.setParent(None)
        self.control_widget_ref = control_widget
        if self.control_widget_ref:
            self._internal_layout.addWidget(self.control_widget_ref, 0)
        self.updateGeometry()  # Crucial: tell layout to re-evaluate after adding/removing widget

    # sizeHint and minimumSizeHint for ChartFrame should now sum up children's hints
    def sizeHint(self):
        width = 0
        height = 0

        if self.chart_widget:
            chart_hint = self.chart_widget.sizeHint()
            width = max(width, chart_hint.width())
            height += chart_hint.height()

        if self.control_widget_ref:
            ctrl_hint = self.control_widget_ref.sizeHint()
            width = max(width, ctrl_hint.width())
            height += ctrl_hint.height()
            if self.chart_widget:  # Add spacing if both exist
                height += self._internal_layout.spacing()

        margins = self.layout().contentsMargins()
        width += margins.left() + margins.right()
        height += margins.top() + margins.bottom()

        return QSize(
            max(self.minimumSize().width(), width),
            max(self.minimumSize().height(), height),
        )

    def minimumSizeHint(self):
        width = 0
        height = 0

        if self.chart_widget:
            chart_min_hint = self.chart_widget.minimumSizeHint()
            width = max(width, chart_min_hint.width())
            height += chart_min_hint.height()

        if self.control_widget_ref:
            ctrl_min_hint = self.control_widget_ref.minimumSizeHint()
            width = max(width, ctrl_min_hint.width())
            height += ctrl_min_hint.height()
            if self.chart_widget:
                height += self._internal_layout.spacing()

        margins = self.layout().contentsMargins()
        width += margins.left() + margins.right()
        height += margins.top() + margins.bottom()

        return QSize(
            max(self.minimumSize().width(), width),
            max(self.minimumSize().height(), height),
        )
