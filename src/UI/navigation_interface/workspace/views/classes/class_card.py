# UI\navigation_interface\workspace\views\classes\class_card.py
import logging
import random
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QContextMenuEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QMenu, QVBoxLayout
from qfluentwidgets import CaptionLabel, CardWidget, ImageLabel, isDarkTheme


class ClassCard(CardWidget):
    """Class card that displays a pre-rendered image."""

    class_double_clicked = Signal(str)

    def __init__(self, iconPath: str, classes_presenter, class_id: str, parent=None):
        self.iconWidget = None
        super().__init__(parent)

        self._borderRadius = 0
        self.classes_presenter = classes_presenter
        self.class_id = class_id
        self.is_hovered = False

        self.class_name = "Loading..."
        self.class_color_hex = "#FFFFFF"
        if self.classes_presenter and self.classes_presenter.data_manager:
            class_obj = self.classes_presenter.data_manager.get_class(class_id)
            if class_obj:
                self.class_name = class_obj.name
                self.class_color_hex = class_obj.color

        self.iconWidget = ImageLabel(parent=self)
        self.label = CaptionLabel(self.class_name, self)
        self.label.setAlignment(Qt.AlignCenter)

        self.mainWidgetLayout = QVBoxLayout(self)
        self.mainWidgetLayout.setSpacing(3)
        self.mainWidgetLayout.setAlignment(Qt.AlignCenter)
        self.mainWidgetLayout.addWidget(
            self.iconWidget, 0, Qt.AlignCenter
        )  # iconWidget will be primary size driver
        self.mainWidgetLayout.addWidget(self.label, 0, Qt.AlignCenter)
        pad = 10
        self.mainWidgetLayout.setContentsMargins(pad, pad, pad, 5 + pad)
        self._original_icon_size = QSize(135, 135)  # Default if iconPath is bad

        if iconPath:
            pixmap = QPixmap(iconPath)
            if not pixmap.isNull():
                self.iconWidget.setPixmap(pixmap)
                self._original_icon_size = pixmap.size()
                self.iconWidget.setMinimumSize(self._original_icon_size)
                # self.setFixedSize(self.mainWidgetLayout.sizeHint()) # Set ClassCard size based on content
                self.adjustSize()  # Let ClassCard determine its size from layout
            else:
                logging.warning(f"ClassCard: Could not load pixmap from {iconPath}")
                self.iconWidget.setMinimumSize(self._original_icon_size)
                self.adjustSize()
        else:
            logging.warning(f"ClassCard: No iconPath provided for class {class_id}")
            self.iconWidget.setMinimumSize(self._original_icon_size)
            self.adjustSize()

        if self.classes_presenter:
            self.class_double_clicked.connect(self.classes_presenter.show_class_viewer)

        self.setMouseTracking(True)
        self._size_animation = QPropertyAnimation(self.iconWidget, b"size", self)
        self._size_animation.setDuration(200)  # ms
        self._size_animation.setEasingCurve(QEasingCurve.OutQuad)  # Smoother curve
        self._size_animation.valueChanged.connect(
            self._on_animation_value_changed
        )  # Update during animation
        self._size_animation.finished.connect(self._on_animation_finished)

    def _normalBackgroundColor(self):
        return QColor(0, 0, 0, 0)

    def _hoverBackgroundColor(self):
        return QColor(0, 0, 0, 0)

    def _pressedBackgroundColor(self):
        return QColor(0, 0, 0, 0)

    def paintEvent(self, e):
        pass

    def enterEvent(self, event):
        super().enterEvent(event)
        if not self.is_hovered:
            self.is_hovered = True
            self.start_hover_animation(enlarge=True)  # grows & lifts the maxSize

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self.is_hovered:
            self.is_hovered = False
            self.start_hover_animation(enlarge=False)  # shrinks & resets maxSize

    def start_hover_animation(self, enlarge: bool):
        """Animate the icon size smoothly in and out."""
        if not self.iconWidget:
            return

        base = self._original_icon_size
        factor = 1.025 if enlarge else 1.0
        target = QSize(int(base.width() * factor), int(base.height() * factor))

        # ❶ Always lift the ceiling *to the current target*,
        #    never clamp it before the animation runs.
        self.iconWidget.setMaximumSize(target)

        self._size_animation.stop()
        self._size_animation.setStartValue(self.iconWidget.size())
        self._size_animation.setEndValue(target)
        self._size_animation.start()

    def _on_animation_finished(self):
        """Called when the size animation ends."""
        # ❷ If the mouse is no longer inside, restore the tight maximum
        if not self.is_hovered:
            self.iconWidget.setMaximumSize(self._original_icon_size)

        self.adjustSize()
        if p := self.parentWidget():
            if p.layout():
                p.layout().activate()

    def _on_animation_value_changed(self):
        """Called when the animation value (size) changes."""
        # When iconWidget's size changes, ClassCard needs to adjust its own size.
        self.adjustSize()
        # Also, tell the parent layout that this card's geometry might need updating.
        if self.parentWidget() and self.parentWidget().layout():
            self.parentWidget().layout().update()  # Request an update of the layout

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu {background-color: #234f4b; color: white;}")
        show_summary_action = menu.addAction("Show Summary")
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")

        if self.classes_presenter:
            show_summary_action.triggered.connect(
                lambda: self.classes_presenter.show_summary(self.class_id)
            )
            rename_action.triggered.connect(
                lambda: self.classes_presenter.handle_rename_class(self.class_id)
            )
            delete_action.triggered.connect(
                lambda: self.classes_presenter.request_delete_class(self.class_id)
            )

        menu.exec_(event.globalPos())

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.class_double_clicked.emit(self.class_id)

    def update_display(self, new_icon_path: str, new_name: str, new_color_hex: str):
        self.class_name = new_name
        self.label.setText(self.class_name)
        self.class_color_hex = new_color_hex

        if new_icon_path:
            pixmap = QPixmap(new_icon_path)
            if not pixmap.isNull():
                self._original_icon_size = pixmap.size()
                self.iconWidget.setPixmap(pixmap)
                self.iconWidget.setFixedSize(self._original_icon_size)
                self.adjustSize()
            else:
                logging.warning(
                    f"ClassCard Update: Could not load new pixmap from {new_icon_path}"
                )
        else:
            logging.warning(
                f"ClassCard Update: No new iconPath provided for class {self.class_id}"
            )
        self.update()
