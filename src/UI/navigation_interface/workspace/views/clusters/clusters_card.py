import logging
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDrag,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QVBoxLayout
from qfluentwidgets import BodyLabel, CardWidget, ImageLabel, isDarkTheme

from backend.config import (
    CLUSTERS_CARD_HEIGHT,
    CLUSTERS_CARD_IMAGE_HEIGHT,
    CLUSTERS_CARD_WIDTH,
)


class ClustersCard(CardWidget):
    """Clusters card with drag-and-drop functionality."""

    split_requested = Signal()
    merge_requested = Signal(
        list
    )  # Emitting the source and target cluster IDs to merge
    assign_class_requested = Signal(str)
    cluster_double_clicked = Signal(str)
    card_clicked = Signal(str, Qt.KeyboardModifiers, Qt.MouseButton)

    def __init__(self, preview_image_path: Optional[str], parent=None):
        super().__init__(parent)
        self.cluster_id = ""
        self.cluster_color = "#FFFFFF"  # Default color

        # Fix: Initialize ImageLabel with parent, then set the image
        self.iconWidget = ImageLabel(self)  # Pass parent only
        if preview_image_path and os.path.exists(preview_image_path):
            # Assuming ImageLabel has a method like setPixmap or setImageFromFile
            # Let's try setPixmap with a QPixmap created from the path
            pixmap = QPixmap(preview_image_path)
            if not pixmap.isNull():
                self.iconWidget.setPixmap(pixmap)
            else:
                logging.warning(
                    f"Could not load pixmap for cluster card: {preview_image_path}"
                )
                # Optionally set a default/error icon
                # self.iconWidget.setPixmap(DEFAULT_ICON_PATH)
        else:
            logging.debug(
                f"No preview image path provided or path invalid for cluster card: {preview_image_path}"
            )
            # Optionally set a default/placeholder icon
            # self.iconWidget.setPixmap(DEFAULT_ICON_PATH)

        # Use FluentIcon.PHOTO if path is None or invalid initially?
        # self.iconWidget = IconWidget(FluentIcon.PHOTO, self)

        self.iconWidget.setBorderRadius(4, 4, 4, 4)
        self.iconWidget.setFixedSize(150, 150)

        self.label = BodyLabel(
            Path(preview_image_path).stem if preview_image_path else "", self
        )
        self.clusters_presenter = None
        self.selected = False

        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setAlignment(Qt.AlignCenter)
        self.vBoxLayout.addStretch(1)
        self.vBoxLayout.addWidget(self.iconWidget, 0, Qt.AlignCenter)
        self.vBoxLayout.addStretch(1)
        self.vBoxLayout.addWidget(self.label, 0, Qt.AlignHCenter | Qt.AlignBottom)

        self.card_height = CLUSTERS_CARD_HEIGHT
        self.card_width = CLUSTERS_CARD_WIDTH

        self.setFixedSize(self.card_width, self.card_height)

        # Enable dragging and dropping
        self.setAcceptDrops(True)

        # Variables to track dragging state
        self.drag_start_position = None

    def _normalBackgroundColor(self):
        return QColor(255, 255, 255, 13 if isDarkTheme() else 170)

    def _hoverBackgroundColor(self):
        return QColor(237, 255, 245, 90 if isDarkTheme() else 64)

    def _pressedBackgroundColor(self):
        return QColor(255, 255, 255, 150 if isDarkTheme() else 64)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = self._borderRadius
        d = 2 * r

        isDark = isDarkTheme()
        topBorderColor = QColor(240, 240, 240, 60)
        path = QPainterPath()

        # Define colors
        base_color = QColor(50, 180, 165)  # Teal base color for selected state
        top_border_color = base_color.lighter(150)  # Slightly lighter for top border
        tracing_color = (
            base_color.darker(150) if isDark else base_color.darker(200)
        )  # Darker for tracing
        film_color = (
            QColor(0, 0, 0, 0) if isDark else QColor(0, 0, 0, 0)
        )  # Translucent white or black film

        if self.selected:
            # Draw rounded rectangle with film
            painter.setBrush(film_color)  # Translucent film
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect(), r, r)

            # Draw tracing (inner border)
            path.addRoundedRect(
                4, 4, w - 8, h - 8, r - 2, r - 2
            )  # More inset and slightly smaller radius
            painter.strokePath(path, QPen(tracing_color, 6, Qt.DashLine))  # Dashed line
        else:
            # Draw background and border
            if isDark:
                if self.isPressed:
                    topBorderColor = QColor(255, 255, 255, 18)
                elif self.isHover:
                    topBorderColor = QColor(255, 255, 255, 13)
            else:
                if self.selected:
                    topBorderColor = QColor(50, 180, 165, 230)
                else:
                    if self.cluster_color:  # Use cluster color if available
                        topBorderColor = QColor(self.cluster_color)
                    else:
                        topBorderColor = QColor(240, 240, 240, 60)

            painter.strokePath(path, topBorderColor)

            pen = QPen(topBorderColor)
            pen.setWidth(6)
            painter.setPen(pen)
            rect = self.rect().adjusted(1, 1, -1, -1)
            painter.setBrush(self.backgroundColor)
            painter.drawRoundedRect(rect, r, r)

            # Draw top border
            path = QPainterPath()
            path.arcMoveTo(1, h - d - 1, d, d, 240)
            path.arcTo(1, h - d - 1, d, d, 225, -60)
            path.lineTo(1, r)
            path.arcTo(1, 1, d, d, -180, -90)
            path.lineTo(w - r, 1)
            path.arcTo(w - d - 1, 1, d, d, 90, -90)
            path.lineTo(w - 1, h - r)
            path.arcTo(w - d - 1, h - d - 1, d, d, 0, -60)

        # Draw bottom border
        bottomBorderColor = topBorderColor
        if not isDark and self.isHover and not self.isPressed:
            bottomBorderColor = QColor(0, 0, 0, 27)

        painter.strokePath(path, bottomBorderColor)

    def mousePressEvent(self, event: QMouseEvent):
        """Store the position where the mouse is pressed."""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Initiate the drag operation if the mouse is moved sufficiently."""
        if event.buttons() & Qt.LeftButton:
            if self.drag_start_position:
                distance = (
                    event.position().toPoint() - self.drag_start_position
                ).manhattanLength()
                if distance >= QApplication.startDragDistance():
                    self.start_drag()
                    self.drag_start_position = None
        super().mouseMoveEvent(event)

    def start_drag(self):
        """Start the drag operation with the cluster ID."""
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(
            self.cluster_id
        )  # You can use a more structured format if needed
        drag.setMimeData(mime_data)

        # Optional: Set drag pixmap for better UX
        pixmap = self.grab()
        drag.setPixmap(
            pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        drag.setHotSpot(pixmap.rect().center())

        drop_action = drag.exec(Qt.MoveAction)
        if drop_action == Qt.MoveAction:
            logging.debug(f"Cluster {self.cluster_id} dragged successfully.")

    def dragEnterEvent(self, event):
        """Accept the drag if it contains a cluster ID and is not the same as this cluster."""
        if event.mimeData().hasText():
            source_cluster_id = event.mimeData().text()
            if self.cluster_id != source_cluster_id:
                event.acceptProposedAction()
                self.set_selected(True)  # Highlight as valid drop target
            else:
                # Do not accept drag from self to prevent self-merging
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle the drag moving over the widget."""
        if event.mimeData().hasText():
            source_cluster_id = event.mimeData().text()
            if self.cluster_id != source_cluster_id:
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle the drop event to merge clusters."""
        source_cluster_id = event.mimeData().text()
        target_cluster_id = self.cluster_id
        logging.debug(
            f"Drop detected: source={source_cluster_id}, target={target_cluster_id}"
        )
        if source_cluster_id != target_cluster_id:
            logging.debug(
                f"Merging Cluster {source_cluster_id} into Cluster {target_cluster_id}."
            )
            self.merge_requested.emit(
                [source_cluster_id, target_cluster_id]
            )  # Emit the signal with both IDs
            event.acceptProposedAction()
        else:
            event.ignore()

        self.set_selected(False)  # Remove highlight after drop

    def dragLeaveEvent(self, event):
        """Handle the drag leaving the widget."""
        self.set_selected(False)
        super().dragLeaveEvent(event)

    def set_selected(self, selected: bool):
        """Set the selection state for visual feedback."""
        self.selected = selected
        self.update()

    def mouseReleaseEvent(self, e):
        """Handle mouse release events for selection."""
        super().mouseReleaseEvent(e)
        self.card_clicked.emit(self.cluster_id, e.modifiers(), e.button())

    def contextMenuEvent(self, event):
        """Handles the right-click context menu."""
        self.clusters_presenter.context_menu_handler.show_context_menu(self, event)

    def on_split_requested(self):
        """Handles the split cluster action."""
        self.split_requested.emit()

    def on_merge_requested(self):
        """Handles the merge clusters action."""
        self.merge_requested.emit()

    def on_assign_class_requested(self, class_name):
        """Handles the assign class action."""
        self.assign_class_requested.emit(class_name)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handles double-click events."""
        if event.button() == Qt.LeftButton:
            self.cluster_double_clicked.emit(self.cluster_id)
