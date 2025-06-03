# src/UI/navigation_interface/workspace/views/segmentation/interactive_segmentation_canvas.py

import logging
import uuid
from enum import Enum, auto
from typing import Any, Dict, List

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QGuiApplication,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

logger = logging.getLogger(__name__)


class CanvasInteractionMode(Enum):
    PAN = auto()
    POINT_PROMPT = auto()
    ERASER = auto()
    BOX_PROMPT = auto()
    # BRUSH_EDIT = auto()


class InteractiveSegmentationView(QGraphicsView):
    prompt_added = Signal(str, QPointF, bool)
    prompts_erased = Signal(list)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMouseTracking(True)

        self.current_interaction_mode = CanvasInteractionMode.PAN
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self._dot_radius_screen = 5
        self._eraser_radius_screen = 10

        self._prompt_data_list = []
        self._is_erasing_on_move = False

        self.positive_prompt_color = QColor(0, 200, 0, 220)
        self.negative_prompt_color = QColor(220, 0, 0, 220)
        self.eraser_cursor_color = QColor(128, 128, 128, 100)

        self.setBackgroundBrush(QColor(230, 230, 230))
        self.set_image(
            QPixmap(
                r"E:\ANDREY\PROJECTS\dino-pretrain\data\datasets\plain\IFCB\padded_75827.png"
            )
        )  # Initialize with empty pixmap

    def set_interaction_mode(self, mode: CanvasInteractionMode):
        if self.current_interaction_mode == mode:
            return
        logger.debug(
            f"Canvas mode changed from {self.current_interaction_mode.name} to {mode.name}"
        )
        self.current_interaction_mode = mode
        if self.current_interaction_mode == CanvasInteractionMode.PAN:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(
                Qt.OpenHandCursor
            )  # Changed to OpenHandCursor for PAN
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            if self.current_interaction_mode == CanvasInteractionMode.ERASER:
                self.viewport().setCursor(
                    Qt.BlankCursor
                )  # Hide system cursor, we draw our own
            elif self.current_interaction_mode == CanvasInteractionMode.POINT_PROMPT:
                self.viewport().setCursor(Qt.CrossCursor)
            else:
                self.viewport().setCursor(Qt.ArrowCursor)
        self.viewport().update()  # Request viewport repaint for cursor change

    def set_image(self, pixmap: QPixmap):
        self.clear_all_prompts()

        if pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            logger.warning("InteractiveSegmentationView: Null pixmap set.")
            return

        self._pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.view_changed.emit()

    def current_zoom_factor(self) -> float:
        return self.transform().m11()

    def _clear_prompt_visuals_from_scene(self):  # Renamed to be more specific
        """Removes only the visual QGraphicsEllipseItem objects from the scene."""
        for entry in self._prompt_data_list:
            visual_item = entry.get("item")
            if visual_item and visual_item.scene() == self._scene:
                self._scene.removeItem(visual_item)

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        zoom_val = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        new_transform_m11 = self.transform().m11() * zoom_val
        if new_transform_m11 < 0.05 or new_transform_m11 > 100.0:
            return

        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(zoom_val, zoom_val)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self.view_changed.emit()
        self.viewport().update()  # For eraser cursor redraw

    def mousePressEvent(self, event: QMouseEvent):
        if self.current_interaction_mode == CanvasInteractionMode.POINT_PROMPT:
            if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
                scene_pos = self.mapToScene(event.pos())
                if (
                    not self._pixmap_item.pixmap().isNull()
                    and self._pixmap_item.sceneBoundingRect().contains(scene_pos)
                ):
                    image_pos = self._pixmap_item.mapFromScene(scene_pos)
                    if self._pixmap_item.pixmap().rect().contains(image_pos.toPoint()):
                        is_positive = event.button() == Qt.LeftButton
                        prompt_id = str(uuid.uuid4())

                        logger.info(
                            f"Point Prompt: ID={prompt_id}, ImgCoords=({image_pos.x():.1f},{image_pos.y():.1f}), Positive={is_positive}"
                        )
                        self.prompt_added.emit(prompt_id, image_pos, is_positive)
                        self._add_prompt_visual_and_data(
                            prompt_id, image_pos, is_positive
                        )  # Updated method
                        event.accept()
                        return  # Consume event
            # Fall through to super if not accepted (e.g. not on image) or not correct button
            super().mousePressEvent(event)

        elif self.current_interaction_mode == CanvasInteractionMode.ERASER:
            if event.button() == Qt.LeftButton:
                self._is_erasing_on_move = True
                self._erase_prompts_at_view_pos(event.pos())
                event.accept()
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.current_interaction_mode == CanvasInteractionMode.ERASER:
            if self._is_erasing_on_move and (event.buttons() & Qt.LeftButton):
                self._erase_prompts_at_view_pos(event.pos())
            self.viewport().update()
            event.accept()
        else:
            if self.current_interaction_mode == CanvasInteractionMode.PAN and (
                event.buttons() & Qt.LeftButton
            ):
                self.view_changed.emit()
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.current_interaction_mode == CanvasInteractionMode.ERASER:
            if event.button() == Qt.LeftButton:
                self._is_erasing_on_move = False
                event.accept()
            else:
                super().mouseReleaseEvent(event)
        else:
            if (
                self.current_interaction_mode == CanvasInteractionMode.PAN
            ):  # Stop hand cursor after pan
                if not (
                    event.buttons() & Qt.LeftButton
                ):  # If left button is no longer pressed
                    self.viewport().setCursor(
                        Qt.OpenHandCursor
                        if self.current_interaction_mode == CanvasInteractionMode.PAN
                        else Qt.ArrowCursor
                    )
            super().mouseReleaseEvent(event)

    def paintEvent(
        self, event
    ):  # This is paintEvent for InteractiveSegmentationView itself
        super().paintEvent(event)

        # Draw custom eraser cursor if in eraser mode AND mouse is over the viewport
        if (
            self.current_interaction_mode == CanvasInteractionMode.ERASER
            and self.viewport().underMouse()
        ):
            painter = QPainter(self.viewport())  # Painter for the viewport
            painter.setRenderHint(QPainter.Antialiasing)

            # Get mouse position relative to the viewport
            mouse_pos_viewport = self.viewport().mapFromGlobal(QCursor.pos())

            pen = QPen(self.eraser_cursor_color.darker(150), 1)
            brush = QBrush(self.eraser_cursor_color)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(
                mouse_pos_viewport,
                self._eraser_radius_screen,
                self._eraser_radius_screen,
            )

    def _add_prompt_visual_and_data(
        self, prompt_id: str, image_pos: QPointF, is_positive: bool
    ):
        """Adds both the visual marker and the data entry for a prompt."""
        if self._pixmap_item.pixmap().isNull():
            return

        color = (
            self.positive_prompt_color if is_positive else self.negative_prompt_color
        )
        scene_dot_pos = self._pixmap_item.mapToScene(image_pos)

        dot = QGraphicsEllipseItem(
            -self._dot_radius_screen,
            -self._dot_radius_screen,
            2 * self._dot_radius_screen,
            2 * self._dot_radius_screen,
        )
        dot.setPos(scene_dot_pos)
        dot.setPen(QPen(color.darker(120), 0.5))  # Thinner border for dots
        dot.setBrush(QBrush(color))
        dot.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)

        self._scene.addItem(dot)
        self._prompt_data_list.append(
            {
                "id": prompt_id,
                "image_pos": image_pos,
                "is_positive": is_positive,
                "item": dot,
            }
        )
        self.update()

    def _erase_prompts_at_view_pos(self, view_mouse_pos: QPoint):
        """Erases prompts whose visuals are near the given mouse position in view coordinates."""
        if self._pixmap_item.pixmap().isNull() or not self._prompt_data_list:
            return

        removed_prompt_ids = []
        prompts_to_keep = []

        for prompt_entry in self._prompt_data_list:
            prompt_visual_item = prompt_entry["item"]

            prompt_center_view_pos = self.mapFromScene(
                prompt_visual_item.pos()
            )  # pos() is its center in scene

            distance_vector = view_mouse_pos - prompt_center_view_pos
            distance = distance_vector.manhattanLength()  # Using manhattan for speed

            if distance <= (
                self._eraser_radius_screen + self._dot_radius_screen
            ):  # Check against combined radii
                logger.debug(f"Erasing prompt ID: {prompt_entry['id']}")
                if prompt_visual_item.scene() == self._scene:
                    self._scene.removeItem(prompt_visual_item)
                removed_prompt_ids.append(prompt_entry["id"])
            else:
                prompts_to_keep.append(prompt_entry)

        if removed_prompt_ids:
            self._prompt_data_list = prompts_to_keep
            self.prompts_erased.emit(removed_prompt_ids)
            self.viewport().update()  # Request repaint of viewport if eraser cursor is drawn there
            self.update()  # Request repaint of the view itself (if prompts changed)

    def clear_all_prompts(self):
        removed_ids = [p_data["id"] for p_data in self._prompt_data_list]

        # Remove visual items from scene
        for entry in self._prompt_data_list:
            visual_item = entry.get("item")
            if visual_item and visual_item.scene() == self._scene:
                self._scene.removeItem(visual_item)

        self._prompt_data_list.clear()

        if removed_ids:
            self.prompts_erased.emit(removed_ids)
        logger.info("All prompts cleared from the canvas.")
        self.update()

    def load_prompts(self, prompts_data: List[Dict[str, Any]]):
        """
        Loads a list of prompts and displays them.
        Each dict in prompts_data: {'id': str, 'image_pos': QPointF, 'is_positive': bool}
        """
        self.clear_all_prompts()
        for p_data in prompts_data:
            self._add_prompt_visual_and_data(
                p_data["id"], p_data["image_pos"], p_data["is_positive"]
            )

    def get_current_prompts_data(self) -> List[Dict[str, Any]]:
        """Returns a list of current prompt data (excluding visual items)."""
        return [
            {
                "id": p["id"],
                "image_pos": p["image_pos"],
                "is_positive": p["is_positive"],
            }
            for p in self._prompt_data_list
        ]
