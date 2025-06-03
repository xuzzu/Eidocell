# src/UI/navigation_interface/workspace/views/segmentation/advanced_segmentation_widget.py

import logging
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QPointF, QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ComboBox,
    FluentIcon,
    IconWidget,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SegmentedWidget,
    Slider,
    TogglePushButton,
    ToolButton,
)

from UI.common.style_sheet import EidocellStyleSheet
from UI.navigation_interface.workspace.views.segmentation.interactive_segmentation_canvas import (
    CanvasInteractionMode,
    InteractiveSegmentationView,
)

logger_adv_seg_widget = logging.getLogger(__name__)  # Renamed logger


class ActiveMaskItemWidget(CardWidget):
    visibility_toggled = Signal(bool)
    opacity_changed = Signal(int)
    delete_requested = Signal()

    def __init__(
        self,
        mask_name: str,
        mask_color_hex: str,
        initial_opacity=60,
        is_visible=True,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("activeMaskItemCard")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        top_layout = QHBoxLayout()
        self.color_indicator = QFrame(self)
        self.color_indicator.setObjectName("colorIndicatorFrame")
        self.color_indicator.setFixedSize(16, 16)
        self.color_indicator.setStyleSheet(
            f"background-color: {mask_color_hex}; border-radius: 8px;"
        )
        top_layout.addWidget(self.color_indicator)

        self.name_label = BodyLabel(mask_name, self)
        self.name_label.setFont(QFont("Arial", 10, QFont.Bold))
        top_layout.addWidget(self.name_label, 1)
        main_layout.addLayout(top_layout)

        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(BodyLabel("Opacity", self))
        self.opacity_slider = Slider(Qt.Horizontal, self)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(initial_opacity)
        self.opacity_slider.setFixedWidth(155)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_value_label = BodyLabel(f"{initial_opacity}%", self)
        opacity_layout.addWidget(self.opacity_value_label)
        main_layout.addLayout(opacity_layout)

        actions_layout = QHBoxLayout()
        actions_layout.addStretch(1)
        self.visibility_button = ToolButton(
            FluentIcon.VIEW if is_visible else FluentIcon.HIDE, self
        )
        self.visibility_button.setCheckable(True)
        self.visibility_button.setChecked(is_visible)
        self.visibility_button.setToolTip("Toggle Mask Visibility")
        actions_layout.addWidget(self.visibility_button)

        self.delete_button = ToolButton(FluentIcon.DELETE, self)
        self.delete_button.setToolTip("Delete Mask")
        actions_layout.addWidget(self.delete_button)
        main_layout.addLayout(actions_layout)

        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        self.visibility_button.toggled.connect(self.visibility_toggled)
        self.delete_button.clicked.connect(self.delete_requested)

    def _on_opacity_slider_changed(self, value):
        self.opacity_value_label.setText(f"{value}%")
        self.opacity_changed.emit(value)

    def set_name(self, name):
        self.name_label.setText(name)

    def set_color(self, color_hex):
        self.color_indicator.setStyleSheet(
            f"background-color: {color_hex}; border-radius: 8px;"
        )

    def set_opacity(self, opacity):
        self.opacity_slider.setValue(opacity)

    def set_visibility_icon(self, is_visible):
        self.visibility_button.setIcon(
            FluentIcon.VIEW if is_visible else FluentIcon.HIDE
        )
        self.visibility_button.setChecked(is_visible)


class AdvSegLeftPanel(QFrame):
    interaction_mode_selected = Signal(
        CanvasInteractionMode
    )  # Emits the selected CanvasInteractionMode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("advSegLeftPanelContainer")
        self.setFixedWidth(300)

        self.styled_content_frame = QFrame(self)
        self.styled_content_frame.setObjectName("advSegLeftStyledFrame")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(1)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.styled_content_frame.setGraphicsEffect(shadow)

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(self.styled_content_frame)
        outer_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(outer_layout)

        layout = QVBoxLayout(self.styled_content_frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignTop)

        collections_group = QGroupBox("Active Collection", self.styled_content_frame)
        collections_layout = QVBoxLayout(collections_group)
        collections_layout.setSpacing(8)
        collections_layout.setContentsMargins(0, 20, 0, 0)
        self.collection_combo = ComboBox(collections_group)
        self.collection_combo.addItems(
            ["All Gallery Images", "Class: Example 1", "Cluster: XYZ"]
        )
        collections_layout.addWidget(self.collection_combo)
        layout.addWidget(collections_group)

        model_selection_group = QGroupBox("Model Selection", self.styled_content_frame)
        ms_layout = QVBoxLayout(model_selection_group)
        ms_layout.setSpacing(8)
        ms_layout.setContentsMargins(0, 20, 0, 0)
        self.model_combo = ComboBox(model_selection_group)
        self.model_combo.addItems(["SAM2 Base", "SAM2 Tiny Hiera"])
        ms_layout.addWidget(self.model_combo)
        model_description = QLabel(
            "General purpose segmentation", model_selection_group
        )
        model_description.setObjectName("modelDescriptionLabel")
        ms_layout.addWidget(model_description)
        self.fine_tune_button = PushButton(
            FluentIcon.TRAIN, "Fine-tune Model", model_selection_group
        )
        self.fine_tune_button.setObjectName("fineTuneButton")
        self.fine_tune_button.setEnabled(False)
        ms_layout.addWidget(self.fine_tune_button)
        layout.addWidget(model_selection_group)

        prompting_tools_group = QGroupBox("Prompting Tools", self.styled_content_frame)
        tools_layout = QVBoxLayout(prompting_tools_group)
        tools_layout.setSpacing(6)
        tools_layout.setContentsMargins(0, 20, 0, 0)

        self.prompt_button_group = QButtonGroup(self)
        self.prompt_button_group.setExclusive(True)

        self.pan_button = TogglePushButton(
            FluentIcon.ZOOM_IN, "Pan/Zoom Tool", prompting_tools_group
        )  # Icon Change
        self.pan_button.setCheckable(True)
        self.pan_button.setChecked(True)
        tools_layout.addWidget(self.pan_button)
        self.prompt_button_group.addButton(self.pan_button)

        self.point_prompt_button = TogglePushButton(
            FluentIcon.PIN, "Point Prompt", prompting_tools_group
        )
        self.point_prompt_button.setCheckable(True)
        tools_layout.addWidget(self.point_prompt_button)
        self.prompt_button_group.addButton(self.point_prompt_button)

        self.box_prompt_button = TogglePushButton(
            FluentIcon.LAYOUT, "Box Prompt", prompting_tools_group
        )
        self.box_prompt_button.setCheckable(True)
        self.box_prompt_button.setEnabled(False)
        tools_layout.addWidget(self.box_prompt_button)
        self.prompt_button_group.addButton(self.box_prompt_button)

        self.eraser_button = TogglePushButton(
            FluentIcon.ERASE_TOOL, "Eraser", prompting_tools_group
        )
        self.eraser_button.setCheckable(True)
        tools_layout.addWidget(self.eraser_button)
        self.prompt_button_group.addButton(self.eraser_button)

        layout.addWidget(prompting_tools_group)

        # Clear Prompts Button
        self.clear_prompts_button = PushButton(
            FluentIcon.CLEAR_SELECTION, "Clear All Prompts", self.styled_content_frame
        )  # Moved button
        layout.addWidget(self.clear_prompts_button)  # Add it below the tools group

        layout.addStretch(1)

        self.apply_to_collection_button = PrimaryPushButton(
            FluentIcon.PLAY_SOLID, "Apply to Collection", self.styled_content_frame
        )
        self.apply_to_collection_button.setDisabled(True)  # Initially disabled
        layout.addWidget(self.apply_to_collection_button)

        self.pan_button.toggled.connect(
            lambda checked: self._on_tool_button_toggled(
                CanvasInteractionMode.PAN, checked
            )
        )
        self.point_prompt_button.toggled.connect(
            lambda checked: self._on_tool_button_toggled(
                CanvasInteractionMode.POINT_PROMPT, checked
            )
        )
        self.box_prompt_button.toggled.connect(
            lambda checked: self._on_tool_button_toggled(
                CanvasInteractionMode.BOX_PROMPT, checked
            )
        )  # Keep for future
        self.eraser_button.toggled.connect(
            lambda checked: self._on_tool_button_toggled(
                CanvasInteractionMode.ERASER, checked
            )
        )

    @Slot(CanvasInteractionMode, bool)
    def _on_tool_button_toggled(self, mode: CanvasInteractionMode, checked: bool):
        if checked:
            logger_adv_seg_widget.debug(f"Left Panel: Tool selected - {mode.name}")
            self.interaction_mode_selected.emit(mode)


class AdvSegStripCardWidget(QFrame):
    clicked = Signal()

    def __init__(
        self, image_id: str, thumbnail_pixmap: Optional[QPixmap] = None, parent=None
    ):
        super().__init__(parent)
        self.image_id = image_id
        self.thumbnail_pixmap = thumbnail_pixmap
        self.is_selected = False
        self.has_mask = False
        self._is_hovered = False

        self.setFixedSize(90, 90)
        self.setCursor(Qt.PointingHandCursor)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setObjectName("thumbnailImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(70, 70)

        if self.thumbnail_pixmap and not self.thumbnail_pixmap.isNull():
            self.image_label.setPixmap(
                self.thumbnail_pixmap.scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        else:
            self.image_label.setText(image_id[:4] + "...")
            self.image_label.setFont(QFont("Arial", 7))

        self.main_layout.addWidget(self.image_label)

    def set_thumbnail(self, pixmap: QPixmap):
        self.thumbnail_pixmap = pixmap
        if self.thumbnail_pixmap and not self.thumbnail_pixmap.isNull():
            self.image_label.setPixmap(
                self.thumbnail_pixmap.scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            self.image_label.setText("")
        else:
            self.image_label.clear()
            self.image_label.setText(self.image_id[:4] + "...")
        self.update()

    def set_selected(self, selected: bool):
        if self.is_selected != selected:
            self.is_selected = selected
            self.update()

    def set_has_mask(self, has_mask: bool):
        if self.has_mask != has_mask:
            self.has_mask = has_mask
            self.update()

    def enterEvent(self, event):
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        border_radius = 6
        bg_color = QColor("#ffffff")
        if self._is_hovered:
            bg_color = QColor("#e8f0fe")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), border_radius, border_radius)
        painter.setClipPath(path)
        painter.drawRoundedRect(self.rect(), border_radius, border_radius)
        border_color = QColor(Qt.transparent)
        border_width = 0
        if self.is_selected:
            border_color = QColor("#0078d4")
            border_width = 2
        elif self.has_mask:
            border_color = QColor("#4CAF50")
            border_width = 2
        elif self._is_hovered and not self.is_selected and not self.has_mask:
            border_color = QColor("#a0c4e8")
            border_width = 1
        if border_width > 0:
            painter.setBrush(Qt.NoBrush)
            pen = QPen(border_color, border_width)
            painter.setPen(pen)
            painter.drawRoundedRect(
                rect,
                border_radius - (border_width / 2.0),
                border_radius - (border_width / 2.0),
            )
        super().paintEvent(event)


class AdvSegImageStrip(QWidget):
    image_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("advSegImageStripContainer")
        self.setFixedHeight(110)
        self.styled_content_frame = QFrame(self)
        self.styled_content_frame.setObjectName("advSegImageStripStyledFrame")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(-1)
        shadow.setColor(QColor(0, 0, 0, 15))
        self.styled_content_frame.setGraphicsEffect(shadow)
        outer_strip_layout = QVBoxLayout(self)
        outer_strip_layout.addWidget(self.styled_content_frame)
        outer_strip_layout.setContentsMargins(5, 8, 5, 2)
        self.setLayout(outer_strip_layout)
        internal_layout = QHBoxLayout(self.styled_content_frame)
        internal_layout.setContentsMargins(10, 8, 10, 8)
        internal_layout.setSpacing(8)
        self.prev_button = ToolButton(FluentIcon.LEFT_ARROW, self.styled_content_frame)
        self.prev_button.setFixedSize(28, 28)
        self.image_list_widget = QListWidget(self.styled_content_frame)
        self.image_list_widget.setViewMode(QListWidget.IconMode)
        self.image_list_widget.setFlow(QListWidget.LeftToRight)
        self.image_list_widget.setMovement(QListWidget.Static)
        self.image_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_list_widget.setSpacing(4)
        self.next_button = ToolButton(FluentIcon.RIGHT_ARROW, self.styled_content_frame)
        self.next_button.setFixedSize(28, 28)
        internal_layout.addWidget(self.prev_button)
        internal_layout.addWidget(self.image_list_widget, 1)
        internal_layout.addWidget(self.next_button)
        self.prev_button.clicked.connect(
            lambda: self.image_list_widget.horizontalScrollBar().setValue(
                self.image_list_widget.horizontalScrollBar().value()
                - self.image_list_widget.width() // 2
            )
        )
        self.next_button.clicked.connect(
            lambda: self.image_list_widget.horizontalScrollBar().setValue(
                self.image_list_widget.horizontalScrollBar().value()
                + self.image_list_widget.width() // 2
            )
        )
        self.image_list_widget.itemClicked.connect(self._on_item_clicked)
        self.current_selected_item_widget: Optional[AdvSegStripCardWidget] = None

    def _on_item_clicked(self, item: QListWidgetItem):
        card_widget = self.image_list_widget.itemWidget(item)
        if isinstance(card_widget, AdvSegStripCardWidget):
            if self.current_selected_item_widget:
                self.current_selected_item_widget.set_selected(False)
            card_widget.set_selected(True)
            self.current_selected_item_widget = card_widget
            self.image_selected.emit(card_widget.image_id)

    def populate_images(self, image_infos: list):
        self.image_list_widget.clear()
        self.current_selected_item_widget = None
        for info in image_infos:
            pixmap = None
            if info.get("thumbnail_path") and os.path.exists(info["thumbnail_path"]):
                pixmap = QPixmap(info["thumbnail_path"])
            card_widget = AdvSegStripCardWidget(info["id"], pixmap)
            card_widget.set_has_mask(info.get("has_mask", False))
            item = QListWidgetItem(self.image_list_widget)
            item.setSizeHint(card_widget.sizeHint())
            self.image_list_widget.addItem(item)
            self.image_list_widget.setItemWidget(item, card_widget)
        if self.image_list_widget.count() > 0:
            first_item = self.image_list_widget.item(0)
            if first_item:
                self._on_item_clicked(first_item)

    def update_card_mask_status(self, image_id: str, has_mask: bool):
        for i in range(self.image_list_widget.count()):
            item = self.image_list_widget.item(i)
            widget = self.image_list_widget.itemWidget(item)
            if (
                isinstance(widget, AdvSegStripCardWidget)
                and widget.image_id == image_id
            ):
                widget.set_has_mask(has_mask)
                break

    def set_selected_card(self, image_id: str):
        for i in range(self.image_list_widget.count()):
            item = self.image_list_widget.item(i)
            widget = self.image_list_widget.itemWidget(item)
            if (
                isinstance(widget, AdvSegStripCardWidget)
                and widget.image_id == image_id
            ):
                self._on_item_clicked(item)
                self.image_list_widget.scrollToItem(
                    item, QListWidget.ScrollHint.EnsureVisible
                )
                break


class AdvSegRightPanel(QWidget):
    # Signals to Presenter (layer management related)
    mask_layer_create_requested = Signal(str, str)  # name, color_hex
    mask_layer_delete_requested = Signal(str)  # layer_id (from UI item)
    mask_layer_visibility_changed = Signal(str, bool)  # layer_id, is_visible
    mask_layer_opacity_changed = Signal(str, int)  # layer_id, opacity (0-100)
    mask_layer_color_change_requested = Signal(
        str, str
    )  # layer_id, new_color_hex (TODO: Add color picker to item)
    mask_layer_rename_requested = Signal(
        str, str
    )  # layer_id, new_name (TODO: Add rename mechanism)
    active_layer_changed = Signal(Optional[str])  # layer_id or None if deselected

    # Signals to Presenter (segmentation model interaction related)
    segmentation_proposal_accept_requested = Signal(str)  # layer_id to accept into
    segmentation_proposal_clear_requested = Signal()
    segmentation_undo_last_action_requested = (
        Signal()
    )  # Undo last prompt or mask modification step

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("advSegRightPanelContainer")
        self.setFixedWidth(320)
        self.styled_content_frame = QFrame(self)
        self.styled_content_frame.setObjectName("advSegRightStyledFrame")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(1)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.styled_content_frame.setGraphicsEffect(shadow)
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(self.styled_content_frame)
        outer_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(outer_layout)
        main_layout = QVBoxLayout(self.styled_content_frame)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        main_layout.setAlignment(Qt.AlignTop)

        creator_group = QGroupBox("Mask Creator", self.styled_content_frame)
        creator_layout = QVBoxLayout(creator_group)
        creator_layout.setSpacing(10)
        creator_layout.setContentsMargins(0, 20, 0, 0)
        name_label = QLabel("Mask Layer Name", creator_group)
        name_label.setObjectName("maskNameLabel")
        creator_layout.addWidget(name_label)
        self.mask_name_edit = LineEdit(creator_group)
        self.mask_name_edit.setPlaceholderText("e.g., Nucleus, Cell Body...")
        creator_layout.addWidget(self.mask_name_edit)
        color_label = QLabel("Layer Color", creator_group)
        color_label.setObjectName("maskColorLabel")
        creator_layout.addWidget(color_label)
        self.color_palette_layout = QHBoxLayout()
        self.color_palette_layout.setSpacing(5)
        self.color_buttons_group = QButtonGroup(self)
        self.color_buttons_group.setExclusive(True)  # For radio-button like behavior
        self.color_buttons = []
        colors = [
            "#A162D8",
            "#F47070",
            "#4CAF50",
            "#2196F3",
            "#FFC107",
            "#FF9800",
            "#E91E63",
            "#673AB7",
            "#3F51B5",
        ]
        for color_hex in colors:
            color_button = PushButton("", creator_group)
            color_button.setProperty("class", "colorPaletteButton")
            color_button.setFixedSize(24, 24)
            color_button.setCheckable(True)
            color_button.setStyleSheet(
                f"""QPushButton {{ background-color: {color_hex}; border-radius: 3px; border: 1px solid transparent; }}
                                            QPushButton:hover {{ border: 1px solid #707070; }}
                                            QPushButton:checked {{ border: 2px solid black; }}"""
            )
            color_button.setProperty("color_hex", color_hex)
            color_button.clicked.connect(self._on_color_button_selected)
            self.color_palette_layout.addWidget(color_button)
            self.color_buttons.append(color_button)
            self.color_buttons_group.addButton(color_button)  # Add to button group
        self.color_palette_layout.addStretch(1)
        creator_layout.addLayout(self.color_palette_layout)
        self.selected_color_hex = None
        if self.color_buttons:
            self.color_buttons[0].setChecked(True)
            self.selected_color_hex = self.color_buttons[0].property("color_hex")
        self.create_mask_layer_button = PrimaryPushButton(
            FluentIcon.ADD_TO, "Create Mask Layer", creator_group
        )
        self.create_mask_layer_button.clicked.connect(
            self._on_create_mask_layer_clicked
        )
        creator_layout.addWidget(self.create_mask_layer_button)
        main_layout.addWidget(creator_group)

        active_masks_group = QGroupBox("Active Mask Layers", self.styled_content_frame)
        active_masks_main_layout = QVBoxLayout(active_masks_group)
        self.active_masks_scroll_area = ScrollArea(active_masks_group)
        self.active_masks_scroll_area.setWidgetResizable(True)
        self.active_masks_scroll_area.setFrameShape(QFrame.NoFrame)
        self.active_masks_scroll_area.setStyleSheet(
            "ScrollArea { background-color: transparent; }"
        )
        self.active_masks_widget_container = QWidget()
        self.active_masks_widget_container.setStyleSheet(
            "background-color: transparent;"
        )
        self.active_masks_layout = QVBoxLayout(self.active_masks_widget_container)
        self.active_masks_layout.setAlignment(Qt.AlignTop)
        self.active_masks_layout.setSpacing(8)
        self.active_masks_layout.setContentsMargins(0, 0, 0, 0)
        self.active_masks_scroll_area.setWidget(self.active_masks_widget_container)
        active_masks_main_layout.addWidget(self.active_masks_scroll_area)
        active_masks_main_layout.setContentsMargins(0, 20, 0, 0)
        main_layout.addWidget(active_masks_group, 1)
        self.active_mask_item_widgets: Dict[str, ActiveMaskItemWidget] = (
            {}
        )  # layer_id (uuid str) -> widget

        proposal_actions_group = QGroupBox(
            "Current Proposal", self.styled_content_frame
        )
        pa_layout = QHBoxLayout(proposal_actions_group)
        pa_layout.setContentsMargins(0, 20, 0, 0)
        self.accept_proposal_button = PushButton(
            FluentIcon.ACCEPT_MEDIUM, "Accept", proposal_actions_group
        )
        self.clear_proposal_button = PushButton(
            FluentIcon.CANCEL_MEDIUM, "Clear", proposal_actions_group
        )
        self.accept_proposal_button.setEnabled(False)
        self.clear_proposal_button.setEnabled(False)
        pa_layout.addWidget(self.accept_proposal_button)
        pa_layout.addWidget(self.clear_proposal_button)
        main_layout.addWidget(proposal_actions_group)

        # Connect proposal actions
        self.accept_proposal_button.clicked.connect(self._on_accept_proposal_clicked)
        self.clear_proposal_button.clicked.connect(
            self.segmentation_proposal_clear_requested
        )

    def _on_color_button_selected(self):
        clicked_button = self.sender()
        if isinstance(clicked_button, PushButton) and clicked_button.isChecked():
            self.selected_color_hex = clicked_button.property("color_hex")
            logger_adv_seg_widget.debug(
                f"Right Panel: Color selected: {self.selected_color_hex}"
            )
        elif (
            not self.color_buttons_group.checkedButton()
        ):  # If unchecking leads to no selection
            self.selected_color_hex = None

    def _on_create_mask_layer_clicked(self):
        layer_name = self.mask_name_edit.text().strip()
        if not layer_name:
            logger_adv_seg_widget.warning("Mask layer name empty.")
            self.mask_name_edit.setFocus()
            return
        if not self.selected_color_hex:
            logger_adv_seg_widget.warning("No color selected.")
            return
        self.mask_layer_create_requested.emit(layer_name, self.selected_color_hex)
        self.mask_name_edit.clear()

    @Slot(str, str, int, bool)
    def add_mask_layer_item_from_presenter(
        self, layer_id: str, name: str, color_hex: str, opacity=60, is_visible=True
    ):
        if layer_id in self.active_mask_item_widgets:
            logger_adv_seg_widget.warning(
                f"Layer item UI with ID {layer_id} already exists. Updating."
            )
            self.update_mask_layer_item(layer_id, name, color_hex, opacity, is_visible)
            return
        item_widget = ActiveMaskItemWidget(
            name, color_hex, opacity, is_visible, self.active_masks_widget_container
        )
        item_widget.set_color(color_hex)
        item_widget.visibility_toggled.connect(
            lambda visible, lid=layer_id: self.mask_layer_visibility_changed.emit(
                lid, visible
            )
        )
        item_widget.opacity_changed.connect(
            lambda val, lid=layer_id: self.mask_layer_opacity_changed.emit(lid, val)
        )
        item_widget.delete_requested.connect(
            lambda lid=layer_id: self.mask_layer_delete_requested.emit(lid)
        )
        self.active_masks_layout.addWidget(item_widget)
        self.active_mask_item_widgets[layer_id] = item_widget

    @Slot(str)
    def remove_mask_layer_item_from_presenter(self, layer_id: str):
        item_widget = self.active_mask_item_widgets.pop(layer_id, None)
        if item_widget:
            self.active_masks_layout.removeWidget(item_widget)
            item_widget.deleteLater()
        else:
            logger_adv_seg_widget.warning(
                f"Attempt to remove non-existent UI layer item ID: {layer_id}"
            )

    @Slot(str, str, str, int, bool)
    def update_mask_layer_item_from_presenter(
        self,
        layer_id: str,
        new_name: Optional[str] = None,
        new_color_hex: Optional[str] = None,
        new_opacity: Optional[int] = None,
        new_visibility: Optional[bool] = None,
    ):
        item_widget = self.active_mask_item_widgets.get(layer_id)
        if not item_widget:
            return
        if new_name is not None:
            item_widget.set_name(new_name)
        if new_color_hex is not None:
            item_widget.set_color(new_color_hex)
        if new_opacity is not None:
            item_widget.set_opacity(new_opacity)
        if new_visibility is not None:
            item_widget.set_visibility_icon(new_visibility)

    def get_selected_active_layer_id(self) -> Optional[str]:
        # This requires ActiveMaskItemWidget to have selection state and a way to query it,
        # or for the right panel to manage selection of one layer item.
        # For now, returning None. This needs to be implemented if "Accept Proposal"
        # needs to know which layer is currently targeted for accepting.
        # One way: Add a QListWidget to hold ActiveMaskItemWidgets and use its selection.
        # Or, make ActiveMaskItemWidgets checkable and part of a QButtonGroup.
        logger_adv_seg_widget.warning(
            "get_selected_active_layer_id not fully implemented yet."
        )
        # Example: if you had a list widget for these items:
        # current_item = self.active_masks_list_widget.currentItem()
        # if current_item: return current_item.data(Qt.UserRole) # Assuming ID stored in UserRole
        return None

    @Slot()
    def _on_accept_proposal_clicked(self):
        selected_layer_id = self.get_selected_active_layer_id()
        if selected_layer_id:
            self.segmentation_proposal_accept_requested.emit(selected_layer_id)
        else:
            # TODO: Show InfoBar: "Please select a mask layer to accept the proposal into."
            logger_adv_seg_widget.info(
                "Accept proposal: No active mask layer selected."
            )

    @Slot(bool)
    def set_proposal_actions_enabled_from_presenter(self, proposal_available: bool):
        self.accept_proposal_button.setEnabled(proposal_available)
        self.clear_proposal_button.setEnabled(proposal_available)


class AdvancedSegmentationWidget(QWidget):
    adv_canvas_prompt_added = Signal(
        str, str, QPointF, bool
    )  # image_id, prompt_id, image_pos, is_positive
    adv_canvas_prompts_erased = Signal(str, list)  # image_id, list of prompt_ids

    # Signals forwarded from AdvSegLeftPanel
    left_panel_interaction_mode_selected = Signal(CanvasInteractionMode)
    left_panel_active_collection_changed = Signal(
        str
    )  # The text of the selected collection
    left_panel_model_selected = Signal(str)  # Name of the selected segmentation model
    left_panel_fine_tune_requested = Signal()
    left_panel_apply_to_collection_requested = Signal()
    left_panel_save_current_requested = Signal()
    left_panel_export_current_requested = Signal()
    left_panel_clear_all_prompts_requested = Signal()

    # Signals forwarded from AdvSegRightPanel
    right_panel_mask_layer_create_requested = Signal(str, str)
    right_panel_mask_layer_delete_requested = Signal(str)
    right_panel_mask_layer_visibility_changed = Signal(str, bool)
    right_panel_mask_layer_opacity_changed = Signal(str, int)
    right_panel_segmentation_proposal_accept_requested = Signal(str)
    right_panel_segmentation_proposal_clear_requested = Signal()
    right_panel_segmentation_undo_last_action_requested = Signal()

    # Signals forwarded from AdvSegImageStrip
    strip_image_selected = Signal(str)  # image_id

    def __init__(self, parent=None):
        super().__init__(parent)
        EidocellStyleSheet.ADVANCED_SEGMENTATION_MODULE.apply(self)
        self.setObjectName("advancedSegmentationPane")
        self.presenter = None

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.left_panel = AdvSegLeftPanel(self)
        main_layout.addWidget(self.left_panel)
        center_area_layout = QVBoxLayout()
        center_area_layout.setSpacing(0)
        self.canvas_widget = InteractiveSegmentationView(self)
        center_area_layout.addWidget(self.canvas_widget, 1)
        self.image_strip_widget = AdvSegImageStrip(self)
        center_area_layout.addWidget(self.image_strip_widget)
        main_layout.addLayout(center_area_layout, 1)
        self.right_panel = AdvSegRightPanel(self)
        main_layout.addWidget(self.right_panel)
        self.setLayout(main_layout)
        self._connect_internal_signals()

    def _connect_internal_signals(self):
        self.left_panel.interaction_mode_selected.connect(
            self.canvas_widget.set_interaction_mode
        )
        self.left_panel.clear_prompts_button.clicked.connect(
            self.canvas_widget.clear_all_prompts
        )  # Connect clear button

        self.canvas_widget.prompt_added.connect(self._forward_canvas_prompt_added)
        self.canvas_widget.prompts_erased.connect(self._forward_canvas_prompts_erased)

        # Forward signals from child panels
        self.left_panel.interaction_mode_selected.connect(
            self.left_panel_interaction_mode_selected
        )
        self.left_panel.collection_combo.currentTextChanged.connect(
            self.left_panel_active_collection_changed
        )
        self.left_panel.model_combo.currentTextChanged.connect(
            self.left_panel_model_selected
        )
        self.left_panel.fine_tune_button.clicked.connect(
            self.left_panel_fine_tune_requested
        )
        self.left_panel.apply_to_collection_button.clicked.connect(
            self.left_panel_apply_to_collection_requested
        )
        self.left_panel.clear_prompts_button.clicked.connect(
            self.left_panel_clear_all_prompts_requested
        )

        self.right_panel.mask_layer_create_requested.connect(
            self.right_panel_mask_layer_create_requested
        )
        self.right_panel.mask_layer_delete_requested.connect(
            self.right_panel_mask_layer_delete_requested
        )
        self.right_panel.mask_layer_visibility_changed.connect(
            self.right_panel_mask_layer_visibility_changed
        )
        self.right_panel.mask_layer_opacity_changed.connect(
            self.right_panel_mask_layer_opacity_changed
        )
        self.right_panel.segmentation_proposal_accept_requested.connect(
            self.right_panel_segmentation_proposal_accept_requested
        )
        self.right_panel.segmentation_proposal_clear_requested.connect(
            self.right_panel_segmentation_proposal_clear_requested
        )
        self.right_panel.segmentation_undo_last_action_requested.connect(
            self.right_panel_segmentation_undo_last_action_requested
        )

        self.image_strip_widget.image_selected.connect(self.strip_image_selected)

    def set_presenter(self, presenter):
        self.presenter = presenter
        # Connect signals FROM presenter TO UI elements
        # These are illustrative and depend on presenter's exact signals/slots
        if hasattr(presenter, "adv_image_ready_to_display"):
            presenter.adv_image_ready_to_display.connect(self.canvas_widget.set_image)
        if hasattr(presenter, "adv_image_info_updated"):
            presenter.adv_image_info_updated.connect(
                lambda img_id, current_idx, total_imgs: self.left_panel.image_info_label.setText(
                    f"Image {current_idx+1}/{total_imgs} ({img_id[:8]}...)"
                )
            )
        if hasattr(presenter, "adv_image_navigation_state_updated"):
            presenter.adv_image_navigation_state_updated.connect(
                lambda can_go_prev, can_go_next: (
                    self.left_panel.prev_button.setEnabled(can_go_prev),
                    self.left_panel.next_button.setEnabled(can_go_next),
                )
            )
        if hasattr(
            presenter, "adv_image_collection_options_updated"
        ):  # For collection combo in left panel
            presenter.adv_image_collection_options_updated.connect(
                lambda options: (
                    self.left_panel.collection_combo.blockSignals(True),
                    self.left_panel.collection_combo.clear(),
                    self.left_panel.collection_combo.addItems(options),
                    self.left_panel.collection_combo.blockSignals(False),
                    # Optionally select first item if list is not empty
                    (
                        self.left_panel.collection_combo.setCurrentIndex(0)
                        if options
                        else None
                    ),
                )
            )
        if hasattr(
            presenter, "adv_segmentation_models_updated"
        ):  # For model combo in left panel
            presenter.adv_segmentation_models_updated.connect(
                lambda models_list, current_model_name: (
                    self.left_panel.model_combo.blockSignals(True),
                    self.left_panel.model_combo.clear(),
                    self.left_panel.model_combo.addItems(models_list),
                    (
                        self.left_panel.model_combo.setCurrentText(current_model_name)
                        if current_model_name in models_list
                        else (
                            self.left_panel.model_combo.setCurrentIndex(0)
                            if models_list
                            else None
                        )
                    ),
                    self.left_panel.model_combo.blockSignals(False),
                )
            )

        if hasattr(presenter, "adv_image_collection_updated"):
            presenter.adv_image_collection_updated.connect(
                self.image_strip_widget.populate_images
            )
        if hasattr(presenter, "adv_current_image_selected_in_strip"):
            presenter.adv_current_image_selected_in_strip.connect(
                self.image_strip_widget.set_selected_card
            )

        # Connect signals TO presenter (forwarded from child panels)
        self.left_panel_interaction_mode_selected.connect(
            presenter.handle_adv_interaction_mode_change
        )
        self.left_panel_active_collection_changed.connect(
            presenter.load_image_set_for_adv_seg
        )
        self.left_panel_model_selected.connect(
            presenter.handle_adv_model_selection_change
        )
        # ... connect other left_panel action signals ...
        self.left_panel_clear_all_prompts_requested.connect(
            lambda: presenter.handle_adv_canvas_clear_all_prompts(
                self.canvas_widget.current_image_id if self.canvas_widget else None
            )
        )

        self.adv_canvas_prompt_added.connect(presenter.handle_adv_canvas_prompt_added)
        self.adv_canvas_prompts_erased.connect(
            presenter.handle_adv_canvas_prompts_erased
        )
        self.strip_image_selected.connect(presenter.load_specific_image_for_adv_seg)

        # Right panel signals TO presenter
        self.right_panel_mask_layer_create_requested.connect(
            presenter.handle_adv_mask_layer_create
        )
        self.right_panel_mask_layer_delete_requested.connect(
            presenter.handle_adv_mask_layer_delete
        )
        self.right_panel_mask_layer_visibility_changed.connect(
            presenter.handle_adv_mask_layer_visibility_change
        )
        self.right_panel_mask_layer_opacity_changed.connect(
            presenter.handle_adv_mask_layer_opacity_change
        )
        self.right_panel_segmentation_proposal_accept_requested.connect(
            presenter.handle_adv_accept_proposal
        )
        self.right_panel_segmentation_proposal_clear_requested.connect(
            presenter.handle_adv_clear_proposal
        )
        self.right_panel_segmentation_undo_last_action_requested.connect(
            presenter.handle_adv_undo_action
        )

        # Right panel updates FROM presenter
        if hasattr(presenter, "adv_add_mask_layer_to_ui"):
            presenter.adv_add_mask_layer_to_ui.connect(
                self.right_panel.add_mask_layer_item_from_presenter
            )
        if hasattr(presenter, "adv_remove_mask_layer_from_ui"):
            presenter.adv_remove_mask_layer_from_ui.connect(
                self.right_panel.remove_mask_layer_item_from_presenter
            )
        if hasattr(presenter, "adv_update_mask_layer_in_ui"):
            presenter.adv_update_mask_layer_in_ui.connect(
                self.right_panel.update_mask_layer_item_from_presenter
            )
        if hasattr(
            presenter, "adv_mask_proposal_available"
        ):  # From presenter indicating if proposal actions should be enabled
            presenter.adv_mask_proposal_available.connect(
                self.right_panel.set_proposal_actions_enabled_from_presenter
            )

    @Slot(str, QPointF, bool)  # prompt_id, image_pos, is_positive (from canvas)
    def _forward_canvas_prompt_added(
        self, prompt_id: str, image_pos: QPointF, is_positive: bool
    ):
        current_image_on_canvas = (
            self.canvas_widget.current_image_id
        )  # Assuming canvas stores this
        if current_image_on_canvas:
            self.adv_canvas_prompt_added.emit(
                current_image_on_canvas, prompt_id, image_pos, is_positive
            )
        else:
            logger_adv_seg_widget.warning(
                "Canvas emitted prompt_added, but canvas has no current_image_id."
            )

    @Slot(list)  # erased_prompt_ids (from canvas)
    def _forward_canvas_prompts_erased(self, erased_prompt_ids: list):
        current_image_on_canvas = self.canvas_widget.current_image_id
        if current_image_on_canvas:
            self.adv_canvas_prompts_erased.emit(
                current_image_on_canvas, erased_prompt_ids
            )
        else:
            logger_adv_seg_widget.warning(
                "Canvas emitted prompts_erased, but canvas has no current_image_id."
            )

    #  Public methods for presenter to update UI components
    def update_left_panel_collections(
        self, options: List[str], current_selection: Optional[str] = None
    ):
        self.left_panel.collection_combo.blockSignals(True)
        self.left_panel.collection_combo.clear()
        self.left_panel.collection_combo.addItems(options)
        if current_selection and current_selection in options:
            self.left_panel.collection_combo.setCurrentText(current_selection)
        elif options:
            self.left_panel.collection_combo.setCurrentIndex(0)
        self.left_panel.collection_combo.blockSignals(False)
        if options:  # Trigger initial load if options exist
            self.left_panel.collection_combo.currentTextChanged.emit(
                self.left_panel.collection_combo.currentText()
            )

    def update_left_panel_models(
        self, models: List[str], current_model: Optional[str] = None
    ):
        self.left_panel.model_combo.blockSignals(True)
        self.left_panel.model_combo.clear()
        self.left_panel.model_combo.addItems(models)
        if current_model and current_model in models:
            self.left_panel.model_combo.setCurrentText(current_model)
        elif models:
            self.left_panel.model_combo.setCurrentIndex(0)
        self.left_panel.model_combo.blockSignals(False)

    def get_current_adv_seg_image_id(self) -> Optional[str]:
        """Returns the ID of the image currently displayed in the advanced segmentation canvas."""
        return self.canvas_widget.current_image_id if self.canvas_widget else None

    # Add more methods as needed for presenter to update other parts of the advanced UI.
