# UI/navigation_interface/workspace/views/gallery/gallery_delegate.py

import logging

from PySide6.QtCore import QModelIndex, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QStyle, QStyledItemDelegate


class GalleryDelegate(QStyledItemDelegate):
    card_size_changed = Signal(QSize)

    def __init__(self, parent=None, view_mode="image", card_size=QSize(100, 130)):
        super().__init__(parent)
        self.view_mode = view_mode
        self._card_size = card_size

        selection_color = QColor("#02d1ca")
        selection_color.setAlphaF(0.2)
        self.selected_brush = QBrush(selection_color)
        self.hover_brush = QBrush(QColor("#E0F7FA"))
        self.default_brush = QBrush(QColor(Qt.white))
        self.error_pen = QPen(QColor(Qt.red))
        self.text_pen = QPen(QColor(Qt.black))
        self.selection_border_pen = QPen(QColor("#02d1ca"), 2)

        # Define proportions for internal layout
        self.padding_ratio_w = 0.08  # Padding as a ratio of card width
        self.padding_ratio_h = 0.06  # Padding as a ratio of card height
        self.text_height_ratio = 0.15  # Text area height as a ratio of card height
        self.plank_height_ratio = 0.05
        self.plank_width_ratio = 0.33

        # Minimum absolute sizes for layout elements
        self.min_padding = 5
        self.min_text_height = 16  # Adjusted for potential smaller fonts
        self.min_plank_height = 4
        self.min_font_pixel_size = 8
        self.max_font_pixel_size = 14  # Prevent text from becoming overly large
        self.radius_ratio = 0.08  # Corner radius as a ratio of shorter card dimension

    @property
    def card_size(self) -> QSize:
        return self._card_size

    def set_card_size(self, new_size: QSize):
        if (
            new_size != self._card_size
            and new_size.isValid()
            and new_size.width() > 0
            and new_size.height() > 0
        ):
            logging.debug(
                f"Delegate: card_size changing from {self._card_size} to {new_size}"
            )
            self._card_size = new_size
            self.card_size_changed.emit(new_size)

            if self.parent():
                self.parent().doItemsLayout()
                self.parent().viewport().update()
        elif not new_size.isValid() or new_size.width() <= 0 or new_size.height() <= 0:
            logging.warning(f"Delegate: Attempted to set invalid card_size: {new_size}")

    def get_image_display_area_size(self, for_card_size: QSize) -> QSize:
        """
        Calculates the target size for the image pixmap itself,
        based on the provided total card_size and internal layout proportions.
        """
        if (
            not for_card_size.isValid()
            or for_card_size.width() <= 0
            or for_card_size.height() <= 0
        ):
            return QSize(1, 1)  # Fallback for invalid input

        padding_w = max(
            self.min_padding, int(for_card_size.width() * self.padding_ratio_w)
        )
        padding_h_top = max(
            self.min_padding, int(for_card_size.height() * self.padding_ratio_h)
        )
        text_h = max(
            self.min_text_height, int(for_card_size.height() * self.text_height_ratio)
        )
        padding_h_bottom = (
            padding_h_top  # Assuming symmetric vertical padding for image area
        )

        image_area_width = for_card_size.width() - (2 * padding_w)
        image_area_height = (
            for_card_size.height() - padding_h_top - text_h - padding_h_bottom
        )

        return QSize(max(1, image_area_width), max(1, image_area_height))

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        image_card_data = index.data(Qt.UserRole)
        if not image_card_data:
            painter.restore()
            return

        pixmap_from_model = index.data(
            Qt.DecorationRole
        )  # This is the pre-scaled pixmap or placeholder
        rect = option.rect  # This is the current card's bounding rectangle

        #  Calculate dynamic layout values based on current card rect size
        padding_x = max(self.min_padding, int(rect.width() * self.padding_ratio_w))
        padding_y_top = max(self.min_padding, int(rect.height() * self.padding_ratio_h))
        # Text height proportional to card height, with min/max
        text_h = max(self.min_text_height, int(rect.height() * self.text_height_ratio))

        radius = max(
            3, int(min(rect.width(), rect.height()) * self.radius_ratio)
        )  # Min radius 3

        #  Draw Card Background and Selection/Hover States
        path_bg = QPainterPath()
        path_bg.addRoundedRect(rect, radius, radius)
        painter.setClipPath(path_bg)

        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, self.selected_brush)
            painter.setPen(self.selection_border_pen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(rect, self.hover_brush)
        else:
            painter.fillRect(rect, self.default_brush)

        #  Define Image Display Area
        # This is the actual area where the image pixmap will be drawn.
        image_display_width = rect.width() - 2 * padding_x
        image_display_height = (
            rect.height() - padding_y_top - text_h - padding_y_top
        )  # Top & bottom padding for image area

        image_display_rect = QRect(
            rect.x() + padding_x,
            rect.y() + padding_y_top,
            image_display_width,
            image_display_height,
        )

        #  Draw Pixmap
        # The pixmap_from_model should have been pre-scaled by LoadPixmapTask to fit
        # the size returned by self.get_image_display_area_size(self._card_size).
        pixmap_to_draw_final = None
        if pixmap_from_model and not pixmap_from_model.isNull():
            model_placeholder = None
            gallery_view = self.parent()
            if (
                gallery_view
                and hasattr(gallery_view, "model")
                and hasattr(gallery_view.model, "placeholder_pixmap")
            ):
                model_placeholder = gallery_view.model.placeholder_pixmap

            if pixmap_from_model is model_placeholder:
                # If it's the placeholder, scale IT to fit the current image_display_rect
                pixmap_to_draw_final = pixmap_from_model.scaled(
                    image_display_rect.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            else:
                # It's a pre-scaled image. It was scaled by LoadPixmapTask to fit
                # the calculated image display area for the current card_size.
                pixmap_to_draw_final = pixmap_from_model

        if pixmap_to_draw_final and not pixmap_to_draw_final.isNull():
            # Center the pixmap (which should be already correctly scaled for this area)
            # within the image_display_rect.
            target_rect_for_pixmap = QRect(QPoint(0, 0), pixmap_to_draw_final.size())
            target_rect_for_pixmap.moveCenter(image_display_rect.center())
            painter.drawPixmap(target_rect_for_pixmap.topLeft(), pixmap_to_draw_final)
        else:
            # Draw placeholder text if pixmap is still loading or failed
            painter.setPen(self.error_pen)  # Or a less alarming color for "Loading..."
            painter.drawText(image_display_rect, Qt.AlignCenter, "Loading...")

        #  Draw Image Name
        name = image_card_data.name
        text_display_rect = QRect(  # Area for the text
            rect.x() + padding_x,
            rect.y()
            + rect.height()
            - padding_y_top
            - text_h,  # Positioned at the bottom
            rect.width() - 2 * padding_x,
            text_h,
        )

        font = painter.font()
        # Dynamically adjust font size based on available text_h, with min/max
        font_pixel_size = max(
            self.min_font_pixel_size, min(self.max_font_pixel_size, int(text_h * 0.7))
        )
        font.setPixelSize(font_pixel_size)
        painter.setFont(font)

        painter.setPen(self.text_pen)
        fm = QFontMetrics(font)
        elided_name = fm.elidedText(name, Qt.ElideRight, text_display_rect.width())
        painter.drawText(
            text_display_rect, Qt.AlignCenter | Qt.TextDontClip, elided_name
        )

        #  Draw Class Color Marker (Plank)
        if hasattr(image_card_data, "class_color") and image_card_data.class_color:
            try:
                class_color = QColor(image_card_data.class_color)
                if class_color.isValid():
                    plank_w = max(10, int(rect.width() * self.plank_width_ratio))
                    plank_h = max(
                        self.min_plank_height,
                        int(rect.height() * self.plank_height_ratio),
                    )

                    plank_rect = QRect(
                        rect.x() + (rect.width() - plank_w) // 2,
                        rect.y()
                        + padding_y_top
                        // 2,  # Position near top, using proportional padding
                        plank_w,
                        plank_h,
                    )
                    painter.setBrush(QBrush(class_color))
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(
                        plank_rect, plank_h / 2, plank_h / 2
                    )  # Rounded plank
            except Exception as e:
                logging.error(
                    f"Error processing class_color '{image_card_data.class_color}': {e}"
                )

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return self._card_size
