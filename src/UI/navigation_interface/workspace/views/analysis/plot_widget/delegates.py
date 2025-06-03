# delegates.py

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate


class ImageDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()

        # Draw the background for selected items
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # Get data from the model
        id_ = index.data(Qt.DisplayRole)
        pixmap = index.data(Qt.DecorationRole)

        if isinstance(pixmap, QPixmap):
            # Scale pixmap to fit within the item rect while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                option.rect.size() * 0.8, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            # Calculate position to center the pixmap
            pixmap_x = (
                option.rect.x() + (option.rect.width() - scaled_pixmap.width()) / 2
            )
            pixmap_y = (
                option.rect.y()
                + (option.rect.height() - scaled_pixmap.height()) / 2
                - 10
            )  # Adjust for text
            painter.drawPixmap(pixmap_x, pixmap_y, scaled_pixmap)

        # Draw the ID text below the image
        text_rect = QRect(
            option.rect.x(),
            option.rect.y() + option.rect.height() * 0.6,
            option.rect.width(),
            option.rect.height() * 0.4,
        )
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignCenter, f"{id_}")

        painter.restore()
