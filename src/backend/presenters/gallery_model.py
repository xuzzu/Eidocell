# backend/presenters/gallery_model.py

import logging
from collections import OrderedDict
from typing import Any, List, Optional

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
)
from PySide6.QtCore import Signal as pyqtSignal
from PySide6.QtGui import QColor, QPixmap

from UI.navigation_interface.workspace.views.gallery.image_card import ImageCard


class LoadPixmapTask(QRunnable):
    def __init__(
        self, image_card_data: ImageCard, target_size: QSize, load_mask: bool, callback
    ):
        super().__init__()
        self.image_card_data = image_card_data
        self.target_size = target_size
        self.load_mask = load_mask
        self.callback = (
            callback  # Expected: callback(image_id, scaled_pixmap, is_mask_type)
        )

    def run(self):
        original_pixmap: Optional[QPixmap] = None
        is_mask_type = self.load_mask
        image_id = self.image_card_data.id
        scaled_pixmap: Optional[QPixmap] = None  # Initialize

        try:
            if self.load_mask:
                if hasattr(self.image_card_data, "load_mask_pixmap") and callable(
                    self.image_card_data.load_mask_pixmap
                ):
                    original_pixmap = self.image_card_data.load_mask_pixmap()
                    if original_pixmap is None:
                        logging.debug(
                            f"LoadPixmapTask: image_card_data.load_mask_pixmap() returned None for ID {image_id}."
                        )
                else:
                    logging.error(
                        f"LoadPixmapTask: ImageCard ID {image_id} missing load_mask_pixmap method."
                    )
                    self.callback(image_id, None, is_mask_type)
                    return
            else:  # Load original image
                if hasattr(self.image_card_data, "load_pixmap") and callable(
                    self.image_card_data.load_pixmap
                ):
                    original_pixmap = self.image_card_data.load_pixmap()
                    if original_pixmap is None:
                        logging.debug(
                            f"LoadPixmapTask: image_card_data.load_pixmap() returned None for ID {image_id}."
                        )
                else:
                    logging.error(
                        f"LoadPixmapTask: ImageCard ID {image_id} missing load_pixmap method."
                    )
                    self.callback(image_id, None, is_mask_type)
                    return

            if original_pixmap and not original_pixmap.isNull():
                # Ensure target_size is valid before attempting to scale
                if (
                    self.target_size.isValid()
                    and self.target_size.width() > 0
                    and self.target_size.height() > 0
                ):
                    scaled_pixmap = original_pixmap.scaled(
                        self.target_size,
                        Qt.KeepAspectRatio,  # Keep aspect ratio
                        Qt.SmoothTransformation,
                    )
                    if scaled_pixmap.isNull():
                        logging.warning(
                            f"LoadPixmapTask: Scaling failed for ID {image_id} (mask={self.load_mask}). "
                            f"Original size: {original_pixmap.size()}, Target: {self.target_size}. Using original (unscaled) pixmap if small enough, or placeholder."
                        )
                        scaled_pixmap = None
                else:
                    logging.warning(
                        f"LoadPixmapTask: Invalid target_size {self.target_size} for scaling ID {image_id} (mask={self.load_mask}). Using original (unscaled) if small enough."
                    )
                    scaled_pixmap = None
            else:
                if self.load_mask:
                    logging.debug(
                        f"LoadPixmapTask: Original MASK pixmap for ID {image_id} is null or load failed. Mask path: {self.image_card_data.mask_path}"
                    )
                else:
                    logging.debug(
                        f"LoadPixmapTask: Original IMAGE pixmap for ID {image_id} is null or load failed. Image path: {self.image_card_data.path}"
                    )

            self.callback(image_id, scaled_pixmap, is_mask_type)

        except Exception as e:
            logging.error(
                f"LoadPixmapTask: Exception for ID {image_id} (mask={self.load_mask}): {e}",
                exc_info=True,
            )
            self.callback(image_id, None, is_mask_type)


class GalleryModel(QAbstractListModel):
    """
    Custom model to manage gallery images efficiently with pre-scaled pixmaps.
    """

    def __init__(self, images: Optional[List[ImageCard]] = None, parent=None):
        super().__init__(parent)
        self._images: List[ImageCard] = images or []
        self._pixmap_cache = OrderedDict()
        self._mask_pixmap_cache = OrderedDict()
        self._max_cache_size = 2000  # Max items per cache

        self.placeholder_pixmap = QPixmap(64, 64)
        self.placeholder_pixmap.fill(QColor(220, 220, 220, 150))
        self.thread_pool = QThreadPool.globalInstance()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._images)

    def data(
        self, index: QModelIndex, role: int = Qt.DisplayRole
    ) -> Any:  # Changed to Any
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        image_card_data = self._images[index.row()]

        if role == Qt.DisplayRole:
            return image_card_data.name
        elif role == Qt.UserRole:
            return image_card_data
        elif role == Qt.DecorationRole:
            # Determine current view mode and target card size from the delegate
            view = self.parent()  # QListView instance
            current_view_mode = "image"
            total_card_item_size = QSize(100, 130)

            if view and hasattr(view, "delegate") and view.delegate:
                delegate = view.delegate
                if hasattr(delegate, "view_mode"):
                    current_view_mode = delegate.view_mode
                if hasattr(delegate, "card_size"):
                    total_card_item_size = delegate.card_size

                if hasattr(delegate, "get_image_display_area_size"):
                    # Pass the total card size to the delegate's helper
                    pixmap_target_size = delegate.get_image_display_area_size(
                        total_card_item_size
                    )
                else:
                    # Fallback if method doesn't exist (should not happen with new delegate)
                    pixmap_target_size = total_card_item_size * 0.7  # Rough estimate
                    logging.warning(
                        "Delegate missing 'get_image_display_area_size', using fallback."
                    )
            else:
                pixmap_target_size = QSize(70, 90)
                logging.warning(
                    "View or delegate not found for GalleryModel, using default pixmap target size."
                )

            cache_to_use = (
                self._mask_pixmap_cache
                if current_view_mode == "mask"
                else self._pixmap_cache
            )

            if image_card_data.id in cache_to_use:
                return cache_to_use[image_card_data.id]
            else:
                self.loadPixmapAsync(
                    image_card_data,
                    index.row(),
                    pixmap_target_size,
                    current_view_mode == "mask",
                )
                return self.placeholder_pixmap
        return None

    def loadPixmapAsync(
        self,
        image_card_data: ImageCard,
        row_index: int,
        target_size: QSize,
        load_mask: bool,
    ):
        # For now, relying on cache check in data() and callback logic.

        def onPixmapLoaded(
            image_id: str, scaled_pixmap: Optional[QPixmap], is_mask_type: bool
        ):
            if scaled_pixmap and not scaled_pixmap.isNull():
                cache_dict = (
                    self._mask_pixmap_cache if is_mask_type else self._pixmap_cache
                )
                cache_dict[image_id] = scaled_pixmap
                # logging.debug(f"Cached {'mask' if is_mask_type else 'image'} for {image_id}, size: {scaled_pixmap.size()}")

                # Prune cache
                while len(cache_dict) > self._max_cache_size:
                    oldest_id, _ = cache_dict.popitem(last=False)
                    # logging.debug(f"Cache full, removed oldest {'mask' if is_mask_type else 'image'} for {oldest_id}")
            else:
                logging.warning(
                    f"Async load returned null scaled pixmap for ID {image_id}, is_mask={is_mask_type}"
                )

            # Find current row for the image_id, as it might have changed
            current_row = -1
            for i, img_card_in_model in enumerate(self._images):
                if img_card_in_model.id == image_id:
                    current_row = i
                    break

            if current_row != -1:
                idx = self.index(current_row)
                if idx.isValid():
                    view = self.parent()
                    current_view_mode_now = "image"
                    if (
                        view
                        and hasattr(view, "delegate")
                        and view.delegate
                        and hasattr(view.delegate, "view_mode")
                    ):
                        current_view_mode_now = view.delegate.view_mode

                    if (is_mask_type and current_view_mode_now == "mask") or (
                        not is_mask_type and current_view_mode_now == "image"
                    ):
                        self.dataChanged.emit(idx, idx, [Qt.DecorationRole])
            else:
                logging.debug(
                    f"Image ID {image_id} no longer in model when pixmap loaded."
                )

        task = LoadPixmapTask(image_card_data, target_size, load_mask, onPixmapLoaded)
        self.thread_pool.start(task)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def addImages(self, image_cards_data: List[ImageCard]):
        if not image_cards_data:
            return

        self.beginInsertRows(
            QModelIndex(), self.rowCount(), self.rowCount() + len(image_cards_data) - 1
        )
        self._images.extend(image_cards_data)
        self.endInsertRows()

    def addImage(self, image_card_data: ImageCard):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._images.append(image_card_data)
        self.endInsertRows()

    def removeImage(self, row: int):
        if 0 <= row < self.rowCount():
            self.beginRemoveRows(QModelIndex(), row, row)
            image_card_data = self._images.pop(row)
            if image_card_data.id in self._pixmap_cache:
                del self._pixmap_cache[image_card_data.id]
            if image_card_data.id in self._mask_pixmap_cache:
                del self._mask_pixmap_cache[image_card_data.id]
            self.endRemoveRows()

    def clear(self):
        self.beginResetModel()
        self._images.clear()
        self._pixmap_cache.clear()
        self._mask_pixmap_cache.clear()
        self.endResetModel()

    def updateImage(self, row: int, new_image_card_data: ImageCard):
        if 0 <= row < self.rowCount():
            old_image_card_data = self._images[row]
            self._images[row] = new_image_card_data

            if old_image_card_data.id != new_image_card_data.id:
                if old_image_card_data.id in self._pixmap_cache:
                    del self._pixmap_cache[old_image_card_data.id]
                if old_image_card_data.id in self._mask_pixmap_cache:
                    del self._mask_pixmap_cache[old_image_card_data.id]
            else:
                if new_image_card_data.id in self._pixmap_cache:
                    del self._pixmap_cache[new_image_card_data.id]
                if new_image_card_data.id in self._mask_pixmap_cache:
                    del self._mask_pixmap_cache[new_image_card_data.id]

            idx = self.index(row)
            self.dataChanged.emit(idx, idx)

    def reorderImagesByIds(self, sorted_image_ids: List[str]):
        id_to_image_card = {image.id: image for image in self._images}
        new_images_ordered = []
        processed_ids = set()

        for image_id in sorted_image_ids:
            if image_id in id_to_image_card:
                new_images_ordered.append(id_to_image_card[image_id])
                processed_ids.add(image_id)

        # Add any images that were in the model but not in sorted_image_ids (e.g., new items)
        for image_card in self._images:
            if image_card.id not in processed_ids:
                new_images_ordered.append(image_card)  # Append at the end

        if len(self._images) != len(new_images_ordered) or any(
            o.id != n.id for o, n in zip(self._images, new_images_ordered)
        ):
            self.layoutAboutToBeChanged.emit()
            self._images = new_images_ordered
            self.layoutChanged.emit()

    def invalidate_pixmap_cache_and_refresh(self):
        """
        Clears the pixmap caches and triggers a refresh of all items in the view.
        Call this when the target card size changes (e.g., from GalleryView).
        """
        logging.debug("Model: Invalidating pixmap cache and refreshing all items.")
        self._pixmap_cache.clear()
        self._mask_pixmap_cache.clear()

        if self.rowCount() > 0:
            top_left = self.index(0, 0)
            bottom_right = self.index(self.rowCount() - 1, 0)
            self.dataChanged.emit(top_left, bottom_right, [Qt.DecorationRole])
            logging.debug("Model: dataChanged emitted for all items (DecorationRole).")

    def get_selected_images(self):
        """
        Returns a list of selected images.

        Returns:
            list[Sample]: List of selected images.
        """
        selected_indices = self.selectedIndexes()
        selected_images = [self._images[idx.row()] for idx in selected_indices]
        return selected_images
