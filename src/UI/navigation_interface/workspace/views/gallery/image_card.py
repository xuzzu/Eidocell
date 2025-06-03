# backend/objects/sample.py
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tifffile
from PySide6.QtGui import QImage, QPixmap


@dataclass
class ImageCard:
    """
    Represents an image in the gallery.

    Attributes:
        id (str): Unique identifier for the sample.
        name (str): Display name of the sample.
        path (str): File path to the image.
        class_id (str): Identifier for the class/category the image belongs to.
        class_color (str): Color associated with the image's class for visual indicators.
        mask_path (Optional[str]): File path to the mask image, if applicable.
        # Add additional fields as necessary, e.g., features, metadata, etc.
    """

    id: str
    name: str
    path: str
    class_id: str
    class_color: str
    mask_path: Optional[str] = None
    _pixmap_cache: Optional[QPixmap] = field(default=None, init=False, repr=False)
    _mask_pixmap_cache: Optional[QPixmap] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization processing. Can be used to validate paths or preprocess data.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Image path does not exist: {self.path}")

        if self.mask_path:
            if not os.path.exists(self.mask_path):
                print(
                    f"Warning: Mask path does not exist for sample {self.id}: {self.mask_path}"
                )

    def load_pixmap(self) -> Optional[QPixmap]:
        """Loads the image as a QPixmap, handling TIFFs."""
        if self._pixmap_cache:
            return self._pixmap_cache

        pixmap = self._load_image_to_qpixmap(self.path)
        if pixmap is None:
            return None  # Or return a placeholder QPixmap

        self._pixmap_cache = pixmap
        return pixmap

    def _load_image_to_qpixmap(self, image_path: str) -> Optional[QPixmap]:
        if not image_path or not os.path.exists(image_path):
            return None

        pixmap = None
        if image_path.lower().endswith((".tif", ".tiff")):
            try:
                img_data_tiff = tifffile.imread(image_path)

                # Handle multi-page TIFFs (take first page/frame)
                if (
                    img_data_tiff.ndim > 2
                    and img_data_tiff.shape[0] > 1
                    and img_data_tiff.ndim == 3
                ):  # (pages/channels, H, W)
                    # Heuristic: if first dim small (<=4), assume channels (C,H,W), else stack of 2D (P,H,W)
                    if (
                        img_data_tiff.shape[0] <= 4
                        and img_data_tiff.shape[0] != img_data_tiff.shape[1]
                        and img_data_tiff.shape[0] != img_data_tiff.shape[2]
                    ):  # (C,H,W)
                        img_data_tiff = np.moveaxis(
                            img_data_tiff, 0, -1
                        )  # Convert to (H,W,C)
                    else:  # (P,H,W)
                        img_data_tiff = img_data_tiff[0]
                elif (
                    img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1
                ):  # (pages, H, W, C)
                    img_data_tiff = img_data_tiff[0]  # Take first frame

                # Normalize to 0-255 uint8 if necessary for QImage conversion
                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        min_v, max_v = np.min(img_data_tiff), np.max(img_data_tiff)
                        if max_v > min_v:
                            img_data_tiff = (
                                (img_data_tiff - min_v) / (max_v - min_v + 1e-9)
                            ) * 255
                        else:
                            img_data_tiff = (
                                np.zeros_like(img_data_tiff)
                                if min_v == 0
                                else np.full_like(img_data_tiff, 128)
                            )
                    elif img_data_tiff.max() > 255:  # e.g. uint16
                        img_data_tiff = img_data_tiff / (
                            img_data_tiff.max() / 255.0
                        )  # Avoids clipping high values if max is just over 255
                    img_data_tiff = img_data_tiff.astype(np.uint8)

                # Convert NumPy array to QImage
                height, width = img_data_tiff.shape[:2]
                bytes_per_line = (
                    width * img_data_tiff.shape[2] if img_data_tiff.ndim == 3 else width
                )

                q_image_format = QImage.Format_Grayscale8  # Default
                if img_data_tiff.ndim == 3:
                    if img_data_tiff.shape[-1] == 3:  # RGB
                        q_image_format = QImage.Format_RGB888
                    elif img_data_tiff.shape[-1] == 4:  # RGBA
                        q_image_format = QImage.Format_RGBA8888
                    elif img_data_tiff.shape[-1] == 1:  # Grayscale (H,W,1)
                        img_data_tiff = img_data_tiff.squeeze(axis=-1)  # Make it (H,W)
                        bytes_per_line = width  # Recalculate for 2D
                        q_image_format = QImage.Format_Grayscale8
                    else:
                        return None
                elif img_data_tiff.ndim == 2:  # Grayscale (H,W)
                    q_image_format = QImage.Format_Grayscale8
                else:
                    return None

                if not img_data_tiff.flags["C_CONTIGUOUS"]:
                    img_data_tiff = np.ascontiguousarray(img_data_tiff)

                q_img = QImage(
                    img_data_tiff.data, width, height, bytes_per_line, q_image_format
                )
                if q_img.isNull():
                    return None
                pixmap = QPixmap.fromImage(q_img)
            except Exception as e:
                return None
        else:
            pixmap = QPixmap(image_path)

        if pixmap and pixmap.isNull():
            return None
        return pixmap

    def load_mask_pixmap(self) -> Optional[QPixmap]:
        """Loads the mask image as a QPixmap."""
        if not self.mask_path:
            # logging.debug(f"ImageCard ID {self.id}: No mask_path provided.") # More for debug
            return None

        if (
            self._mask_pixmap_cache and not self._mask_pixmap_cache.isNull()
        ):  # Check if cache is valid
            return self._mask_pixmap_cache

        if not os.path.exists(self.mask_path):
            logging.warning(
                f"ImageCard ID {self.id}: Mask path does not exist: {self.mask_path}"
            )
            return None

        pixmap = QPixmap(self.mask_path)
        if pixmap.isNull():
            logging.warning(
                f"ImageCard ID {self.id}: Failed to load QPixmap from mask_path: {self.mask_path}"
            )
            # Potentially try QImageReader for more robust loading if QPixmap fails directly
            # reader = QImageReader(self.mask_path)
            # if reader.canRead():
            #     image = reader.read()
            #     if not image.isNull():
            #         pixmap = QPixmap.fromImage(image)
            #     else:
            #         logging.warning(f"ImageCard ID {self.id}: QImageReader also failed for mask_path: {self.mask_path}, error: {reader.errorString()}")
            #         return None
            # else:
            #     logging.warning(f"ImageCard ID {self.id}: QImageReader cannot read mask_path: {self.mask_path}, error: {reader.errorString()}")
            #     return None

            # If QPixmap(path) failed, it's likely a critical issue with the file or path.
            return None

        self._mask_pixmap_cache = pixmap
        return pixmap
