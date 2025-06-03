from concurrent.futures import ThreadPoolExecutor, as_completed

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImageReader

from backend.config import SAMPLE_RES_SCALE


class CreateCardsThread(QThread):
    progress_updated = Signal(int)
    pixmaps_loaded = Signal(dict)  # Signal to emit the loaded pixmaps

    def __init__(self, images, gallery_presenter, max_workers=None):
        """
        Initializes the thread.

        :param images: Dictionary of image objects to process.
        :param gallery_presenter: Reference to the gallery presenter.
        :param max_workers: Maximum number of threads to use.
        """
        super().__init__()
        self.images = images
        self.gallery_presenter = gallery_presenter
        self.max_workers = max_workers or 4  # Default to 4 threads
        self.thumbnail_scale = self.gallery_presenter.thumbnail_quality / 100

    def run(self):
        """Loads image pixmaps in a separate thread using ThreadPoolExecutor."""
        pixmaps = {}  # Dictionary to store image_id: QPixmap pairs
        total_images = len(self.images)

        if total_images == 0:
            self.pixmaps_loaded.emit(pixmaps)
            return

        def load_and_scale(image):
            """
            Loads and scales a single image.

            :param image: Image object containing the path and ID.
            :return: Tuple of (image_id, QPixmap or None).
            """
            reader = QImageReader(image.path)
            if not reader.canRead():
                print(f"Cannot read image: {image.path}")
                return image.id, None
            q_image = reader.read()
            if q_image.isNull():
                print(f"Failed to load image: {image.path}")
                return image.id, None
            # Rescale the image
            new_width = int(q_image.width() * self.thumbnail_scale)
            new_height = int(q_image.height() * self.thumbnail_scale)
            q_image = q_image.scaled(
                new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            return image.id, q_image

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all image processing tasks to the executor
            future_to_image = {
                executor.submit(load_and_scale, image): image
                for image in self.images.values()
            }
            for i, future in enumerate(as_completed(future_to_image), 1):
                image_id, q_pixmap = future.result()
                if q_pixmap is not None:
                    pixmaps[image_id] = q_pixmap
                # Update progress
                progress = int(i / total_images * 100)
                self.progress_updated.emit(progress)

        self.pixmaps_loaded.emit(pixmaps)  # Emit the loaded pixmaps
