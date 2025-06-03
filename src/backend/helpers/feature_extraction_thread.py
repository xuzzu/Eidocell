from PySide6.QtCore import QThread, Signal


class FeatureExtractionThread(QThread):
    """Thread for extracting image features in the background."""

    progress_updated = Signal(int)

    def __init__(self, image_ids, data_manager):
        super().__init__()
        self.image_ids = image_ids
        self.data_manager = data_manager

    def run(self):
        """Extracts features for each image ID."""
        total_images = len(self.image_ids)
        for i, image_id in enumerate(self.image_ids):
            self.data_manager.extract_and_set_features(image_id)
            progress = int((i + 1) / total_images * 100)
            self.progress_updated.emit(progress)
        self.data_manager._update_objects_metadata()
        self.data_manager._update_features_metadata()
