# backend/workers.py
import os

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from backend.objects.mask import Mask
from backend.objects.sample import Sample
from backend.utils.file_utils import read_json


class WorkerSignals(QObject):
    progress = Signal(int)  # Emitting progress percentage
    finished = Signal()  # Emitting when the worker is done
    error = Signal(str)  # Emitting error messages
    result = Signal(object)  # Emitting the result data


# backend/workers.py (continued)
class ImageLoaderWorker(QRunnable):
    def __init__(self, session, signals):
        super().__init__()
        self.session = session
        self.signals = signals

    @Slot()
    def run(self):
        try:
            objects_metadata_path = os.path.join(
                self.session.metadata_directory, "objects_metadata.json"
            )
            objects_data = read_json(objects_metadata_path)
            images = {}
            total_images = len(objects_data.get("objects", []))
            for i, obj in enumerate(objects_data.get("objects", []), 1):
                try:
                    image = Sample.from_dict(obj)
                    images[image.id] = image
                except Exception as e:
                    error_msg = f"Error loading image object: {e}"
                    print(error_msg)
                    self.signals.error.emit(error_msg)
                progress = int(i / total_images * 100)
                self.signals.progress.emit(progress)
            self.signals.result.emit(images)
        except Exception as e:
            error_msg = f"Exception in ImageLoaderWorker: {e}"
            print(error_msg)
            self.signals.error.emit(error_msg)
        finally:
            self.signals.finished.emit()


class FeatureLoaderWorker(QRunnable):
    def __init__(self, session, images, signals):
        super().__init__()
        self.session = session
        self.images = images  # Dictionary of Image objects
        self.signals = signals

    @Slot()
    def run(self):
        try:
            features_metadata_path = os.path.join(
                self.session.metadata_directory, "features_metadata.json"
            )
            features_data = read_json(features_metadata_path)
            features = {}
            total_features = len(features_data.get("features", []))
            for i, feature_entry in enumerate(features_data.get("features", []), 1):
                image_id = feature_entry["image_id"]
                feature_path = feature_entry["path"]
                image = self.images.get(image_id)
                if image and os.path.exists(feature_path):
                    try:
                        image.features = np.load(feature_path)
                        features[image_id] = feature_path
                    except Exception as e:
                        error_msg = (
                            f"Error loading feature for Image ID {image_id}: {e}"
                        )
                        print(error_msg)
                        self.signals.error.emit(error_msg)
                else:
                    error_msg = f"Feature file does not exist for Image ID {image_id}: {feature_path}"
                    print(error_msg)
                    self.signals.error.emit(error_msg)
                progress = int(i / total_features * 100)
                self.signals.progress.emit(progress)
            self.signals.result.emit(features)
        except Exception as e:
            error_msg = f"Exception in FeatureLoaderWorker: {e}"
            print(error_msg)
            self.signals.error.emit(error_msg)
        finally:
            self.signals.finished.emit()


class MaskLoaderWorker(QRunnable):
    def __init__(self, session, images, signals):
        super().__init__()
        self.session = session
        self.images = images  # Dictionary of Image objects
        self.signals = signals

    @Slot()
    def run(self):
        try:
            masks_metadata_path = os.path.join(
                self.session.metadata_directory, "masks_metadata.json"
            )
            masks_data = read_json(masks_metadata_path)
            masks = {}
            total_masks = len(masks_data.get("masks", []))
            for i, mask_entry in enumerate(masks_data.get("masks", []), 1):
                try:
                    mask = Mask.from_dict(mask_entry)
                    if os.path.exists(mask.path):
                        mask.mask_data = np.load(mask.path)
                        if mask.masked_image_path:
                            mask.masked_image = cv2.imread(mask.masked_image_path)
                        masks[mask.id] = mask
                    else:
                        error_msg = f"Mask file does not exist: {mask.path}"
                        print(error_msg)
                        self.signals.error.emit(error_msg)
                except Exception as e:
                    error_msg = f"Error loading mask object: {e}"
                    print(error_msg)
                    self.signals.error.emit(error_msg)
                progress = int(i / total_masks * 100)
                self.signals.progress.emit(progress)
            self.signals.result.emit(masks)
        except Exception as e:
            error_msg = f"Exception in MaskLoaderWorker: {e}"
            print(error_msg)
            self.signals.error.emit(error_msg)
        finally:
            self.signals.finished.emit()
