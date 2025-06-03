import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from backend.segmentation import SegmentationModel
from backend.utils.image_utils import combine_image_and_mask

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SegmentationThread(QThread):
    progress_updated = Signal(int)

    # full‑apply mode
    raw_mask_generated = Signal(str, object, object, str)

    # preview mode
    preview_mask_ready = Signal(str, object)

    def __init__(
        self,
        image_ids: List[str],
        image_paths: List[str],
        masked_images_preview_dir: str,
        segmentation_model_instance: SegmentationModel,
        segmentation_method_name: str = "Otsu's Thresholding",
        param1_value: Optional[Any] = None,
        param2_value: Optional[Any] = None,
        is_preview_run: bool = True,
        session_scale_factor: Optional[float] = None,
    ):
        super().__init__()

        if len(image_ids) != len(image_paths):
            raise ValueError("Length of image_ids and image_paths must match.")

        self.image_ids = image_ids
        self.image_paths = image_paths
        self.masked_images_preview_dir = masked_images_preview_dir
        self.segmentation_model = segmentation_model_instance
        self.method = segmentation_method_name
        self.param1 = param1_value
        self.param2 = param2_value
        self.is_preview_run = is_preview_run
        self.scale_factor = session_scale_factor or 1.0

        self._mutex = QMutex()
        self._stop_requested = False

    def requestInterruption(self):
        with QMutexLocker(self._mutex):
            self._stop_requested = True

    def _interrupted(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stop_requested

    def _segment_single(self, idx: int, img_id: str, img_path: str):
        if self._interrupted():
            return None  # caller will skip

        if self.method == "Otsu's Thresholding":
            raw_mask = self.segmentation_model.predict_mask_otsu(
                img_path,
                max_distance_ratio=(self.param1 or 30) / 100,
                min_component_size=self.param2 or 10,
            )
        elif self.method == "Adaptive Thresholding":
            raw_mask = self.segmentation_model.predict_mask_adaptive(
                img_path, block_size=self.param1 or 35, c_value=self.param2 or 10
            )
        elif self.method == "Watershed":
            raw_mask = self.segmentation_model.predict_mask_watershed(
                img_path,
                foreground_threshold_ratio=(self.param1 or 70) / 100,
                morph_kernel_size_val=self.param2 or 3,
            )
        else:
            logging.error(f"Unknown segmentation method '{self.method}'.")
            return img_id, None, None, None

        if raw_mask is None:
            logging.warning(f"Segmentation failed for image {img_id}.")
            return img_id, None, None, None

        if self.is_preview_run:
            preview_np = combine_image_and_mask(img_path, raw_mask)
            return img_id, raw_mask, None, preview_np

        # full‑apply mode
        attrs = self.segmentation_model.get_object_properties(
            img_path, raw_mask, scale_factor=self.scale_factor
        )
        preview_path = ""
        if self.masked_images_preview_dir:
            os.makedirs(self.masked_images_preview_dir, exist_ok=True)
            preview_path = os.path.join(
                self.masked_images_preview_dir, f"masked_preview_{img_id}.png"
            )
            try:
                disp_np = combine_image_and_mask(img_path, raw_mask)
                if disp_np is not None:
                    cv2.imwrite(preview_path, cv2.cvtColor(disp_np, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logging.error(f"Failed to save preview for {img_id}: {e}")
                preview_path = ""
        return img_id, raw_mask, attrs, preview_path

    def run(self):
        total = len(self.image_ids)
        if not total:
            return

        max_workers = max(1, os.cpu_count() - 1)
        completed = 0

        logging.info(
            f"SegmentationThread started → method: {self.method}, "
            f"preview: {self.is_preview_run}, images: {total}, "
            f"workers: {max_workers}."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._segment_single, i, iid, ipath)
                for i, (iid, ipath) in enumerate(zip(self.image_ids, self.image_paths))
            ]

            for fut in as_completed(futures):
                if self._interrupted():
                    break  # Stop consuming results

                result = fut.result()
                if result is None:
                    continue  # interrupted during worker set‑up

                img_id, raw_mask, attrs, extra = result
                if raw_mask is None:
                    if self.is_preview_run:
                        self.preview_mask_ready.emit(img_id, None)
                    else:
                        self.raw_mask_generated.emit(img_id, None, {}, "")
                else:
                    if self.is_preview_run:
                        self.preview_mask_ready.emit(
                            img_id, extra
                        )  # extra is preview np
                    else:
                        self.raw_mask_generated.emit(
                            img_id, raw_mask, attrs, extra
                        )  # extra is path

                completed += 1
                self.progress_updated.emit(int(completed / total * 100))

        logging.info("SegmentationThread finished.")
