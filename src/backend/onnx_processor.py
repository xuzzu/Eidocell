# backend/onnx_processor.py
import logging
import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import tifffile
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ONNXFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        feature_dim: int,
        architecture: Optional[str],
        execution_provider: str,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.architecture = architecture
        self.execution_provider = execution_provider
        self.ort_session = None
        self._initialize_onnx_session()

    def _initialize_onnx_session(self):
        if not os.path.exists(self.model_path):
            logging.error(f"ONNX model file not found at: {self.model_path}")
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        try:
            available_providers = ort.get_available_providers()
            providers_to_try = [self.execution_provider]
            if self.execution_provider not in available_providers:
                logging.warning(
                    f"ONNX Provider '{self.execution_provider}' not available. Available: {available_providers}. Trying CPU."
                )
                providers_to_try = ["CPUExecutionProvider"]
            elif (
                "CPUExecutionProvider" not in providers_to_try
                and self.execution_provider != "CPUExecutionProvider"
            ):
                providers_to_try.append("CPUExecutionProvider")

            self.ort_session = ort.InferenceSession(
                self.model_path, providers=providers_to_try
            )
            logging.info(
                f"ONNX session initialized for model '{self.model_name}' with provider(s): {self.ort_session.get_providers()}"
            )
        except Exception as e:
            logging.error(
                f"Failed to initialize ONNX Runtime session for {self.model_path}: {e}",
                exc_info=True,
            )
            self.ort_session = None
            raise RuntimeError(
                f"Failed to initialize ONNX session for {self.model_name}"
            ) from e

    def _load_and_preprocess_image_onnx(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image_np = None
            if image_path.lower().endswith((".tif", ".tiff")):
                img_data_tiff = tifffile.imread(image_path)

                # Handle multi-page TIFFs (take first page)
                if (
                    img_data_tiff.ndim > 2
                    and img_data_tiff.shape[0] > 1
                    and img_data_tiff.ndim == 3
                ):  # (pages, H, W) or (pages, H, C)
                    # check if it's a stack of grayscale or if the first dim is channels
                    if (
                        img_data_tiff.shape[0] <= 4
                    ):  # Heuristic: if first dim small, assume it's channels (C,H,W)
                        img_data_tiff = np.moveaxis(
                            img_data_tiff, 0, -1
                        )  # Convert (C,H,W) to (H,W,C)
                    else:  # Assume stack of 2D images
                        img_data_tiff = img_data_tiff[0]
                        logging.info(
                            f"Multi-page TIFF detected for {image_path}. Using first page."
                        )
                elif (
                    img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1
                ):  # (pages, H, W, C)
                    img_data_tiff = img_data_tiff[0]
                    logging.info(
                        f"Multi-page TIFF detected for {image_path}. Using first page."
                    )

                # Normalize to 0-255 if not already
                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        img_data_tiff = (
                            (img_data_tiff - np.min(img_data_tiff))
                            / (np.max(img_data_tiff) - np.min(img_data_tiff) + 1e-9)
                            * 255
                        )
                    elif img_data_tiff.max() > 255:  # e.g. uint16
                        img_data_tiff = (
                            img_data_tiff / (img_data_tiff.max() / 255.0)
                        ).astype(np.uint8)
                    img_data_tiff = img_data_tiff.astype(np.uint8)

                # Ensure 3 channels (RGB)
                if img_data_tiff.ndim == 2:  # Grayscale
                    image_np = cv2.cvtColor(img_data_tiff, cv2.COLOR_GRAY2RGB)
                elif img_data_tiff.ndim == 3:
                    if img_data_tiff.shape[-1] == 1:  # Grayscale with channel dim
                        image_np = cv2.cvtColor(img_data_tiff, cv2.COLOR_GRAY2RGB)
                    elif img_data_tiff.shape[-1] == 4:  # RGBA
                        image_np = cv2.cvtColor(img_data_tiff, cv2.COLOR_RGBA2RGB)
                    elif img_data_tiff.shape[-1] == 3:  # RGB
                        image_np = img_data_tiff
                    else:
                        logging.error(
                            f"Unsupported TIFF channel count for {image_path}: {img_data_tiff.shape[-1]}."
                        )
                        return None
                else:
                    logging.error(
                        f"Unsupported TIFF dimensions for {image_path}: {img_data_tiff.ndim}."
                    )
                    return None
            else:  # For non-TIFF files, use PIL
                pil_image = Image.open(image_path)
                if pil_image.mode == "RGBA":
                    background = Image.new("RGB", pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    rgb_image = background
                elif pil_image.mode != "RGB":
                    rgb_image = pil_image.convert("RGB")
                else:
                    rgb_image = pil_image
                image_np = np.array(rgb_image)

            return image_np
        except FileNotFoundError:
            logging.error(f"Image file not found at: {image_path}")
            return None
        except Exception as e:
            logging.error(
                f"Error reading or preprocessing image {image_path} for ONNX: {e}",
                exc_info=True,
            )
            return None

    def extract_features(self, image_path: str) -> np.ndarray:  # Returns np.ndarray
        if not self.ort_session:
            logging.error(f"ONNX session not ready for model {self.model_name}.")
            return np.zeros(self.feature_dim, dtype=np.float32)

        image_data_onnx = self._load_and_preprocess_image_onnx(image_path)
        if image_data_onnx is None:
            logging.warning(
                f"Preprocessing failed for ONNX model {self.model_name} with image: {image_path}"
            )
            return np.zeros(self.feature_dim, dtype=np.float32)

        input_name = self.ort_session.get_inputs()[0].name
        try:
            target_size = (224, 224)  # Default
            if self.architecture and "mobilevit" in self.architecture.lower():
                target_size = (256, 256)

            image_resized = cv2.resize(image_data_onnx, target_size)
            image_transposed = (
                image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            image_batched = image_transposed[np.newaxis, ...]

            outputs = self.ort_session.run(None, {input_name: image_batched})
            feature_vector_onnx = outputs[0].reshape(-1)
            if feature_vector_onnx.shape[0] != self.feature_dim:
                logging.error(
                    f"ONNX feature dimension mismatch for {self.model_name}. Expected {self.feature_dim}, got {feature_vector_onnx.shape[0]}. Returning zeros."
                )
                return np.zeros(self.feature_dim, dtype=np.float32)
            return feature_vector_onnx
        except Exception as e:
            logging.error(
                f"Error during ONNX feature extraction for {image_path} with model {self.model_name}: {e}",
                exc_info=True,
            )
            return np.zeros(self.feature_dim, dtype=np.float32)
