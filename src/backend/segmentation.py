# backend/segmentation.py
import logging
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort
import tifffile
from PySide6.QtCore import QObject, Signal  # Removed QThread as this is just the model
from scipy import ndimage
from scipy.spatial import ConvexHull


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SegmentationModel(QObject):
    progress_updated = Signal(int)

    def __init__(self, model_path: str = None):
        super().__init__()
        self.session = None
        self.input_name = None
        self.output_name = None

        if model_path and os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logging.info(f"Segmentation ONNX model loaded from: {model_path}")
            except Exception as e:
                logging.error(
                    f"Failed to load ONNX segmentation model from {model_path}: {e}",
                    exc_info=True,
                )
        else:
            logging.warning(
                f"ONNX segmentation model path not found or not specified: {model_path}. Model-based segmentation disabled."
            )

    def _load_image_for_segmentation(
        self, image_path: str, target_format: str = "bgr"
    ) -> Optional[np.ndarray]:
        """
        Loads an image, handling TIFFs, and converts to the target format (BGR or Grayscale).
        target_format: "bgr" or "gray"
        """
        try:
            image_data = None
            if image_path.lower().endswith((".tif", ".tiff")):
                img_data_tiff = tifffile.imread(image_path)

                if (
                    img_data_tiff.ndim > 2
                    and img_data_tiff.shape[0] > 1
                    and img_data_tiff.ndim == 3
                ):
                    if img_data_tiff.shape[0] <= 4:
                        img_data_tiff = np.moveaxis(img_data_tiff, 0, -1)
                    else:
                        img_data_tiff = img_data_tiff[0]
                        logging.info(f"Multi-page TIFF {image_path}, using first page.")
                elif img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1:
                    img_data_tiff = img_data_tiff[0]
                    logging.info(f"Multi-page TIFF {image_path}, using first page.")

                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        min_val, max_val = np.min(img_data_tiff), np.max(img_data_tiff)
                        if max_val > min_val:
                            img_data_tiff = (
                                (img_data_tiff - min_val) / (max_val - min_val + 1e-9)
                            ) * 255
                        else:
                            img_data_tiff = (
                                np.zeros_like(img_data_tiff)
                                if min_val == 0
                                else np.full_like(img_data_tiff, 128)
                            )
                    elif img_data_tiff.max() > 255:
                        img_data_tiff = img_data_tiff / (img_data_tiff.max() / 255.0)
                    img_data_tiff = img_data_tiff.astype(np.uint8)

                # Convert to target format
                if target_format == "bgr":
                    if img_data_tiff.ndim == 2:
                        image_data = cv2.cvtColor(img_data_tiff, cv2.COLOR_GRAY2BGR)
                    elif img_data_tiff.ndim == 3:
                        if img_data_tiff.shape[-1] == 1:
                            image_data = cv2.cvtColor(img_data_tiff, cv2.COLOR_GRAY2BGR)
                        elif img_data_tiff.shape[-1] == 3:
                            image_data = cv2.cvtColor(img_data_tiff, cv2.COLOR_RGB2BGR)
                        elif img_data_tiff.shape[-1] == 4:
                            image_data = cv2.cvtColor(img_data_tiff, cv2.COLOR_RGBA2BGR)
                        else:
                            logging.error(
                                f"Unsupported TIFF channel count for BGR: {img_data_tiff.shape[-1]} in {image_path}"
                            )
                            return None
                    else:
                        logging.error(
                            f"Unsupported TIFF dimensions for BGR: {img_data_tiff.ndim} in {image_path}"
                        )
                        return None
                elif target_format == "gray":
                    if img_data_tiff.ndim == 2:
                        image_data = img_data_tiff
                    elif img_data_tiff.ndim == 3:
                        if img_data_tiff.shape[-1] == 1:
                            image_data = img_data_tiff.squeeze(axis=-1)
                        elif img_data_tiff.shape[-1] == 3:  # RGB from tifffile
                            bgr_temp = cv2.cvtColor(img_data_tiff, cv2.COLOR_RGB2BGR)
                            image_data = cv2.cvtColor(bgr_temp, cv2.COLOR_BGR2GRAY)
                        elif img_data_tiff.shape[-1] == 4:  # RGBA from tifffile
                            bgr_temp = cv2.cvtColor(img_data_tiff, cv2.COLOR_RGBA2BGR)
                            image_data = cv2.cvtColor(bgr_temp, cv2.COLOR_BGR2GRAY)
                        else:
                            logging.error(
                                f"Unsupported TIFF channel count for Grayscale: {img_data_tiff.shape[-1]} in {image_path}"
                            )
                            return None
                    else:
                        logging.error(
                            f"Unsupported TIFF dimensions for Grayscale: {img_data_tiff.ndim} in {image_path}"
                        )
                        return None
                else:
                    logging.error(f"Unsupported target_format: {target_format}")
                    return None
            else:  # Non-TIFF
                if target_format == "bgr":
                    # Load as color, then ensure it's 3-channel BGR
                    img_loaded_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img_loaded_color is None:
                        logging.error(
                            f"cv2.imread failed for (BGR target): {image_path}"
                        )
                        return None
                    if (
                        img_loaded_color.ndim == 2
                    ):  # If IMREAD_COLOR still gave grayscale (rare, but safeguard)
                        logging.warning(
                            f"Image {image_path} loaded as grayscale by IMREAD_COLOR. Converting to BGR."
                        )
                        image_data = cv2.cvtColor(img_loaded_color, cv2.COLOR_GRAY2BGR)
                    elif (
                        img_loaded_color.ndim == 3 and img_loaded_color.shape[-1] == 4
                    ):  # BGRA
                        logging.warning(
                            f"Image {image_path} loaded as BGRA by IMREAD_COLOR. Converting to BGR."
                        )
                        image_data = cv2.cvtColor(img_loaded_color, cv2.COLOR_BGRA2BGR)
                    elif (
                        img_loaded_color.ndim == 3 and img_loaded_color.shape[-1] == 3
                    ):  # Correct BGR
                        image_data = img_loaded_color
                    else:
                        logging.error(
                            f"Unexpected image format from IMREAD_COLOR for {image_path}: shape {img_loaded_color.shape}"
                        )
                        return None
                elif target_format == "gray":
                    image_loaded_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image_loaded_gray is None:
                        logging.error(
                            f"cv2.imread failed for (gray target): {image_path}"
                        )
                        return None
                    # Ensure it's 2D if grayscale
                    if image_loaded_gray.ndim == 3 and image_loaded_gray.shape[-1] == 1:
                        image_data = image_loaded_gray.squeeze(axis=-1)
                    elif image_loaded_gray.ndim == 2:
                        image_data = image_loaded_gray
                    else:
                        logging.error(
                            f"Unexpected image format from IMREAD_GRAYSCALE for {image_path}: shape {image_loaded_gray.shape}"
                        )
                        return None
                else:
                    logging.error(
                        f"Unsupported target_format: {target_format} for non-TIFF."
                    )
                    return None

            return image_data
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logging.error(
                f"Error loading image {image_path} for segmentation: {e}", exc_info=True
            )
            return None

    def predict_mask_onnx(self, image_path: str) -> Optional[np.ndarray]:
        """
        Predicts a mask using the loaded ONNX model.
        Returns a binary NumPy array (H, W) or None if prediction fails.
        """
        if not self.session or not self.input_name or not self.output_name:
            logging.error("ONNX session not initialized. Cannot predict mask.")
            return None

        try:
            image = self._load_image_for_segmentation(image_path, target_format="bgr")
            if image is None:
                return None

            original_height, original_width = image.shape[:2]

            # Preprocess (example for a model expecting 224x224 RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (224, 224))  # Example size
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_batched = np.expand_dims(
                image_normalized, axis=0
            )  # Add batch dimension

            self.progress_updated.emit(20)  # Example progress

            # Run inference
            raw_mask_output = self.session.run(
                [self.output_name], {self.input_name: image_batched}
            )[0]
            self.progress_updated.emit(70)

            if (
                raw_mask_output.ndim == 4 and raw_mask_output.shape[-1] == 1
            ):  # (B, H, W, C)
                mask_squeezed = raw_mask_output.squeeze()  # Becomes (H,W)
            elif raw_mask_output.ndim == 3:  # (B,H,W)
                mask_squeezed = raw_mask_output.squeeze(axis=0)
            else:  # Adapt as needed
                mask_squeezed = raw_mask_output.squeeze()

            # Threshold (common for sigmoid output)
            binary_mask_model_size = (mask_squeezed > 0.5).astype(np.uint8)

            # Resize mask back to original image dimensions
            binary_mask_original_size = cv2.resize(
                binary_mask_model_size,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,  # Use nearest for binary masks
            )
            self.progress_updated.emit(100)
            return (
                binary_mask_original_size  # Should be (original_height, original_width)
            )

        except Exception as e:
            logging.error(
                f"Error during ONNX mask prediction for {image_path}: {e}",
                exc_info=True,
            )
            self.progress_updated.emit(100)  # Ensure progress completes
            return None

    def predict_mask_otsu(
        self,
        image_path: str,
        grayscale: bool = True,
        max_distance_ratio: float = 0.3,
        min_component_size: int = 10,
    ) -> Optional[np.ndarray]:
        """
        Create a binary mask using Otsu's thresholding.
        Returns a binary NumPy array (H, W) or None if fails.
        """
        try:
            image = self._load_image_for_segmentation(image_path, target_format="gray")
            if image is None:
                return None

            if grayscale and len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # OpenCV reads BGR
            elif len(image.shape) == 2:
                gray = image
            else:  # Already grayscale or unexpected format
                gray = (
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if len(image.shape) == 3
                    else image.copy()
                )

            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )  # Common to use THRESH_BINARY_INV

            # Post-processing to keep components near center (as in original)
            labeled_mask, num_features = ndimage.label(
                mask // 255
            )  # Convert to 0/1 for label
            height, width = mask.shape
            center_y, center_x = height / 2, width / 2
            image_diagonal = np.sqrt(height**2 + width**2)
            max_dist = max_distance_ratio * image_diagonal

            mask_selected = np.zeros_like(mask, dtype=np.uint8)
            for component in range(1, num_features + 1):
                component_mask_pixels = labeled_mask == component
                if np.any(component_mask_pixels):
                    component_size = np.sum(component_mask_pixels)
                    if component_size < min_component_size:
                        continue
                    centroid = ndimage.center_of_mass(
                        component_mask_pixels
                    )  # Use on binary component_mask_pixels
                    distance = np.sqrt(
                        (centroid[0] - center_y) ** 2 + (centroid[1] - center_x) ** 2
                    )
                    if distance <= max_dist:
                        mask_selected[component_mask_pixels] = 255  # Keep as 255

            return mask_selected.astype(np.uint8)  # Return as 0 or 255

        except Exception as e:
            logging.error(
                f"Error in predict_mask_otsu for {image_path}: {e}", exc_info=True
            )
            return None

    def predict_mask_adaptive(
        self,
        image_path: str,
        grayscale: bool = True,
        block_size: int = 35,
        c_value: int = 10,
    ) -> Optional[np.ndarray]:
        """
        Create a binary mask using adaptive thresholding.
        Returns a binary NumPy array (H, W) or None if fails.
        """
        try:
            image = self._load_image_for_segmentation(image_path, target_format="gray")
            if image is None:
                return None

            if grayscale and len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray = image
            else:
                gray = (
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if len(image.shape) == 3
                    else image.copy()
                )

            # Ensure block_size is odd and >= 3
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1

            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # Often THRESH_BINARY_INV for objects
                block_size,
                c_value,
            )
            return mask.astype(np.uint8)  # Already 0 or 255
        except Exception as e:
            logging.error(
                f"Error in predict_mask_adaptive for {image_path}: {e}", exc_info=True
            )
            return None

    def predict_mask_watershed(
        self,
        image_path: str,
        grayscale: bool = True,
        foreground_threshold_ratio: float = 0.7,
        morph_kernel_size_val: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Create a binary mask using the watershed algorithm.
        Returns a binary NumPy array (H, W) or None if fails.
        """
        try:
            image_bgr = self._load_image_for_segmentation(
                image_path, target_format="bgr"
            )
            if image_bgr is None:
                return None

            if grayscale:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                # If not grayscale, watershed needs a 3-channel image.
                # If original was already gray, convert it to BGR for watershed.
                if len(image_bgr.shape) == 2:
                    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
                gray = cv2.cvtColor(
                    image_bgr, cv2.COLOR_BGR2GRAY
                )  # Still need gray for thresholding steps

            # Noise removal
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )  # Invert for objects

            # Sure background area
            kernel = np.ones((morph_kernel_size_val, morph_kernel_size_val), np.uint8)
            sure_bg = cv2.dilate(thresh, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(
                dist_transform,
                foreground_threshold_ratio * dist_transform.max(),
                255,
                0,
            )
            sure_fg = np.uint8(sure_fg)

            # Finding unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labeling
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = (
                markers + 1
            )  # Add 1 to all labels so that sure background is not 0, but 1
            markers[unknown == 255] = 0  # Mark the region of unknown with zero

            # Apply watershed
            markers = cv2.watershed(
                image_bgr, markers
            )  # Use original BGR image for watershed

            # Create final mask: regions with markers > 1 are segmented objects
            final_mask = np.zeros_like(gray, dtype=np.uint8)
            final_mask[markers > 1] = (
                255  # Watershed boundaries are -1, objects are > 1
            )

            return final_mask

        except Exception as e:
            logging.error(
                f"Error in predict_mask_watershed for {image_path}: {e}", exc_info=True
            )
            return None

    def get_valid_contours(
        self, contours, image_area: int, min_area_ratio: float = 0.001
    ) -> list:  # Changed image_size to image_area
        """Filters contours by area relative to the image area."""
        if image_area == 0:
            return []
        min_contour_area = min_area_ratio * image_area
        return [c for c in contours if cv2.contourArea(c) > min_contour_area]

    def get_object_properties(
        self,
        image_path: str,
        mask: np.ndarray,
        min_area_ratio: float = 0.001,
        scale_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute object properties from an image and its binary mask.
        Mask is expected to be a 2D NumPy array (H, W) with 0 for background and non-zero (e.g., 255) for foreground.
        """
        default_properties = {
            "area": 0.0,
            "perimeter": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "aspect_ratio": 0.0,
            "circularity": 0.0,
            "major_axis_length": 0.0,
            "minor_axis_length": 0.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "compactness": 0.0,
            "convexity": 0.0,
            "curl": 0.0,
            "volume": 0.0,
        }

        if mask is None or mask.size == 0:
            logging.warning(
                f"Empty or None mask provided for property calculation for {image_path}."
            )
            return default_properties.copy()

        # Ensure mask is binary (0 or 255, uint8)
        if mask.dtype != np.uint8 or set(np.unique(mask)).difference({0, 255}):
            binary_mask_u8 = (mask > 0).astype(np.uint8) * 255
        else:
            binary_mask_u8 = mask

        try:
            image = self._load_image_for_segmentation(image_path, target_format="gray")
            if image is None:
                logging.warning(
                    f"Could not load image for property calculation: {image_path}"
                )
                return default_properties.copy()

            if image.shape[:2] != binary_mask_u8.shape:
                # This shouldn't happen if masks are resized to original image dimensions
                logging.warning(
                    f"Image and mask dimensions do not match for {image_path}. Image: {image.shape[:2]}, Mask: {binary_mask_u8.shape}. Skipping property calc."
                )
                return default_properties.copy()

            if image.ndim == 2:  # shape (H, W)
                image_gray = image
            elif image.ndim == 3 and image.shape[-1] == 1:
                image_gray = image.squeeze(-1)  # shape (H, W, 1) → (H, W)
            elif image.ndim == 3 and image.shape[-1] == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 3 and image.shape[-1] == 4:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                logging.error(
                    f"Unsupported channel count ({image.shape}) for {image_path}"
                )
                return default_properties.copy()

            image_area = image.shape[0] * image.shape[1]

        except Exception as e:
            logging.error(
                f"Error loading image or converting to grayscale for {image_path}: {e}"
            )
            return default_properties.copy()

        try:
            contours, _ = cv2.findContours(
                binary_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception as e:
            logging.error(f"Error finding contours for {image_path}: {e}")
            return default_properties.copy()

        valid_contours = self.get_valid_contours(contours, image_area, min_area_ratio)
        if not valid_contours:
            return default_properties.copy()

        all_points_list = [c.reshape(-1, 2) for c in valid_contours]
        if not all_points_list:  # Should be redundant due to valid_contours check
            return default_properties.copy()

        all_points = np.concatenate(all_points_list, axis=0)

        combined_mask_for_intensity = np.zeros_like(binary_mask_u8)
        cv2.drawContours(
            combined_mask_for_intensity, valid_contours, -1, 255, thickness=cv2.FILLED
        )

        properties = default_properties.copy()
        try:
            properties["area"] = float(
                cv2.contourArea(all_points)
            )  # Area of the convex hull of all points, or sum?
        except:
            pass
        try:
            properties["perimeter"] = float(
                cv2.arcLength(all_points, True)
            )  # Perimeter of the convex hull
        except:
            pass

        if (
            properties["area"] > 0 and all_points.shape[0] >= 3
        ):  # Convex hull needs at least 3 points
            try:
                hull = ConvexHull(all_points)
                hull_area = float(hull.volume)  # In 2D, volume is area
                properties["solidity"] = (
                    properties["area"] / hull_area if hull_area > 0 else 0.0
                )

                hull_points_indices = hull.vertices
                hull_contour_points = all_points[hull_points_indices].astype(
                    np.int32
                )  # Needs to be int32 for arcLength
                hull_perimeter = float(cv2.arcLength(hull_contour_points, True))
                properties["convexity"] = (
                    properties["perimeter"] / hull_perimeter
                    if hull_perimeter > 0
                    else 0.0
                )
            except:
                pass  # ConvexHull can fail for collinear points

        if all_points.shape[0] >= 5:  # fitEllipse needs at least 5 points
            try:
                ellipse = cv2.fitEllipse(all_points)
                properties["major_axis_length"] = float(max(ellipse[1]))
                properties["minor_axis_length"] = float(min(ellipse[1]))
                if properties["minor_axis_length"] > 0:
                    properties["eccentricity"] = float(
                        np.sqrt(
                            1
                            - (
                                properties["minor_axis_length"] ** 2
                                / properties["major_axis_length"] ** 2
                            )
                        )
                    )
            except:
                pass

        try:
            x, y, w, h = cv2.boundingRect(all_points)
            properties["aspect_ratio"] = float(w) / h if h > 0 else 0.0
        except:
            pass

        if properties["perimeter"] > 0:
            try:
                properties["circularity"] = (
                    4 * np.pi * (properties["area"] / (properties["perimeter"] ** 2))
                )
            except:
                pass
            try:
                properties["compactness"] = (
                    (properties["perimeter"] ** 2) / (4 * np.pi * properties["area"])
                    if properties["area"] > 0
                    else 0.0
                )
            except:
                pass
            try:
                properties["curl"] = (
                    properties["perimeter"] / (2 * np.sqrt(np.pi * properties["area"]))
                    if properties["area"] > 0
                    else 0.0
                )
            except:
                pass

        try:
            masked_pixels = image_gray[combined_mask_for_intensity == 255]
            if masked_pixels.size > 0:
                properties["mean_intensity"] = float(np.mean(masked_pixels))
                properties["std_intensity"] = float(np.std(masked_pixels))
        except:
            pass

        # Volume: A simple approximation if we assume roughly spherical/cylindrical shape from area/perimeter
        try:
            # Assuming circular cross-section, diameter from perimeter
            diameter_approx = (
                properties["perimeter"] / np.pi if properties["perimeter"] > 0 else 0
            )
            # Simple spherical volume approximation
            properties["volume"] = (
                (4 / 3) * np.pi * (diameter_approx / 2) ** 3
                if diameter_approx > 0
                else 0.0
            )
        except:
            pass

        if scale_factor is not None and scale_factor > 0:
            properties["area"] *= scale_factor**2
            properties["perimeter"] *= scale_factor
            properties["major_axis_length"] *= scale_factor
            properties["minor_axis_length"] *= scale_factor
            properties["volume"] *= scale_factor**3
            logging.debug(
                f"Applied scale_factor {scale_factor} to properties for {image_path}"
            )

        # Ensure all values are float for consistency, handle NaN/inf
        for key, value in properties.items():
            if isinstance(value, (float, int)):
                if np.isnan(value) or np.isinf(value):
                    properties[key] = 0.0
                else:
                    properties[key] = float(value)
            else:  # Should not happen if defaults are floats
                properties[key] = 0.0

        return properties
