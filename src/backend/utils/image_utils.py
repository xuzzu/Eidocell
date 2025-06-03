import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageColor, ImageEnhance, ImageOps

from backend.config import CLASS_FRAME_PATH, SAMPLE_RES_SCALE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _resize_longest_edge(img: np.ndarray, target: int) -> np.ndarray:
    """Scale *img* so that its longest side equals *target* (antialiased)."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    if scale == 1.0:
        return img
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_poisson(img: np.ndarray, target: int) -> np.ndarray:
    """Return a *square* ``target×target`` image that contains *img* centred and
    blended into a background whose colour matches the border of *img*.

    *Idea*: identical to the stand‑alone ``pad_poisson`` function the user
    provided, but self‑contained (no external *TARGET* constant)."""
    img_r = _resize_longest_edge(img, target)
    h, w = img_r.shape[:2]

    #  background colour: median of outermost pixels -
    border = np.concatenate(
        [
            img_r[0, :, :],
            img_r[-1, :, :],  # top & bottom rows
            img_r[:, 0, :],
            img_r[:, -1, :],  # left & right cols
        ],
        axis=0,
    )
    bg_colour = np.median(border, axis=0).astype(np.uint8)

    background = np.full((target, target, 3), bg_colour, dtype=np.uint8)

    #  Poisson‑blend into the centre
    center = (target // 2, target // 2)
    mask = 255 * np.ones(img_r.shape[:2], dtype=np.uint8)
    blended = cv2.seamlessClone(img_r, background, mask, center, cv2.NORMAL_CLONE)
    return blended


def merge_images_collage(image_paths, margin=5, scale=1.0):
    """
    Merges images into a square-shaped collage efficiently using multithreading.

    Args:
        image_paths (list): A list of image file paths.
        margin (int, optional): Margin between images in pixels. Defaults to 5.
        scale (float, optional): Scale factor for resizing images. Defaults to 1.0.

    Returns:
        Image.Image: The merged collage image, or None if no valid images are found.
    """

    def load_and_scale_image(img_path):
        """Loads and scales an image, handling potential errors and TIFFs."""
        try:
            pil_img = None
            if img_path.lower().endswith((".tif", ".tiff")):
                img_data_tiff = tifffile.imread(img_path)

                if (
                    img_data_tiff.ndim > 2
                    and img_data_tiff.shape[0] > 1
                    and img_data_tiff.ndim == 3
                ):
                    if img_data_tiff.shape[0] <= 4:  # (C,H,W)
                        img_data_tiff = np.moveaxis(img_data_tiff, 0, -1)  # (H,W,C)
                    else:  # Stack of 2D
                        img_data_tiff = img_data_tiff[0]
                elif (
                    img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1
                ):  # (pages,H,W,C)
                    img_data_tiff = img_data_tiff[0]

                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        min_val, max_val = np.min(img_data_tiff), np.max(img_data_tiff)
                        if max_val > min_val:
                            img_data_tiff = (
                                (img_data_tiff - min_val) / (max_val - min_val)
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

                if img_data_tiff.ndim == 2:  # Grayscale
                    pil_img = Image.fromarray(img_data_tiff, mode="L").convert("RGB")
                elif img_data_tiff.ndim == 3:
                    if img_data_tiff.shape[-1] == 1:  # (H,W,1) Grayscale
                        pil_img = Image.fromarray(
                            img_data_tiff.squeeze(axis=-1), mode="L"
                        ).convert("RGB")
                    elif img_data_tiff.shape[-1] == 3:  # RGB
                        pil_img = Image.fromarray(img_data_tiff, mode="RGB")
                    elif img_data_tiff.shape[-1] == 4:  # RGBA
                        pil_img = Image.fromarray(img_data_tiff, mode="RGBA").convert(
                            "RGB"
                        )
                    else:
                        logging.error(
                            f"Unsupported TIFF channel count for collage: {img_data_tiff.shape[-1]} in {img_path}"
                        )
                        return None
                else:
                    logging.error(
                        f"Unsupported TIFF dimensions for collage: {img_data_tiff.ndim} in {img_path}"
                    )
                    return None
            else:  # Non-TIFF
                pil_img = Image.open(img_path).convert(
                    "RGB"
                )  # Ensure RGB for consistency

            if pil_img is None:
                return None

            if scale != 1.0:
                pil_img = pil_img.resize(
                    (int(pil_img.width * scale), int(pil_img.height * scale)),
                    Image.LANCZOS,
                )
            return pil_img
        except Exception as e:
            logging.error(
                f"Error loading image {img_path} for collage: {e}", exc_info=True
            )
            return None

    # -
    with ThreadPoolExecutor() as pool:
        images = list(pool.map(load_and_scale_image, image_paths))

    images = [im for im in images if im is not None]
    if not images:
        logging.error("No valid images for collage.")
        return None

    max_side = max(max(im.width, im.height) for im in images)

    padded: List[Image.Image] = []
    for im in images:
        arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # cv2 expects BGR
        padded_arr = _pad_poisson(arr, max_side)
        padded_im = Image.fromarray(cv2.cvtColor(padded_arr, cv2.COLOR_BGR2RGBA))
        padded.append(padded_im)

    num = len(padded)
    grid = math.ceil(math.sqrt(num))

    collage_w = (max_side + margin) * grid - margin
    collage_h = (max_side + margin) * grid - margin

    collage = Image.new("RGBA", (collage_w, collage_h), (255, 255, 255, 0))

    for idx, im in enumerate(padded):
        x_off = (idx % grid) * (max_side + margin)
        y_off = (idx // grid) * (max_side + margin)
        collage.paste(im, (x_off, y_off), im)

    return collage


def get_dominant_hue_pil(
    image: Image.Image,
    ignore_colors_near_gray=True,
    saturation_threshold=0.1,
    value_threshold=0.1,
):
    """
    Attempts to find a dominant hue in a PIL image, ignoring near-gray colors.
    Returns hue in degrees (0-360), or None if no dominant hue is found.
    """
    if image.mode not in ("RGB", "RGBA"):
        return None

    img_hsv = image.convert("HSV")
    hues = []

    width, height = img_hsv.size
    pixels_to_sample = min(width * height, 10000)  # Sample up to 10k pixels

    np_img_hsv = np.array(img_hsv)
    h_channel = np_img_hsv[:, :, 0]
    s_channel = np_img_hsv[:, :, 1] / 255.0  # Normalize S to 0-1
    v_channel = np_img_hsv[:, :, 2] / 255.0  # Normalize V to 0-1

    if ignore_colors_near_gray:
        mask = (
            (s_channel > saturation_threshold)
            & (v_channel > value_threshold)
            & (v_channel < (1.0 - value_threshold))
        )

        valid_hues_pil = h_channel[mask]  # PIL Hue is 0-255
    else:
        valid_hues_pil = h_channel.flatten()

    if valid_hues_pil.size == 0:
        return None

    hist, bin_edges = np.histogram(
        valid_hues_pil, bins=36, range=(0, 255)
    )  # 36 bins for 0-255 PIL hue
    dominant_bin_index = np.argmax(hist)

    dominant_hue_pil = (
        bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]
    ) / 2
    dominant_hue_degrees = (dominant_hue_pil / 255.0) * 360.0

    return dominant_hue_degrees


def tint_image(
    image: Image.Image,
    target_color_hex: str,
    original_frame_hue_degrees: Optional[float] = None,
    saturation_threshold=0.05,
    value_threshold=0.05,
) -> Image.Image:
    """
    Tints an image by shifting the hue of its colored parts towards a target color's hue,
    while preserving saturation, value, and alpha.
    Ignores near-grayscale pixels and the original alpha channel.

    Args:
        image (Image.Image): The PIL Image object to tint (expected to be RGBA).
        target_color_hex (str): The target tint color in hex (e.g., "#FF0000").
        original_frame_hue_degrees (Optional[float]): The hue (0-360) of the part of the frame
                                                      that should be tinted. If None, it will be
                                                      estimated from the frame.
        saturation_threshold (float): Pixels with saturation below this (0-1) are considered grayscale.
        value_threshold (float): Pixels with value (brightness) below this (0-1) are considered too dark.

    Returns:
        Image.Image: The tinted PIL Image object in RGBA format.
    """
    if image.mode != "RGBA":
        image_rgba = image.convert("RGBA")
    else:
        image_rgba = image.copy()  # Work on a copy

    try:
        target_rgb = ImageColor.getrgb(target_color_hex)
        target_hsv_pil = (
            Image.new("RGB", (1, 1), target_rgb).convert("HSV").getpixel((0, 0))
        )
        target_hue_pil = target_hsv_pil[0]  # PIL Hue is 0-255
    except ValueError:
        return image_rgba  # Return original if color is invalid

    img_hsv = image_rgba.convert("HSV")
    h_orig, s_orig, v_orig = img_hsv.split()
    alpha_channel = image_rgba.split()[3]  # Preserve original alpha

    # Convert to numpy arrays for efficient manipulation
    h_np = np.array(h_orig, dtype=np.float32)
    s_np = np.array(s_orig, dtype=np.float32) / 255.0  # Normalize S to 0-1
    v_np = np.array(v_orig, dtype=np.float32) / 255.0  # Normalize V to 0-1

    tint_mask = (s_np > saturation_threshold) & (v_np > value_threshold)

    if original_frame_hue_degrees is not None:
        # If a specific original hue is given, only tint pixels around that hue
        original_hue_pil = (original_frame_hue_degrees / 360.0) * 255.0
        hue_tolerance_pil = 30  # Tolerance for hue matching (0-255 scale)

        # Handle hue wrap-around for red (0 or 255)
        hue_diff = np.abs(h_np - original_hue_pil)
        hue_distance = np.minimum(hue_diff, 255.0 - hue_diff)

        hue_match_mask = hue_distance <= hue_tolerance_pil
        tint_mask &= hue_match_mask

    h_new_np = h_np.copy()
    h_new_np[tint_mask] = target_hue_pil

    h_final_pil = Image.fromarray(h_new_np.astype(np.uint8), mode="L")
    final_hsv_image = Image.merge("HSV", (h_final_pil, s_orig, v_orig))
    final_rgba_image = final_hsv_image.convert("RGBA")
    final_rgba_image.putalpha(alpha_channel)

    return final_rgba_image


def composite_images(
    frame_image: Image.Image,
    preview_collage: Image.Image,
    position: tuple = (50, 30),
    target_preview_size: tuple = (150, 150),
) -> Image.Image:
    """
    Composites a preview collage onto a frame image at a specified position.
    The preview_collage will be resized to fit target_preview_size while maintaining aspect ratio.

    Args:
        frame_image (Image.Image): The base frame image (PIL Image).
        preview_collage (Image.Image): The collage to place onto the frame (PIL Image).
        position (tuple): (x, y) coordinates on the frame where the top-left of the collage will be placed.
        target_preview_size (tuple): (width, height) the collage should be resized to fit within.

    Returns:
        Image.Image: The composited image.
    """
    if frame_image.mode != "RGBA":
        frame_image = frame_image.convert("RGBA")
    if preview_collage.mode != "RGBA":
        preview_collage = preview_collage.convert("RGBA")

    # Resize preview_collage to fit within target_preview_size, maintaining aspect ratio
    collage_copy = preview_collage.copy()
    collage_copy.thumbnail(
        tuple([int(x * 0.65) for x in target_preview_size]), Image.LANCZOS
    )
    frame_image = frame_image.resize(
        (target_preview_size[0], target_preview_size[1]), Image.LANCZOS
    )

    # Create a new blank image with the size of the frame to composite onto
    composited_image = Image.new("RGBA", frame_image.size, (0, 0, 0, 0))

    composited_image.paste(frame_image, (0, 0), frame_image)
    composited_image.paste(collage_copy, position, collage_copy)

    return composited_image


def enhance_mask_visualization(
    image,
    mask,
    *,
    opacity=0.5,
    contour_color=(0, 255, 0),
    fill_color=(0, 0, 255),
    scale_factor=1.0,
    final_thickness=2,
):
    """
    Draw a filled mask and an outline whose width stays roughly `final_thickness`
    pixels in the output, regardless of `scale_factor`.
    """

    #  I/O
    if isinstance(image, str):
        image_bgr = cv2.imread(image)  # already BGR
        if image_bgr is None:
            raise FileNotFoundError(image)
    else:
        # assume RGB ndarray → convert once
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    H, W = image_bgr.shape[:2]

    # Make sure mask matches image size first (cheap in C++)
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # - down-sample -
    if scale_factor != 1.0:
        small_bgr = cv2.resize(
            image_bgr,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )
        small_mask = cv2.resize(
            mask,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        small_bgr, small_mask = image_bgr, mask

    small_overlay = np.zeros_like(small_bgr)
    small_overlay[small_mask > 0] = fill_color

    blended = cv2.addWeighted(small_bgr, 1.0, small_overlay, opacity, 0)

    #  contours
    contours, _ = cv2.findContours(
        small_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    eff_thickness = max(1, int(round(final_thickness * scale_factor)))

    cv2.drawContours(blended, contours, -1, contour_color, eff_thickness)

    # - upscale
    if scale_factor != 1.0:
        blended = cv2.resize(blended, (W, H), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def combine_image_and_mask(image, mask, **kwargs):
    """
    Combine an image and its segmentation mask into an enhanced visualization.
    Handles TIFF image loading.

    Args:
    image (np.array or str): The original image (expected RGB if np.array) or path to the image file.
    mask (np.array): The binary segmentation mask.
    **kwargs: Additional arguments to pass to enhance_mask_visualization function.

    Returns:
    np.array: The combined image with enhanced mask visualization (RGB).
    """
    loaded_image_rgb = None
    if isinstance(image, str):
        image_path = image
        if image_path.lower().endswith((".tif", ".tiff")):
            try:
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
                elif img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1:
                    img_data_tiff = img_data_tiff[0]

                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        min_val, max_val = np.min(img_data_tiff), np.max(img_data_tiff)
                        if max_val > min_val:
                            img_data_tiff = (
                                (img_data_tiff - min_val) / (max_val - min_val)
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

                if img_data_tiff.ndim == 2:
                    loaded_image_rgb = cv2.cvtColor(img_data_tiff, cv2.COLOR_GRAY2RGB)
                elif img_data_tiff.ndim == 3:
                    if img_data_tiff.shape[-1] == 1:
                        loaded_image_rgb = cv2.cvtColor(
                            img_data_tiff, cv2.COLOR_GRAY2RGB
                        )
                    elif img_data_tiff.shape[-1] == 3:  # Assume RGB from tifffile
                        loaded_image_rgb = img_data_tiff
                    elif img_data_tiff.shape[-1] == 4:  # RGBA
                        loaded_image_rgb = cv2.cvtColor(
                            img_data_tiff, cv2.COLOR_RGBA2RGB
                        )
                    else:
                        logging.error(
                            f"Unsupported TIFF channel count for combine_mask: {img_data_tiff.shape[-1]} in {image_path}"
                        )
                        return None  # Or a placeholder error image
                else:
                    logging.error(
                        f"Unsupported TIFF dimensions for combine_mask: {img_data_tiff.ndim} in {image_path}"
                    )
                    return None
            except Exception as e:
                logging.error(
                    f"Error loading TIFF {image_path} in combine_mask: {e}",
                    exc_info=True,
                )
                return None
        else:  # Non-TIFF path
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logging.error(f"Failed to load image {image_path} with OpenCV.")
                return None
            loaded_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        loaded_image_rgb = image
    else:
        logging.error("Invalid image input type for combine_image_and_mask.")
        return None

    if loaded_image_rgb is None:
        return None  # If loading failed

    # Ensure the mask is binary
    mask_binary = (mask > 0).astype(np.uint8) * 255

    return enhance_mask_visualization(loaded_image_rgb, mask_binary, **kwargs)
