"""Compute morphological, shape, and intensity attributes from a binary mask."""

import math

import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize

from core.processors.image_utils import to_gray


def compute_mask_attributes(
    image: np.ndarray, mask: np.ndarray, scale_factor: float = 1.0
) -> dict:
    gray = to_gray(image)
    binary = (mask > 0).astype(np.uint8)
    attrs = {}

    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return attrs

    contour = max(contours, key=cv2.contourArea)
    sf = scale_factor
    sf2 = sf * sf

    # ── Size & geometry ──
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    attrs["area"] = area_px * sf2
    attrs["perimeter"] = perimeter_px * sf

    x, y, w, h = cv2.boundingRect(contour)
    attrs["bbox_x"] = x * sf
    attrs["bbox_y"] = y * sf
    attrs["bbox_w"] = w * sf
    attrs["bbox_h"] = h * sf

    if len(contour) >= 5:
        (_, _), (rw, rh), angle = cv2.fitEllipse(contour)
        attrs["major_axis_length"] = max(rw, rh) * sf
        attrs["minor_axis_length"] = min(rw, rh) * sf
        attrs["orientation_deg"] = angle
        attrs["eccentricity"] = (
            math.sqrt(1 - (min(rw, rh) / max(rw, rh)) ** 2) if max(rw, rh) > 0 else 0
        )
    else:
        attrs["major_axis_length"] = max(w, h) * sf
        attrs["minor_axis_length"] = min(w, h) * sf
        attrs["orientation_deg"] = 0
        attrs["eccentricity"] = 0

    attrs["equivalent_diameter"] = math.sqrt(4 * area_px / math.pi) * sf if area_px > 0 else 0
    attrs["aspect_ratio"] = (
        attrs["major_axis_length"] / attrs["minor_axis_length"]
        if attrs["minor_axis_length"] > 0 else 0
    )
    attrs["elongatedness"] = attrs["aspect_ratio"]

    # ── Circularity ──
    attrs["form_factor"] = (4 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0
    attrs["compactness"] = (perimeter_px ** 2 / (4 * math.pi * area_px)) if area_px > 0 else 0

    # ── Convexity ──
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    attrs["convex_area"] = hull_area * sf2
    attrs["convex_perimeter"] = hull_perimeter * sf
    attrs["solidity"] = (area_px / hull_area) if hull_area > 0 else 0
    attrs["convexity"] = (hull_perimeter / perimeter_px) if perimeter_px > 0 else 0
    attrs["convex_deficiency"] = (hull_area - area_px) * sf2

    # ── Feret diameters ──
    if len(contour) >= 2:
        pts = contour.reshape(-1, 2).astype(np.float32)
        try:
            hull_pts = cv2.convexHull(pts).reshape(-1, 2)
            dists = [
                np.linalg.norm(hull_pts[i] - hull_pts[j])
                for i in range(len(hull_pts))
                for j in range(i + 1, len(hull_pts))
            ]
            attrs["feret_max"] = max(dists) * sf if dists else 0
            attrs["feret_min"] = min(dists) * sf if dists else 0
        except Exception:
            attrs["feret_max"] = 0
            attrs["feret_min"] = 0

    # ── Thickness (distance transform) ──
    dist_map = cv2.distanceTransform(binary * 255, cv2.DIST_L2, 5)
    masked_dist = dist_map[binary > 0]
    if len(masked_dist) > 0:
        attrs["thickness_mean"] = float(np.mean(masked_dist)) * sf
        attrs["thickness_std"] = float(np.std(masked_dist)) * sf
        attrs["thickness_max"] = float(np.max(masked_dist)) * sf
        attrs["thickness_min"] = (
            float(np.min(masked_dist[masked_dist > 0])) * sf
            if np.any(masked_dist > 0) else 0
        )

    # ── Topology ──
    labeled = label(binary)
    attrs["component_count"] = labeled.max()
    contours_tree, hierarchy = cv2.findContours(binary * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    attrs["holes_count"] = (
        max(0, len(contours_tree) - attrs["component_count"])
        if hierarchy is not None else 0
    )
    attrs["euler_number"] = attrs["component_count"] - attrs["holes_count"]

    # ── Skeleton ──
    try:
        skel = skeletonize(binary > 0)
        attrs["skeleton_length"] = int(np.sum(skel)) * sf
        skel_uint = skel.astype(np.uint8)
        kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
        neighbors = cv2.filter2D(skel_uint, -1, kernel_3x3) * skel_uint
        attrs["skeleton_branch_points"] = int(np.sum(neighbors > 3))
        attrs["skeleton_end_points"] = int(np.sum(neighbors == 2))
    except Exception:
        attrs["skeleton_length"] = 0
        attrs["skeleton_branch_points"] = 0
        attrs["skeleton_end_points"] = 0

    # ── Centroid ──
    M = cv2.moments(contour)
    if M["m00"] > 0:
        attrs["centroid_x"] = (M["m10"] / M["m00"]) * sf
        attrs["centroid_y"] = (M["m01"] / M["m00"]) * sf

    # ── Intensity statistics ──
    pixels = gray[binary > 0]
    if len(pixels) > 0:
        attrs["mean_intensity"] = float(np.mean(pixels))
        attrs["std_intensity"] = float(np.std(pixels))
        attrs["median_intensity"] = float(np.median(pixels))
        attrs["min_intensity"] = float(np.min(pixels))
        attrs["max_intensity"] = float(np.max(pixels))
        attrs["intensity_sum"] = float(np.sum(pixels))

    # ── Background & contrast ──
    dilated = cv2.dilate(binary * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    bg_ring = (dilated > 0) & (binary == 0)
    bg_pixels = gray[bg_ring]
    if len(bg_pixels) > 0:
        attrs["background_mean"] = float(np.mean(bg_pixels))
        attrs["background_std"] = float(np.std(bg_pixels))
        if attrs.get("max_intensity", 0) + attrs.get("min_intensity", 0) > 0:
            attrs["modulation"] = (
                (attrs["max_intensity"] - attrs["min_intensity"])
                / (attrs["max_intensity"] + attrs["min_intensity"])
            )
        bg_std = attrs["background_std"]
        if bg_std > 0:
            attrs["snr"] = attrs.get("mean_intensity", 0) / bg_std

    # ── Gradient (focus measure) ──
    if area_px > 0:
        roi_gray = gray[y:y + h, x:x + w]
        roi_mask = binary[y:y + h, x:x + w]
        sx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sx**2 + sy**2)
        grad_vals = grad_mag[roi_mask > 0]
        if len(grad_vals) > 0:
            attrs["gradient_rms"] = float(np.sqrt(np.mean(grad_vals**2)))
            attrs["gradient_max"] = float(np.max(grad_vals))

    # ── Hu moments ──
    hu = cv2.HuMoments(M).flatten()
    for i, val in enumerate(hu):
        attrs[f"hu{i + 1}"] = float(val)

    # Convert numpy types to Python natives for JSON serialization
    return {k: _to_python(v) for k, v in attrs.items()}


# ── Helpers ─────────────────────────────────────────────────────────────


def _to_python(val):
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


