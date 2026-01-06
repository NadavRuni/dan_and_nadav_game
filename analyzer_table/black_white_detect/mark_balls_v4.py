#!/usr/bin/env python3
"""
A ball detection pipeline with special handling for glare on green balls.

This module is largely a duplicate of 'detect_balls_and_pockets.py' but
contains a specific modification to the felt mask creation to account for
strong reflections on green balls that might cause them to be missed.

Warning:
    This file contains a significant amount of duplicated code from
    'detect_balls_and_pockets.py'. This is a major maintenance issue and
    should be resolved by extracting the common functions into a shared utility
    module.
"""
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from analyzer_table.launcher_helper.json_models import Ball
from const_numbers import (
    FELT_MASK_PATH,
    get_ball_radius,
    get_ball_radius_determinate,
)
from output_utils import get_output_path

# This is not a recommended practice for managing imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Note: The following functions are duplicated from
# 'detect_balls_and_pockets.py' and should be refactored into a shared module.


def _preprocess_roi(roi_gray: np.ndarray) -> np.ndarray:
    """Applies a series of preprocessing steps to a region of interest."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized_roi = clahe.apply(roi_gray)
    denoised_roi = cv2.bilateralFilter(equalized_roi, d=7, sigmaColor=50, sigmaSpace=7)
    blurred_roi = cv2.GaussianBlur(denoised_roi, (5, 5), 0)
    return blurred_roi


def _refine_with_hough(
    gray_image: np.ndarray, bbox: tuple, padding: int = 20
) -> tuple | None:
    """Refines a bounding box to a circle using Hough Circle Transform."""
    image_height, image_width = gray_image.shape[:2]
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    roi_x1 = max(0, bbox_x - padding)
    roi_y1 = max(0, bbox_y - padding)
    roi_x2 = min(image_width, bbox_x + bbox_width + padding)
    roi_y2 = min(image_height, bbox_y + bbox_height + padding)
    roi = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi.size == 0 or roi.max() == 0:
        return None

    preprocessed_roi = _preprocess_roi(roi)
    estimated_radius = 0.5 * min(bbox_width, bbox_height)

    circles = cv2.HoughCircles(
        preprocessed_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(0.8 * estimated_radius)),
        param1=120,
        param2=22,
        minRadius=max(6, int(0.7 * estimated_radius)),
        maxRadius=max(10, int(1.35 * estimated_radius)),
    )

    if circles is not None:
        x, y, r = circles[0][0]
        return int(roi_x1 + x), int(roi_y1 + y), int(r)
    return None


def _touches_border(
    bbox: tuple, image_width: int, image_height: int, padding: int = 3
) -> bool:
    """Checks if a bounding box touches the image border."""
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    return (
        bbox_x <= padding
        or bbox_y <= padding
        or (bbox_x + bbox_width) >= (image_width - 1 - padding)
        or (bbox_y + bbox_height) >= (image_height - 1 - padding)
    )


def _balls_from_connected_components(
    binary_mask: np.ndarray, gray_image: np.ndarray
) -> List[Ball]:
    """Finds balls from connected components in a binary mask."""
    image_height, image_width = binary_mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    balls = []
    image_area = image_height * image_width
    min_component_area = int(0.00015 * image_area)
    max_component_area = int(0.0055 * image_area)

    for label_id in range(1, num_labels):
        stat_x, stat_y, stat_width, stat_height, stat_area = stats[label_id]
        if not (min_component_area <= stat_area <= max_component_area):
            continue
        if _touches_border(
            (stat_x, stat_y, stat_width, stat_height), image_width, image_height
        ):
            continue

        refined_circle = _refine_with_hough(
            gray_image, (stat_x, stat_y, stat_width, stat_height)
        )

        if refined_circle:
            center_x, center_y, radius = refined_circle
            if (
                get_ball_radius() - get_ball_radius_determinate()
                <= radius
                <= get_ball_radius() + get_ball_radius_determinate()
            ):
                balls.append(Ball(center=(center_x, center_y), radius=radius))

    return balls


def _create_felt_mask(
    hsv_image: np.ndarray, handle_glare: bool, original_image: np.ndarray
) -> np.ndarray:
    """Creates a binary mask of the table felt."""
    lower_blue = np.array([80, 40, 30])
    upper_blue = np.array([125, 255, 255])
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    felt_mask = cv2.bitwise_or(blue_mask, green_mask)

    if handle_glare:
        # Find white glare spots, which are assumed to be reflections on balls.
        gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, glare_mask = cv2.threshold(gray_scale, 200, 255, cv2.THRESH_BINARY)

        # Dilate these glare spots to the approximate size of a ball.
        radius_approx = int(get_ball_radius() * 0.8)
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius_approx, radius_approx)
        )
        ball_locations_by_glare = cv2.dilate(glare_mask, dilate_kernel, iterations=1)

        # Subtract these 'ball bubbles' from the felt mask.
        # This ensures that even if a ball's color is similar to the felt,
        # its reflection will create a hole in the mask, allowing detection.
        felt_mask = cv2.bitwise_and(felt_mask, cv2.bitwise_not(ball_locations_by_glare))

    kernel = np.ones((5, 5), np.uint8)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Invert to get a mask where balls/pockets are white
    return cv2.bitwise_not(felt_mask)


def detect_balls_full_pipeline(
    input_path: str, handle_green_glare: bool = False
) -> List[Ball]:
    """
    Full pipeline for detecting balls in an image, with an option to handle
    glare on green balls.

    Args:
        input_path: Path to the source image.
        handle_green_glare: If True, applies a special correction to prevent
                            green balls from being missed due to their color
                            matching the table felt.

    Returns:
        A list of detected Ball objects.
    """
    original_image = cv2.imread(input_path)
    if original_image is None:
        raise FileNotFoundError(f"❌ Could not read input image: {input_path}")

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    binary_mask = _create_felt_mask(hsv_image, handle_green_glare, original_image)

    mask_output_path = get_output_path(FELT_MASK_PATH, sub_dir="black_white_detect")
    cv2.imwrite(mask_output_path, binary_mask)
    print(f"🖤 Mask saved to: {mask_output_path}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    ball_objects = _balls_from_connected_components(binary_mask, gray_image)
    print(f"🎱 Found {len(ball_objects)} balls.")

    # --- Create and save debug visualization ---
    output_image = original_image.copy()
    for ball in ball_objects:
        cv2.circle(output_image, ball.center, ball.radius, (0, 255, 0), 2)

    output_path = get_output_path(
        "output_marked_balls.jpg", sub_dir="black_white_detect"
    )
    cv2.imwrite(output_path, output_image)
    print(f"✅ Final image saved to: {output_path}")

    return ball_objects


if __name__ == "__main__":
    # Example usage of the pipeline
    example_image_path = "photos/img_start.jpeg"
    detected_balls = detect_balls_full_pipeline(
        example_image_path, handle_green_glare=True
    )
    print(
        f"[OK] Example finished. Detected {len(detected_balls)} balls. "
        f"First 3: {[b.center for b in detected_balls[:3]]}"
    )
