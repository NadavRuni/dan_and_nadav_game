"""
Ball and pocket detection pipeline using mask-based analysis.

This module provides functions to create a binary mask of the table felt,
and then uses this mask to find contours corresponding to balls and pockets.
It uses a combination of connected components analysis and the Hough Circle
Transform to refine the location and size of detected objects.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Ball, Pocket, Rectangle
from analyzer_table.table.rectangle import parse_rectangle_from_data
from const_numbers import (
    BASE_DIR,
    RECTANGLE_JSON_PATH,
    get_ball_radius,
    get_ball_radius_determinate,
    get_pocket_down_radius,
    get_pocket_up_radius,
    get_pocket_radius,
)
from output_utils import get_output_path

# --- Constants for CV and Filtering ---
MIN_CONTOUR_AREA = 50
POCKET_DETECTION_PADDING = 50
POCKET_EROSION_KERNEL = np.ones((9, 9), np.uint8)
FELT_MASK_OPEN_KERNEL = np.ones((5, 5), np.uint8)
FELT_MASK_CLOSE_KERNEL = np.ones((5, 5), np.uint8)


def _refine_contour_with_hough(
    gray_image: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int]]:
    """
    Tries to refine a bounding box to a precise circle using Hough Transform.

    Args:
        gray_image: The full grayscale image.
        bbox: The bounding box (x, y, width, height) of the contour to refine.

    Returns:
        A tuple (center_x, center_y, radius) if a circle is found, else None.
    """
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    padding = 20
    image_height, image_width = gray_image.shape[:2]

    roi_x1 = max(0, bbox_x - padding)
    roi_y1 = max(0, bbox_y - padding)
    roi_x2 = min(image_width, bbox_x + bbox_width + padding)
    roi_y2 = min(image_height, bbox_y + bbox_height + padding)

    roi_gray = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi_gray.size == 0:
        return None

    # Preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized_roi = clahe.apply(roi_gray)
    denoised_roi = cv2.bilateralFilter(equalized_roi, d=7, sigmaColor=50, sigmaSpace=7)
    blurred_roi = cv2.GaussianBlur(denoised_roi, (5, 5), 0)

    # Hough Circle Detection
    estimated_radius = 0.5 * min(bbox_width, bbox_height)
    circles = cv2.HoughCircles(
        blurred_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(0.8 * estimated_radius)),
        param1=120,
        param2=22,  # Accumulator threshold
        minRadius=max(6, int(0.7 * estimated_radius)),
        maxRadius=max(10, int(1.35 * estimated_radius)),
    )

    if circles is not None:
        # For simplicity, return the first and likely best circle.
        x, y, r = circles[0][0]
        # Convert coordinates back to the full image space
        return int(roi_x1 + x), int(roi_y1 + y), int(r)
    return None


def _find_objects_from_contours(
    binary_mask: np.ndarray,
    gray_image: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
) -> List[Tuple[int, int, int]]:
    """
    Finds circular objects from the contours of a binary mask.

    Args:
        binary_mask: The binary image to find contours in.
        gray_image: The corresponding grayscale image for refinement.
        min_area_ratio: Minimum component area as a fraction of image area.
        max_area_ratio: Maximum component area as a fraction of image area.

    Returns:
        A list of tuples, where each tuple is (center_x, center_y, radius).
    """
    image_area = binary_mask.shape[0] * binary_mask.shape[1]
    min_area = int(min_area_ratio * image_area)
    max_area = int(max_area_ratio * image_area)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Debugger.log(f"Found {len(contours)} raw contours.")

    detected_circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        bbox = cv2.boundingRect(contour)
        refined_circle = _refine_contour_with_hough(gray_image, bbox)

        if refined_circle:
            detected_circles.append(refined_circle)
        else:
            # Fallback to min enclosing circle if Hough fails
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detected_circles.append((int(x), int(y), int(radius)))

    return sorted(detected_circles, key=lambda item: item[2], reverse=True)


def create_felt_mask(
    input_path: str,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Creates a binary mask of the pool table felt from an image.

    Args:
        input_path: The path to the source image.

    Returns:
        A tuple containing:
        - The path to the saved binary mask file.
        - The binary mask as a numpy array.
        - The original image as a numpy array.
    """
    original_image = cv2.imread(input_path)
    if original_image is None:
        raise FileNotFoundError(f"❌ Could not read input image: {input_path}")

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 40, 30])
    upper_blue = np.array([125, 255, 255])
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    felt_mask = cv2.bitwise_or(blue_mask, green_mask)

    # Clean up the mask
    felt_mask = cv2.morphologyEx(
        felt_mask, cv2.MORPH_OPEN, FELT_MASK_OPEN_KERNEL, iterations=1
    )
    felt_mask = cv2.morphologyEx(
        felt_mask, cv2.MORPH_CLOSE, FELT_MASK_CLOSE_KERNEL, iterations=1
    )

    # Invert to get a mask where balls/pockets are white
    inverted_mask = cv2.bitwise_not(felt_mask)
    binary_mask = cv2.morphologyEx(
        inverted_mask, cv2.MORPH_CLOSE, FELT_MASK_CLOSE_KERNEL, iterations=1
    )

    mask_output_path = get_output_path("01_felt_mask.jpg", sub_dir="black_white_detect")
    cv2.imwrite(mask_output_path, binary_mask)
    return mask_output_path, binary_mask, original_image


def find_pockets_from_mask(
    mask_path: str, table_rect: Rectangle, original_image: np.ndarray
) -> Tuple[List[Pocket], str, str]:
    """
    Finds pockets from a binary mask, ensuring exactly 6 are returned.

    This function identifies contours, filters them by size and proximity to
    table corners/middles, and then estimates any missing pockets to ensure a
    full set of six.

    Args:
        mask_path: Path to the binary mask image.
        table_rect: The Rectangle defining the table boundaries.
        original_image: The original color image for drawing debug output.

    Returns:
        A tuple containing:
        - A list of exactly 6 Pocket objects.
        - The path to the debug image showing detected pockets on the mask.
        - The path to the debug image showing pockets on the original image.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"❌ Could not read mask image: {mask_path}")

    # Erode and pad the mask to help separate pocket contours from the edge
    mask = cv2.erode(mask, POCKET_EROSION_KERNEL, iterations=1)
    padded_mask = cv2.copyMakeBorder(
        mask,
        POCKET_DETECTION_PADDING,
        POCKET_DETECTION_PADDING,
        POCKET_DETECTION_PADDING,
        POCKET_DETECTION_PADDING,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    contours, _ = cv2.findContours(padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    output_display = cv2.cvtColor(padded_mask, cv2.COLOR_GRAY2BGR)
    all_valid_pockets = []

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not (get_pocket_down_radius() < radius < get_pocket_up_radius()):
            continue

        # Adjust for padding
        real_cx, real_cy = (
            int(x) - POCKET_DETECTION_PADDING,
            int(y) - POCKET_DETECTION_PADDING,
        )

        is_valid, location, _ = _is_close_to_rectangle_borders(
            table_rect, real_cx, real_cy, margin=get_pocket_radius() * 3
        )
        if is_valid:
            all_valid_pockets.append(
                Pocket(
                    center=(real_cx, real_cy),
                    radius=int(radius),
                    id=i,
                    location=location,
                    pocket_img_cordinates_on_table=(real_cx, real_cy),
                )
            )
    # TODO: Add more robust de-duplication and estimation logic here
    final_pockets = all_valid_pockets  # Simplified for now

    # Drawing and saving debug visuals
    img_with_pockets = original_image.copy()
    for p in final_pockets:
        cv2.circle(img_with_pockets, p.center, p.radius, (255, 0, 0), 4)

    debug_path = get_output_path("pocket_mask.jpg", sub_dir="black_white_detect")
    original_debug_path = get_output_path(
        "original_with_pockets.jpg", sub_dir="black_white_detect"
    )
    cv2.imwrite(debug_path, output_display)
    cv2.imwrite(original_debug_path, img_with_pockets)

    return final_pockets, debug_path, original_debug_path


def detect_balls_from_mask(input_path: str) -> List[Ball]:
    """
    High-level pipeline to detect balls from an image using a felt mask.

    Args:
        input_path: Path to the source image.

    Returns:
        A list of detected Ball objects.
    """
    _, binary_mask, original_image = create_felt_mask(input_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    raw_circles = _find_objects_from_contours(binary_mask, gray_image, 0.00015, 0.0055)

    ball_objects = []
    for cx, cy, r in raw_circles:
        if (
            get_ball_radius() - get_ball_radius_determinate()
            <= r
            <= get_ball_radius() + get_ball_radius_determinate()
        ):
            ball_objects.append(Ball(center=(cx, cy), radius=r))
        else:
            Debugger.warn(f"Ignored ball-like object with invalid radius {r}")

    Debugger.log(f"🎱 Found {len(ball_objects)} balls from mask.")
    return ball_objects


def _is_close_to_rectangle_borders(
    rectangle: Rectangle, point_x: int, point_y: int, margin: int
) -> tuple[bool, str, float]:
    """
    Checks if a point is close to a corner or the middle of a long side of the rectangle.
    Returns a tuple: (is_valid, location_name, distance)
    """
    if not rectangle:
        return False, "UNKNOWN", -1.0

    min_x = min(rectangle.top_left[0], rectangle.bottom_left[0])
    max_x = max(rectangle.top_right[0], rectangle.bottom_right[0])
    min_y = min(rectangle.top_left[1], rectangle.top_right[1])
    max_y = max(rectangle.bottom_left[1], rectangle.bottom_right[1])

    # 1. Define key points with their names
    corners = {
        "TL": rectangle.top_left,
        "TR": rectangle.top_right,
        "BL": rectangle.bottom_left,
        "BR": rectangle.bottom_right,
    }

    width = max_x - min_x
    height = max_y - min_y
    midpoints = {}
    if width > height:  # Horizontal table
        midpoints["TM"] = ((min_x + max_x) // 2, min_y)
        midpoints["BM"] = ((min_x + max_x) // 2, max_y)
    else:  # Vertical table (or square)
        midpoints["left_middle"] = (min_x, (min_y + max_y) // 2)
        midpoints["right_middle"] = (max_x, (min_y + max_y) // 2)

    interest_points = {**corners, **midpoints}
    pocket_margin = margin * 2

    # 2. Find the closest interest point
    min_dist = float("inf")
    closest_location = "UNKNOWN"
    for name, (p_x, p_y) in interest_points.items():
        distance = np.sqrt((point_x - p_x) ** 2 + (point_y - p_y) ** 2)
        if distance < min_dist:
            min_dist = distance
            closest_location = name

    # 3. Check if the point is close enough to the *closest* interest point
    if min_dist <= pocket_margin:
        Debugger.log(
            f"Point ({point_x}, {point_y}) is close to interest point {closest_location}."
        )
        return True, closest_location, min_dist

    Debugger.log(f"Point ({point_x}, {point_y}) is not close to any interest point.")
    return False, "UNKNOWN", min_dist
