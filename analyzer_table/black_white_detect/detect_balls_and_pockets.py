#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from typing import List

from output_utils import get_output_path
from const_numbers import *
from analyzer_table.launcher_helper.json_models import (
    Ball,
    PocketDetection,
    Pocket_Location_On_Table,
)
from analyzer_table.table.rectangle import parse_rectangle_from_data
from analyzer_table.launcher_helper.json_models import Rectangle
from typing import Optional, List


def is_near_border(center, image_width, image_height, margin_fraction=0.2):
    """Checks if a point is near the border of the image."""
    center_x, center_y = center
    margin_x = image_width * margin_fraction
    margin_y = image_height * margin_fraction

    return (
        center_x < margin_x
        or center_x > image_width - margin_x
        or center_y < margin_y
        or center_y > image_height - margin_y
    )


def preprocess_roi(roi_gray):
    """Applies a series of preprocessing steps to a region of interest."""
    print(f"DEBUG: preprocess_roi - Input roi_gray shape: {roi_gray.shape}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    print(
        f"DEBUG: preprocess_roi - CLAHE object created with clipLimit=2.0, tileGridSize=(4, 4)"
    )
    equalized_roi = clahe.apply(roi_gray)
    print(
        f"DEBUG: preprocess_roi - CLAHE applied. equalized_roi shape: {equalized_roi.shape}"
    )
    denoised_roi = cv2.bilateralFilter(equalized_roi, d=7, sigmaColor=50, sigmaSpace=7)
    print(
        f"DEBUG: preprocess_roi - Bilateral filter applied. denoised_roi shape: {denoised_roi.shape}"
    )
    blurred_roi = cv2.GaussianBlur(denoised_roi, (5, 5), 0)
    print(
        f"DEBUG: preprocess_roi - Gaussian blur applied. blurred_roi shape: {blurred_roi.shape}"
    )
    return blurred_roi


def edge_support_ratio(edges, center_x, center_y, radius):
    """Calculates the ratio of edge pixels within a circular region."""
    print(
        f"DEBUG: edge_support_ratio - Input edges shape: {edges.shape}, center_x: {center_x}, center_y: {center_y}, radius: {radius}"
    )
    height, width = edges.shape[:2]
    print(
        f"DEBUG: edge_support_ratio - Image dimensions: height={height}, width={width}"
    )
    radius = int(max(1, radius))
    print(f"DEBUG: edge_support_ratio - Adjusted radius: {radius}")
    center_x, center_y = int(center_x), int(center_y)
    print(
        f"DEBUG: edge_support_ratio - Adjusted center_x: {center_x}, center_y: {center_y}"
    )
    padding = 1
    roi_x1 = max(0, center_x - radius - padding)
    roi_x2 = min(width - 1, center_x + radius + padding)
    roi_y1 = max(0, center_y - radius - padding)
    roi_y2 = min(height - 1, center_y + radius + padding)
    print(
        f"DEBUG: edge_support_ratio - ROI coordinates: x1={roi_x1}, y1={roi_y1}, x2={roi_x2}, y2={roi_y2}"
    )
    if roi_x2 < roi_x1 or roi_y2 < roi_y1:
        print(f"DEBUG: edge_support_ratio - Invalid ROI. Returning 0.0")
        return 0.0
    roi = edges[roi_y1 : roi_y2 + 1, roi_x1 : roi_x2 + 1]
    print(f"DEBUG: edge_support_ratio - ROI shape: {roi.shape}")
    if roi.size == 0:
        print(f"DEBUG: edge_support_ratio - ROI is empty. Returning 0.0")
        return 0.0
    if roi.max() == 0:
        print(f"DEBUG: edge_support_ratio - ROI max is 0 (no edges). Returning 0.0")
        return 0.0
    support_ratio = float(roi.sum() > 0)
    print(f"DEBUG: edge_support_ratio - Support ratio: {support_ratio}")
    return support_ratio


def refine_with_hough(gray_image, bbox_x, bbox_y, bbox_width, bbox_height, padding=20):
    """Refines a bounding box to a circle using Hough Circle Transform."""
    print(
        f"DEBUG: refine_with_hough - Input gray_image shape: {gray_image.shape}, bbox: ({bbox_x},{bbox_y},{bbox_width},{bbox_height}), padding: {padding}"
    )
    image_height, image_width = gray_image.shape[:2]
    print(
        f"DEBUG: refine_with_hough - Image dimensions: height={image_height}, width={image_width}"
    )
    roi_x1 = max(0, bbox_x - padding)
    roi_y1 = max(0, bbox_y - padding)
    roi_x2 = min(image_width, bbox_x + bbox_width + padding)
    roi_y2 = min(image_height, bbox_y + bbox_height + padding)
    print(
        f"DEBUG: refine_with_hough - ROI coordinates: x1={roi_x1}, y1={roi_y1}, x2={roi_x2}, y2={roi_y2}"
    )
    roi = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    print(f"DEBUG: refine_with_hough - ROI shape: {roi.shape}")
    if roi.size == 0:
        print(f"DEBUG: refine_with_hough - ROI is empty. Returning None")
        return None
    if roi.max() == 0:
        print(f"DEBUG: refine_with_hough - ROI max is 0. Returning None")
        return None
    preprocessed_roi = preprocess_roi(roi)
    print(
        f"DEBUG: refine_with_hough - preprocessed_roi shape: {preprocessed_roi.shape}"
    )
    edges = cv2.Canny(preprocessed_roi, 60, 160)
    print(f"DEBUG: refine_with_hough - Canny edges shape: {edges.shape}")
    estimated_radius = 0.5 * min(bbox_width, bbox_height)
    print(f"DEBUG: refine_with_hough - Estimated radius: {estimated_radius}")

    def try_hough(hough_param2, min_radius_multiplier, max_radius_multiplier):
        min_radius = max(6, int(min_radius_multiplier * estimated_radius))
        max_radius = max(min_radius + 2, int(max_radius_multiplier * estimated_radius))
        print(
            f"DEBUG: refine_with_hough/try_hough - Trying Hough with p2={hough_param2}, minR={min_radius}, maxR={max_radius}"
        )
        return cv2.HoughCircles(
            preprocessed_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, int(0.8 * estimated_radius)),
            param1=120,
            param2=hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

    circles = try_hough(22, 0.7, 1.35)
    if circles is None:
        print(f"DEBUG: refine_with_hough - First Hough attempt failed.")
        circles = try_hough(18, 0.45, 1.9)
    if circles is None:
        print(f"DEBUG: refine_with_hough - Second Hough attempt failed. Returning None")
        return None

    print(f"DEBUG: refine_with_hough - Found {len(circles[0])} candidate circles.")
    best_circle, best_score = None, -1.0
    for circle in circles[0]:
        cx, cy, r = circle
        print(f"DEBUG: refine_with_hough - Candidate circle: cx={cx}, cy={cy}, r={r}")
        if r < 0.4 * estimated_radius:
            print(f"DEBUG: refine_with_hough - Circle radius {r} too small, skipping.")
            continue
        cov = edge_support_ratio(edges, cx, cy, r)
        score = cov * r
        print(
            f"DEBUG: refine_with_hough - Circle score: {score} (coverage={cov}, radius={r})"
        )
        if score > best_score:
            best_score = score
            best_circle = (cx, cy, r)
            print(
                f"DEBUG: refine_with_hough - New best circle found: {best_circle}, score: {best_score}"
            )
    if best_circle is None:
        print(f"DEBUG: refine_with_hough - No best circle found. Returning None")
        return None
    cx, cy, r = best_circle
    print(
        f"DEBUG: refine_with_hough - Final best circle (relative): cx={cx}, cy={cy}, r={r}"
    )
    return int(roi_x1 + cx), int(roi_y1 + cy), int(r)


def touches_border(bbox, image_width, image_height, padding=3):
    """Checks if a bounding box touches the image border."""
    print(
        f"DEBUG: touches_border - Input bbox: {bbox}, image_width: {image_width}, image_height: {image_height}, padding: {padding}"
    )
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    result = (
        bbox_x <= padding
        or bbox_y <= padding
        or (bbox_x + bbox_width) >= (image_width - 1 - padding)
        or (bbox_y + bbox_height) >= (image_height - 1 - padding)
    )
    print(f"DEBUG: touches_border - Result: {result} for bbox: {bbox}")
    return result


def is_close_to_rectangle_borders(
    rectangle: Rectangle, point_x: int, point_y: int, margin: int
) -> tuple[bool, str, float]:
    """
    Checks if a point is close to a corner or the middle of a long side of the rectangle.
    Returns a tuple: (is_valid, location_name, distance)
    """
    if not rectangle:
        return False, Pocket_Location_On_Table.unknown, -1.0

    min_x = min(rectangle.top_left[0], rectangle.bottom_left[0])
    max_x = max(rectangle.top_right[0], rectangle.bottom_right[0])
    min_y = min(rectangle.top_left[1], rectangle.top_right[1])
    max_y = max(rectangle.bottom_left[1], rectangle.bottom_right[1])

    # 1. Define key points with their names
    corners = {
        Pocket_Location_On_Table.top_left: rectangle.top_left,
        Pocket_Location_On_Table.top_right: rectangle.top_right,
        Pocket_Location_On_Table.bottom_left: rectangle.bottom_left,
        Pocket_Location_On_Table.bottom_right: rectangle.bottom_right,
    }

    width = max_x - min_x
    height = max_y - min_y
    midpoints = {}
    if width > height: # Horizontal table
        midpoints[Pocket_Location_On_Table.top_middle] = ((min_x + max_x) // 2, min_y)
        midpoints[Pocket_Location_On_Table.buttom_middle] = ((min_x + max_x) // 2, max_y)
    else: # Vertical table (or square)
        midpoints["left_middle"] = (min_x, (min_y + max_y) // 2)
        midpoints["right_middle"] = (max_x, (min_y + max_y) // 2)

    interest_points = {**corners, **midpoints}
    pocket_margin = margin * 2

    # 2. Find the closest interest point
    min_dist = float("inf")
    closest_location = Pocket_Location_On_Table.unknown
    for name, (p_x, p_y) in interest_points.items():
        distance = np.sqrt((point_x - p_x) ** 2 + (point_y - p_y) ** 2)
        if distance < min_dist:
            min_dist = distance
            closest_location = name

    # 3. Check if the point is close enough to the *closest* interest point
    if min_dist <= pocket_margin:
        return True, closest_location, min_dist

    print("DEBUG: Point not close enough to any interest point. (", point_x, " , ", point_y, " )")
    return False, Pocket_Location_On_Table.unknown, min_dist


def _circles_from_connected_components(
    binary_mask, gray_image, min_area_ratio, max_area_ratio
):
    """Finds circles from connected components in a binary mask."""
    print(
        f"DEBUG: _circles_from_connected_components - Input binary_mask shape: {binary_mask.shape}, gray_image shape: {gray_image.shape}, min_area_ratio: {min_area_ratio}, max_area_ratio: {max_area_ratio}"
    )
    image_height, image_width = binary_mask.shape[:2]
    print(
        f"DEBUG: _circles_from_connected_components - Image dimensions: height={image_height}, width={image_width}"
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    print(
        f"DEBUG: _circles_from_connected_components - Found {num_labels} connected components."
    )
    circles_found = []
    image_area = image_height * image_width
    min_component_area = int(min_area_ratio * image_area)
    max_component_area = int(max_area_ratio * image_area)
    MAX_RADIUS = 100
    print(
        f"DEBUG: _circles_from_connected_components - Component area range: [{min_component_area}, {max_component_area}], MAX_RADIUS: {MAX_RADIUS}"
    )
    for label_id in range(1, num_labels):
        stat_x, stat_y, stat_width, stat_height, stat_area = stats[label_id]
        print(
            f"DEBUG: _circles_from_connected_components - Processing component {label_id}: bbox=({stat_x},{stat_y},{stat_width},{stat_height}), area={stat_area}"
        )
        if not min_component_area <= stat_area <= max_component_area:
            print(
                f"DEBUG: _circles_from_connected_components - Component {label_id} area {stat_area} outside range. Skipping."
            )
            continue
        if touches_border(
            (stat_x, stat_y, stat_width, stat_height), image_width, image_height
        ):
            print(
                f"DEBUG: _circles_from_connected_components - Component {label_id} touches border. Skipping."
            )
            continue
        refined_circle = refine_with_hough(
            gray_image, stat_x, stat_y, stat_width, stat_height
        )
        if refined_circle:
            center_x, center_y, radius = refined_circle
            print(
                f"DEBUG: _circles_from_connected_components - Component {label_id} refined to circle: ({center_x},{center_y},{radius})"
            )
        else:
            center_x, center_y = centroids[label_id]
            estimated_radius = int(0.5 * (stat_width + stat_height) / 2)
            radius = max(6, estimated_radius)
            print(
                f"DEBUG: _circles_from_connected_components - Component {label_id} not refined, using centroid: ({center_x},{center_y},{radius})"
            )
        if radius > MAX_RADIUS:
            print(
                f"DEBUG: _circles_from_connected_components - Circle radius {radius} exceeds MAX_RADIUS. Skipping."
            )
            continue
        circles_found.append((int(center_x), int(center_y), int(radius)))
        print(
            f"DEBUG: _circles_from_connected_components - Added circle: ({int(center_x)}, {int(center_y)}, {int(radius)})"
        )
    print(
        f"DEBUG: _circles_from_connected_components - Found {len(circles_found)} circles."
    )
    return sorted(circles_found, key=lambda item: item[2], reverse=True)


def _estimate_missing_pockets(
    pockets: List[PocketDetection], rect: Rectangle
) -> List[PocketDetection]:
    """
    Estimates the positions of missing pockets based on the locations of found pockets.
    """
    print("DEBUG: Estimating missing pockets...")

    pockets_map = {p.location: p for p in pockets}
    all_locations = {
        Pocket_Location_On_Table.top_left,
        Pocket_Location_On_Table.top_right,
        Pocket_Location_On_Table.bottom_left,
        Pocket_Location_On_Table.bottom_right,
        Pocket_Location_On_Table.top_middle,
        Pocket_Location_On_Table.buttom_middle,
    }
    missing_locations = all_locations - set(pockets_map.keys())

    if not missing_locations:
        return pockets

    # Helper to get point from pocket or rect
    def get_point(loc):
        if loc in pockets_map:
            return pockets_map[loc].center
        # Fallback to rect corners if pocket not found
        min_x, max_x = rect.top_left[0], rect.top_right[0]
        min_y, max_y = rect.top_left[1], rect.bottom_left[1]
        centers = {
            Pocket_Location_On_Table.top_left: (min_x, min_y),
            Pocket_Location_On_Table.top_right: (max_x, min_y),
            Pocket_Location_On_Table.bottom_left: (min_x, max_y),
            Pocket_Location_On_Table.bottom_right: (max_x, max_y),
            Pocket_Location_On_Table.top_middle: ((min_x + max_x) // 2, min_y),
            Pocket_Location_On_Table.buttom_middle: ((min_x + max_x) // 2, max_y),
        }
        return centers[loc]

    estimated_pockets = list(pockets)
    
    # Estimate missing corners by averaging
    for loc in missing_locations:
        center_x, center_y = 0, 0
        if loc == Pocket_Location_On_Table.top_left:
            tr = get_point(Pocket_Location_On_Table.top_right)
            bl = get_point(Pocket_Location_On_Table.bottom_left)
            center_x = bl[0] 
            center_y = tr[1]
        elif loc == Pocket_Location_On_Table.top_right:
            tl = get_point(Pocket_Location_On_Table.top_left)
            br = get_point(Pocket_Location_On_Table.bottom_right)
            center_x = br[0]
            center_y = tl[1]
        elif loc == Pocket_Location_On_Table.bottom_left:
            tl = get_point(Pocket_Location_On_Table.top_left)
            br = get_point(Pocket_Location_On_Table.bottom_right)
            center_x = tl[0]
            center_y = br[1]
        elif loc == Pocket_Location_On_Table.bottom_right:
            bl = get_point(Pocket_Location_On_Table.bottom_left)
            tr = get_point(Pocket_Location_On_Table.top_right)
            center_x = tr[0]
            center_y = bl[1]
        
        # Estimate middle pockets from corners
        elif loc == Pocket_Location_On_Table.top_middle:
            tl = get_point(Pocket_Location_On_Table.top_left)
            tr = get_point(Pocket_Location_On_Table.top_right)
            center_x = (tl[0] + tr[0]) // 2
            center_y = (tl[1] + tr[1]) // 2
        elif loc == Pocket_Location_On_Table.buttom_middle:
            bl = get_point(Pocket_Location_On_Table.bottom_left)
            br = get_point(Pocket_Location_On_Table.bottom_right)
            center_x = (bl[0] + br[0]) // 2
            center_y = (bl[1] + br[1]) // 2

        if center_x != 0 or center_y != 0:
            placeholder = PocketDetection(
                center=(int(center_x), int(center_y)),
                radius=int(get_pocket_radius()),
                id=-1,
                location=loc,
                distance=-1.0,
            )
            estimated_pockets.append(placeholder)
            print(f"  - Estimated placeholder for {loc} at {placeholder.center}")

    return estimated_pockets

def find_corner_pockets_from_mask(
    mask_path: str, binary_mask: np.array, original_image: np.array, rect: Rectangle
) -> tuple[List["PocketDetection"], str, str]:
    """
    Finds and circles white polygonal shapes in the corners of a binary mask.
    It ensures that exactly 6 pockets are returned by selecting the best candidate for each location.
    """
    from const_numbers import set_table_length, set_table_width

    set_table_width(original_image.shape[0])
    set_table_length(original_image.shape[1])
    print(f"DEBUG: find_corner_pockets - Loading mask from: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ Could not read mask image: {mask_path}")
        return [], "", ""

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    padding = 50
    padded_mask = cv2.copyMakeBorder(
        mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )
    output_path = get_output_path("padded_mask.jpg")
    cv2.imwrite(output_path, padded_mask)

    contours, _ = cv2.findContours(padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    output_display = cv2.cvtColor(padded_mask, cv2.COLOR_GRAY2BGR)

    all_valid_pockets = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not (get_pocket_down_radius() < radius < get_pocket_up_radius()):
            continue

        real_center_x = int(x) - padding
        real_center_y = int(y) - padding

        is_valid, location, distance = is_close_to_rectangle_borders(
            rect, real_center_x, real_center_y, margin=get_pocket_radius() * 5
        )

        if is_valid:
            pocket = PocketDetection(
                center=(real_center_x, real_center_y),
                radius=int(radius),
                id=i,
                location=location,
                distance=distance,
            )
            all_valid_pockets.append(pocket)

    # --- Deduplication ---
    pockets_by_location = {}
    for p in all_valid_pockets:
        if p.location not in pockets_by_location:
            pockets_by_location[p.location] = []
        pockets_by_location[p.location].append(p)

    final_pockets = []
    for location, candidates in pockets_by_location.items():
        if len(candidates) > 1:
            best_pocket = min(candidates, key=lambda p: p.distance)
            final_pockets.append(best_pocket)
        else:
            final_pockets.append(candidates[0])

    # --- Placeholder Logic ---
    if len(final_pockets) < 6:
        final_pockets = _estimate_missing_pockets(final_pockets, rect)

    # --- Final ID assignment and sorting ---
    all_locations_order = [
        Pocket_Location_On_Table.top_left,
        Pocket_Location_On_Table.top_right,
        Pocket_Location_On_Table.bottom_left,
        Pocket_Location_On_Table.bottom_right,
        Pocket_Location_On_Table.top_middle,
        Pocket_Location_On_Table.buttom_middle,
    ]
    final_pockets.sort(key=lambda p: all_locations_order.index(p.location) if p.location in all_locations_order else 99)
    for i, pocket in enumerate(final_pockets, 1):
        pocket.id = i
    
    detected_pockets = final_pockets

    # --- Drawing and Final Summary ---
    original_image_with_pockets = original_image.copy()
    for pocket in detected_pockets:
        draw_center_x = pocket.center[0] + padding
        draw_center_y = pocket.center[1] + padding
        cv2.circle(output_display, (draw_center_x, draw_center_y), int(pocket.radius), (0, 255, 0), 2)
        cv2.putText(output_display, str(pocket.id), (draw_center_x + int(pocket.radius), draw_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.circle(original_image_with_pockets, (pocket.center[0], pocket.center[1]), int(pocket.radius), (0, 0, 255), 3)

    debug_output_path = get_output_path("pocket_mask.jpg", sub_dir="black_white_detect")
    cv2.imwrite(debug_output_path, output_display)
    original_debug_path = get_output_path("original_with_pockets.jpg", sub_dir="black_white_detect")
    cv2.imwrite(original_debug_path, original_image_with_pockets)

    print(f"Found {len(detected_pockets)} final pockets.")
    print("Final Pocket Summary:")
    for p in detected_pockets:
        print(f"  ID: {p.id}, Location: {p.location}, Center: {p.center}, Radius: {p.radius}, Dist: {p.distance:.2f}")

    return detected_pockets, debug_output_path, original_debug_path


def crate_mask_table(
    input_path: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    original_image = cv2.imread(input_path)
    MASK_OUTPUT_PATH = get_output_path("01_felt_mask.jpg", sub_dir="black_white_detect")
    if original_image is None:
        raise FileNotFoundError(f"❌ Could not read input image: {input_path}")
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 40, 30], dtype=np.uint8)
    upper_blue = np.array([125, 255, 255], dtype=np.uint8)
    lower_green = np.array([35, 30, 30], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    colorful_mask = cv2.inRange(
        hsv_image, np.array([0, 40, 40], np.uint8), np.array([179, 255, 255], np.uint8)
    )
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    felt_mask = cv2.bitwise_or(blue_mask, green_mask)
    felt_mask = cv2.bitwise_and(felt_mask, colorful_mask)

    kernel = np.ones((5, 5), np.uint8)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    felt_mask = cv2.morphologyEx(felt_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        felt_mask, connectivity=8
    )
    image_height, image_width = felt_mask.shape[:2]
    image_area = image_height * image_width
    min_felt_area = int(0.005 * image_area)
    cleaned_felt_mask = np.zeros_like(felt_mask)
    for label_id in range(1, num_labels):
        _, _, _, _, stat_area = stats[label_id]
        if stat_area >= min_felt_area:
            cleaned_felt_mask[labels == label_id] = 255
    inverted_mask = cv2.bitwise_not(cleaned_felt_mask)
    binary_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(MASK_OUTPUT_PATH, binary_mask)

    return MASK_OUTPUT_PATH, binary_mask, original_image


def detect_balls_pipeline(input_path: str) -> List[Ball]:

    _, binary_mask, original_image = crate_mask_table(input_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    raw_balls = _circles_from_connected_components(
        binary_mask, gray_image, 0.00015, 0.0055
    )  # Use generic function
    print(
        f"DEBUG: detect_balls_pipeline - Raw balls found by connected components: {len(raw_balls)}"
    )

    ball_objects: List[Ball] = []
    for center_x, center_y, radius in raw_balls:
        print(
            f"DEBUG: detect_balls_pipeline - Checking raw ball: cx={center_x}, cy={center_y}, r={radius}"
        )
        if (
            get_ball_radius() - get_ball_radius_determinate()
            <= radius
            <= get_ball_radius() + get_ball_radius_determinate()
        ):
            ball = Ball(center=(int(center_x), int(center_y)), radius=int(radius))
            ball_objects.append(ball)
            print(
                f"✅ Ball detected: x={int(center_x)}, y={int(center_y)}, r={int(radius)}"
            )
        else:
            print(f"⚠️ Ignored ball with invalid radius r={int(radius)}")
            print(
                f"   (valid range: [{get_ball_radius()-get_ball_radius_determinate()}, {get_ball_radius()+get_ball_radius_determinate()}])"
            )
    print(f"🎱 Found {len(ball_objects)} balls.")
    return ball_objects


def detect_pockets_as_dataclasses(binary_mask, gray_image) -> List[PocketDetection]:
    """Converts raw pocket detections to PocketDetection dataclasses and filters by radius and location."""
    print(
        f"DEBUG: detect_pockets_as_dataclasses - Starting pocket filtering. binary_mask shape: {binary_mask.shape}, gray_image shape: {gray_image.shape}"
    )
    raw_pockets = _circles_from_connected_components(
        binary_mask, gray_image, 0.00005, 0.01
    )  # Use generic function
    print(
        f"DEBUG: detect_pockets_as_dataclasses - Found {len(raw_pockets)} raw pockets before radius/location filtering."
    )
    pockets: List[PocketDetection] = []
    image_height, image_width = binary_mask.shape[:2]
    print(
        f"DEBUG: detect_pockets_as_dataclasses - Image dimensions: height={image_height}, width={image_width}"
    )
    for center_x, center_y, radius in raw_pockets:
        print(
            f"DEBUG: detect_pockets_as_dataclasses - Checking raw pocket: cx={center_x}, cy={center_y}, r={radius}"
        )
        pocket_radius_min = get_pocket_radius() - get_pocket_radius_determinate()
        pocket_radius_max = get_pocket_radius() + get_pocket_radius_determinate()
        print(
            f"DEBUG: detect_pockets_as_dataclasses - Pocket valid radius range: [{pocket_radius_min:.2f}, {pocket_radius_max:.2f}]"
        )
        if pocket_radius_min <= radius <= pocket_radius_max:
            print(
                f"DEBUG: detect_pockets_as_dataclasses - Radius {radius} is within valid range."
            )
            if is_near_border((center_x, center_y), image_width, image_height):
                pocket = PocketDetection(
                    center=(int(center_x), int(center_y)), radius=int(radius)
                )
                pockets.append(pocket)
                print(
                    f"✅ Pocket detected: x={int(center_x)}, y={int(center_y)}, r={int(radius)}"
                )
            else:
                print(
                    f"⚠️ Ignored pocket not near border: x={int(center_x)}, y={int(center_y)}, r={int(radius)}"
                )
        else:
            print(f"⚠️ Ignored pocket with invalid radius r={int(radius)}")
            print(
                f"   (valid range: [{pocket_radius_min:.2f}, {pocket_radius_max:.2f}])"
            )
    return pockets


def detect_pockets_pipeline(original_image: np.ndarray) -> List[PocketDetection]:
    """Full pipeline for detecting pockets in an image."""
    print(f"DEBUG: detect_pockets_pipeline - Starting pocket detection.")
    MASK_OUTPUT_PATH = get_output_path(
        "01_pocket_mask.jpg", sub_dir="black_white_detect"
    )
    print(f"DEBUG: detect_pockets_pipeline - MASK_OUTPUT_PATH: {MASK_OUTPUT_PATH}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print(
        f"DEBUG: detect_pockets_pipeline - Converted to grayscale. gray_image shape: {gray_image.shape}"
    )

    # Apply a binary threshold to find dark areas (pockets)
    _, binary_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    print(
        f"DEBUG: detect_pockets_pipeline - Applied binary threshold. binary_mask shape: {binary_mask.shape}"
    )

    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    print(
        f"DEBUG: detect_pockets_pipeline - Applied morphology. binary_mask shape: {binary_mask.shape}"
    )

    cv2.imwrite(MASK_OUTPUT_PATH, binary_mask)
    print(f"DEBUG: detect_pockets_pipeline - Pocket mask saved to: {MASK_OUTPUT_PATH}")

    pocket_objects = detect_pockets_as_dataclasses(binary_mask, gray_image)
    print(f"🎱 Found {len(pocket_objects)} pockets.")
    return pocket_objects


def detect_only_pockets_and_draw(image_path: str) -> str:
    """
    Detects pockets from a given mask, draws them on the original image,
    and returns the path to the output image with pockets circled.
    """
    from const_numbers import set_table_length, set_table_width

    mask_path, binary_mask, original_image = crate_mask_table(image_path)
    set_table_width(original_image.shape[0])
    set_table_length(original_image.shape[1])
    # Load the original image
    if original_image is None:
        raise FileNotFoundError(f"❌ Could not read input image: {image_path}")
    print(
        f"DEBUG: detect_only_pockets_and_draw - Original image loaded. Shape: {original_image.shape}"
    )

    if binary_mask is None:
        raise FileNotFoundError(f"❌ Could not read mask image: {mask_path}")
    print(
        f"DEBUG: detect_only_pockets_and_draw - Mask loaded. Shape: {binary_mask.shape}"
    )

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print(
        f"DEBUG: detect_only_pockets_and_draw - Original image converted to grayscale. gray_image shape: {gray_image.shape}"
    )

    # Detect pockets using the provided mask and grayscale image
    detected_pockets = detect_pockets_as_dataclasses(binary_mask, gray_image)
    print(
        f"DEBUG: detect_only_pockets_and_draw - Found {len(detected_pockets)} pockets."
    )

    # Create a copy of the original image to draw on
    output_image = original_image.copy()

    # Draw pockets in blue
    for pocket in detected_pockets:
        center_x, center_y = pocket.center
        radius = pocket.radius
        draw_radius = max(8, int(radius))
        thickness = max(2, draw_radius // 5)
        cv2.circle(
            output_image, (center_x, center_y), draw_radius, (255, 0, 0), thickness
        )
        cv2.circle(
            output_image,
            (center_x, center_y),
            max(3, draw_radius // 6),
            (0, 0, 255),
            -1,
        )
        label_text = f"P({center_x},{center_y})"
        cv2.putText(
            output_image,
            label_text,
            (center_x + draw_radius + 6, center_y - draw_radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Define the output path
    output_path = get_output_path(
        "output_pockets_only.jpg", sub_dir="black_white_detect"
    )
    cv2.imwrite(output_path, output_image)
    print(f"✅ Image with only pockets circled saved to: {output_path}")

    return output_path


def get_table_rect():
    rect_path = Path(BASE_DIR / RECTANGLE_JSON_PATH)
    if rect_path.exists():
        with open(rect_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[DEBUG] Loaded existing data: {data}")


if __name__ == "__main__":
    FINAL_OUTPUT_PATH = get_output_path(
        "output_balls_and_pockets.jpg", sub_dir="black_white_detect"
    )

    # Load the original image once
    original_image_path = "output/debug/cropped_63a578eaa_test3_b2b.jpeg"
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise FileNotFoundError(f"❌ Could not read input image: {original_image_path}")

    # Detect balls
    detected_balls = detect_balls_pipeline(original_image_path)

    # Find corner pockets from the generated felt mask
    # The felt mask is saved as "01_felt_mask.jpg" by detect_balls_pipeline
    FELT_MASK_FOR_POCKETS = get_output_path(
        "01_felt_mask.jpg", sub_dir="black_white_detect"
    )
    detected_pockets = find_corner_pockets_from_mask(FELT_MASK_FOR_POCKETS)

    # Draw on a single output image
    output_image = original_image.copy()

    # Draw balls in green
    for ball in detected_balls:
        center_x, center_y = ball.center
        radius = ball.radius
        draw_radius = max(8, int(radius))
        thickness = max(2, draw_radius // 5)
        cv2.circle(
            output_image, (center_x, center_y), draw_radius, (0, 255, 0), thickness
        )
        cv2.circle(
            output_image,
            (center_x, center_y),
            max(3, draw_radius // 6),
            (0, 0, 255),
            -1,
        )
        label_text = f"B({center_x},{center_y})"
        cv2.putText(
            output_image,
            label_text,
            (center_x + draw_radius + 6, center_y - draw_radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Draw pockets in blue
    for pocket in detected_pockets:
        center_x, center_y = pocket.center
        radius = pocket.radius
        draw_radius = max(8, int(radius))
        thickness = max(2, draw_radius // 5)
        cv2.circle(
            output_image, (center_x, center_y), draw_radius, (255, 0, 0), thickness
        )
        cv2.circle(
            output_image,
            (center_x, center_y),
            max(3, draw_radius // 6),
            (0, 0, 255),
            -1,
        )
        label_text = f"P({center_x},{center_y})"
        cv2.putText(
            output_image,
            label_text,
            (center_x + draw_radius + 6, center_y - draw_radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(FINAL_OUTPUT_PATH, output_image)
    print(f"✅ Final image with balls and pockets saved to: {FINAL_OUTPUT_PATH}")

    # Test the new detect_only_pockets_and_draw function
    print("\n--- Testing detect_only_pockets_and_draw function ---")
    pockets_only_output_path = detect_only_pockets_and_draw(
        original_image_path, FELT_MASK_FOR_POCKETS
    )
    print(f"✅ Image with only pockets drawn saved to: {pockets_only_output_path}")

    print("[OK] Example finished.")
