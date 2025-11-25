#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from output_utils import get_output_path
import cv2
import numpy as np
from typing import List, Tuple

from const_numbers import *
from analyzer_table.launcher_helper.json_models import Ball
from const_numbers import FELT_MASK_PATH


def preprocess_roi(roi_gray):
    """Applies a series of preprocessing steps to a region of interest."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized_roi = clahe.apply(roi_gray)
    denoised_roi = cv2.bilateralFilter(equalized_roi, d=7, sigmaColor=50, sigmaSpace=7)
    blurred_roi = cv2.GaussianBlur(denoised_roi, (5, 5), 0)
    return blurred_roi


def edge_support_ratio(edges, center_x, center_y, radius):
    """Calculates the ratio of edge pixels within a circular region."""
    height, width = edges.shape[:2]
    radius = int(max(1, radius))
    center_x, center_y = int(center_x), int(center_y)
    padding = 1
    roi_x1 = max(0, center_x - radius - padding)
    roi_x2 = min(width - 1, center_x + radius + padding)
    roi_y1 = max(0, center_y - radius - padding)
    roi_y2 = min(height - 1, center_y + radius + padding)
    if roi_x2 < roi_x1 or roi_y2 < roi_y1:
        return 0.0
    roi = edges[roi_y1 : roi_y2 + 1, roi_x1 : roi_x2 + 1]
    if roi.size == 0:
        return 0.0
    if roi.max() == 0:
        return 0.0
    support_ratio = float(roi.sum() > 0)
    return support_ratio


def refine_with_hough(gray_image, bbox_x, bbox_y, bbox_width, bbox_height, padding=20):
    """Refines a bounding box to a circle using Hough Circle Transform."""
    image_height, image_width = gray_image.shape[:2]
    roi_x1 = max(0, bbox_x - padding)
    roi_y1 = max(0, bbox_y - padding)
    roi_x2 = min(image_width, bbox_x + bbox_width + padding)
    roi_y2 = min(image_height, bbox_y + bbox_height + padding)
    roi = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None
    if roi.max() == 0:
        return None
    preprocessed_roi = preprocess_roi(roi)
    edges = cv2.Canny(preprocessed_roi, 60, 160)
    estimated_radius = 0.5 * min(bbox_width, bbox_height)

    def try_hough(hough_param2, min_radius_multiplier, max_radius_multiplier):
        min_radius = max(6, int(min_radius_multiplier * estimated_radius))
        max_radius = max(min_radius + 2, int(max_radius_multiplier * estimated_radius))
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
        circles = try_hough(18, 0.45, 1.9)
    if circles is None:
        return None
    best_circle, best_score = None, -1.0
    for circle in circles[0]:
        center_x, center_y, radius = circle
        if radius < 0.4 * estimated_radius:
            continue
        coverage_score = edge_support_ratio(edges, center_x, center_y, radius)
        score = coverage_score * radius
        if score > best_score:
            best_score = score
            best_circle = (center_x, center_y, radius)
    if best_circle is None:
        return None
    center_x, center_y, radius = best_circle
    return int(roi_x1 + center_x), int(roi_y1 + center_y), int(radius)


def touches_border(bbox, image_width, image_height, padding=3):
    """Checks if a bounding box touches the image border."""
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    return (
        bbox_x <= padding
        or bbox_y <= padding
        or (bbox_x + bbox_width) >= (image_width - 1 - padding)
        or (bbox_y + bbox_height) >= (image_height - 1 - padding)
    )


def balls_from_connected_components(binary_mask, gray_image):
    """Finds balls from connected components in a binary mask."""
    image_height, image_width = binary_mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    balls = []
    image_area = image_height * image_width
    min_component_area = int(0.00015 * image_area)
    max_component_area = int(0.0055 * image_area)
    MAX_RADIUS = 100
    for label_id in range(1, num_labels):
        stat_x, stat_y, stat_width, stat_height, stat_area = stats[label_id]
        if not min_component_area <= stat_area <= max_component_area:
            continue
        if touches_border(
            (stat_x, stat_y, stat_width, stat_height), image_width, image_height
        ):
            continue
        refined_circle = refine_with_hough(
            gray_image, stat_x, stat_y, stat_width, stat_height
        )
        if refined_circle:
            center_x, center_y, radius = refined_circle
        else:
            center_x, center_y = centroids[label_id]
            estimated_radius = int(0.5 * (stat_width + stat_height) / 2)
            radius = max(6, estimated_radius)
        if radius > MAX_RADIUS:
            continue
        balls.append((int(center_x), int(center_y), int(radius)))
    return sorted(balls, key=lambda item: item[2], reverse=True)


def detect_balls_as_dataclasses(binary_mask, gray_image) -> List[Ball]:
    """Converts raw ball detections to Ball dataclasses and filters by radius."""
    raw_balls = balls_from_connected_components(binary_mask, gray_image)
    balls: List[Ball] = []
    for center_x, center_y, radius in raw_balls:
        if (
            get_ball_radius() - get_ball_radius_determinate()
            <= radius
            <= get_ball_radius() + get_ball_radius_determinate()
        ):
            ball = Ball(center=(int(center_x), int(center_y)), radius=int(radius))
            balls.append(ball)
            print(
                f"✅ Ball detected: x={int(center_x)}, y={int(center_y)}, r={int(radius)}"
            )
        else:
            print(f"⚠️ Ignored ball with invalid radius r={int(radius)}")
            print(
                f"   (valid range: [{get_ball_radius()-get_ball_radius_determinate()}, {get_ball_radius()+get_ball_radius_determinate()}])"
            )
    return balls


def detect_balls_full_pipeline(input_path: str):
    """Full pipeline for detecting balls in an image."""
    MASK_OUTPUT_PATH = get_output_path(FELT_MASK_PATH, sub_dir="black_white_detect")
    OUTPUT_PATH = get_output_path(
        "output_marked_balls.jpg", sub_dir="black_white_detect"
    )
    original_image = cv2.imread(input_path)
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
        stat_x, stat_y, stat_width, stat_height, stat_area = stats[label_id]
        if stat_area >= min_felt_area:
            cleaned_felt_mask[labels == label_id] = 255

    inverted_mask = cv2.bitwise_not(cleaned_felt_mask)
    binary_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(MASK_OUTPUT_PATH, binary_mask)
    print(f"🖤 Mask saved to: {MASK_OUTPUT_PATH}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    ball_objects = detect_balls_as_dataclasses(binary_mask, gray_image)
    print(f"🎱 Found {len(ball_objects)} balls.")

    output_image = original_image.copy()
    for ball in ball_objects:
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
        label_text = f"({center_x},{center_y})"
        cv2.putText(
            output_image,
            label_text,
            (center_x + draw_radius + 6, center_y - draw_radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(OUTPUT_PATH, output_image)
    print(f"✅ Final image saved to: {OUTPUT_PATH}")
    return ball_objects


if __name__ == "__main__":
    example_image_path = "photos/img_start.jpeg"
    detected_balls = detect_balls_full_pipeline(example_image_path)
    print(
        "[OK] Example finished, first 3 balls:", [b.center for b in detected_balls[:3]]
    )
