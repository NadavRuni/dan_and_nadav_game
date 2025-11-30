"""
Provides the core OpenCV-based ball detection logic.

This module uses a pipeline of image enhancements and the Hough Circle Transform
to detect circular objects corresponding to pool balls in an image.
"""

import os
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import (
    Ball,
    PhotoData,
    Origin,
    Rectangle,
)
from const_numbers import get_ball_radius, get_ball_radius_determinate
from output_utils import get_output_path


def _preprocess_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing steps to an image to enhance it for circle detection.

    Steps include:
    1.  Convert to HSV and apply CLAHE (Contrast Limited Adaptive Histogram
        Equalization) to the Value channel to improve local contrast.
    2.  Convert back to BGR.
    3.  Convert to Grayscale and apply a Gaussian Blur.

    Args:
        image: The input BGR image.

    Returns:
        A preprocessed grayscale image ready for circle detection.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(value_channel)
    hsv_image[:, :, 2] = v_equalized

    processed_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    gray_image = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 1.5)
    return blurred_image


def _run_hough_circle_detection(
    gray_image: np.ndarray, is_main_image: bool
) -> Optional[np.ndarray]:
    """
    Runs the Hough Circle Transform to find circles in a grayscale image.

    Args:
        gray_image: The preprocessed grayscale image.
        is_main_image: A flag indicating if the image is the main, un-cropped
                       view, which affects detection sensitivity.

    Returns:
        A numpy array of detected circles [[x, y, r], ...], or None.
    """
    # The 'param2' (accumulator threshold) is adjusted based on the image type.
    # This is brittle and should be handled by a more robust config system.
    hough_param2 = 25 if is_main_image else 50
    min_radius = int(get_ball_radius() - 2 * get_ball_radius_determinate())
    max_radius = int(get_ball_radius() + 2 * get_ball_radius_determinate())

    Debugger.log(
        f"[OpenCV] Running HoughCircles with radius ~{get_ball_radius()} "
        f"(range: {min_radius}-{max_radius}), accumulator_thresh={hough_param2}"
    )

    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.0,  # Inverse ratio of accumulator resolution
        minDist=int(get_ball_radius() * 2),  # Min distance between centers
        param1=60,  # Upper Canny edge threshold
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    return circles


def detect_balls_opencv(
    input_dir: str, output_dir: str, parts_info: List[Dict[str, Any]]
) -> List[PhotoData]:
    """
    Detects balls in a series of image parts using OpenCV's Hough Circle Transform.

    For each image part specified in `parts_info`, this function loads the
    image, preprocesses it, detects circles, and saves the results as both
    a new `PhotoData` JSON file and debug images.

    Args:
        input_dir: The directory containing the cropped image parts.
        output_dir: The directory where output JSON and images will be saved.
        parts_info: A list of dictionaries, each describing an image part,
                    its original position, and dimensions.

    Returns:
        A list of PhotoData objects, one for each processed image part,
        containing the detected balls.
    """
    Debugger.log("Starting OpenCV ball detection pipeline.")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("out_debug", exist_ok=True)

    all_photos_data: List[PhotoData] = []
    total_balls_found = 0

    for part_info in parts_info:
        file_name = part_info["file_name"]
        image_path = os.path.join(input_dir, file_name)
        Debugger.log(f"[OpenCV] Processing {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            Debugger.error(f"Image at {image_path} could not be loaded. Skipping.")
            continue

        # Preprocess and detect
        is_main_image = file_name == "cut_main.png"
        gray_image = _preprocess_image_for_detection(image)
        circles = _run_hough_circle_detection(gray_image, is_main_image)

        detected_balls: List[Ball] = []
        visualization_image = image.copy()

        if circles is not None:
            circles_int = np.uint16(np.around(circles))
            min_r = get_ball_radius() - get_ball_radius_determinate()
            max_r = get_ball_radius() + get_ball_radius_determinate()

            for x, y, r in circles_int[0, :]:
                # Additional filtering by radius
                if not (min_r <= r <= max_r):
                    continue

                # Create Ball object with global coordinates
                origin_x = part_info["origin_x"]
                origin_y = part_info["origin_y"]
                global_x = origin_x + x
                global_y = origin_y + y
                detected_balls.append(
                    Ball(center=(int(global_x), int(global_y)), radius=int(r))
                )

                # Draw on visualization image
                cv2.circle(
                    visualization_image, (int(x), int(y)), int(r), (0, 0, 255), 3
                )

        Debugger.log(f"[OpenCV] Detected {len(detected_balls)} balls in {file_name}")
        total_balls_found += len(detected_balls)

        # Save visualization and PhotoData
        output_img_path = get_output_path(f"detect_{file_name}")
        cv2.imwrite(output_img_path, visualization_image)

        origin = Origin(x=part_info["origin_x"], y=part_info["origin_y"])
        rect = Rectangle(
            top_left=(origin.x, origin.y + part_info["height"]),
            top_right=(origin.x + part_info["width"], origin.y + part_info["height"]),
            bottom_left=(origin.x, origin.y),
            bottom_right=(origin.x + part_info["width"], origin.y),
        )
        photo_data = PhotoData(
            cut_name=file_name, origin=origin, rectangle=rect, balls=detected_balls
        )

        json_path = os.path.join(output_dir, f"{file_name.replace('.png', '.json')}")
        photo_data.save_json(json_path)
        Debugger.log(f"[OpenCV] Saved metadata to: {json_path}")
        all_photos_data.append(photo_data)

    Debugger.warn(
        f"[OpenCV] Total detected balls across all parts: {total_balls_found}"
    )
    return all_photos_data
