"""
A utility for analyzing ball colors to identify the whitest and darkest balls.

This module provides a function that iterates through a list of detected balls,
analyzes the brightness of each one, and identifies which are the most likely
candidates for being the white and black balls.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Ball
from output_utils import get_output_path


def find_whitest_and_darkest_balls(
    image_path: str, balls: List[Ball], output_dir: str = "out/balls"
) -> Tuple[Optional[Ball], Optional[Ball]]:
    """
    Analyzes a list of balls from an image to find the whitest and darkest ones.

    This function reads an image, converts it to the HSV color space, and then
    calculates the mean brightness (Value channel) for each ball's region of
    interest (ROI). It keeps track of the balls with the highest and lowest
    mean brightness. It also saves a cropped image of each ball for debugging.

    Args:
        image_path: Path to the source image file.
        balls: A list of Ball objects to be analyzed.
        output_dir: The directory where cropped ball images will be saved.

    Returns:
        A tuple containing the whitest Ball object and the darkest Ball
        object found. Returns (None, None) if the image cannot be loaded.
    """
    Debugger.log(f"🧠 Analyzing {len(balls)} balls to find the whitest and darkest...")
    image = cv2.imread(image_path)
    if image is None:
        Debugger.error(f"❌ Failed to load image from {image_path}")
        return None, None

    height, width = image.shape[:2]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    os.makedirs(output_dir, exist_ok=True)

    whitest_ball: Optional[Ball] = None
    darkest_ball: Optional[Ball] = None
    max_brightness = -1.0
    min_brightness = float("inf")

    for i, ball in enumerate(balls, 1):
        center_x, center_y = map(int, ball.center)
        # Increase radius to ensure the entire ball is captured
        radius = int(ball.radius * 1.3)

        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(width, center_x + radius)
        y2 = min(height, center_y + radius)

        roi_hsv = hsv_image[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            Debugger.warn(f"⚠️ Ball #{i} has an empty Region of Interest, skipping.")
            continue

        # Calculate brightness from the 'Value' channel of HSV
        value_channel = roi_hsv[:, :, 2]
        mean_brightness = np.mean(value_channel)
        Debugger.log(
            f"⚪ Ball #{i}: center=({center_x},{center_y}), r={radius}, "
            f"brightness={mean_brightness:.2f}"
        )

        # Save a cropped image of the ball for debugging
        ball_image_bgr = image[y1:y2, x1:x2]
        ball_image_path = get_output_path(
            f"ball_{i}_({center_x}_{center_y})_r{radius}.png", sub_dir="ball_color"
        )
        cv2.imwrite(ball_image_path, ball_image_bgr)
        ball.single_ball_path = ball_image_path

        if mean_brightness > max_brightness:
            max_brightness = mean_brightness
            whitest_ball = ball

        if mean_brightness < min_brightness:
            min_brightness = mean_brightness
            darkest_ball = ball

    if whitest_ball:
        Debugger.log(
            f"🏆 Whitest ball → center={whitest_ball.center}, "
            f"r={whitest_ball.radius}, brightness={max_brightness:.2f}"
        )
    if darkest_ball:
        Debugger.log(
            f"⚫ Darkest ball → center={darkest_ball.center}, "
            f"r={darkest_ball.radius}, brightness={min_brightness:.2f}"
        )

    Debugger.log(f"🗂️ All ball images saved under: {output_dir}")
    return whitest_ball, darkest_ball
