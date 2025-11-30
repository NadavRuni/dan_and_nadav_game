"""
A helper utility to crop and save images of detected balls.

This module provides a function that takes a list of Ball objects and the
original image, and saves a cropped image for each ball to a specified
output directory.
"""

import os
from typing import List

import cv2

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Ball
from output_utils import get_output_path


def crop_and_save_balls(
    image_path: str, balls: List[Ball], output_dir: str = "out/balls"
) -> None:
    """
    Crops a square region for each ball and saves it as a PNG file.

    For each ball in the list, this function calculates a bounding box around
    its center, crops the region from the main image, and saves it to the
    output directory. It also mutates the Ball objects by setting their
    'single_ball_path' attribute to the path of the saved file.

    Args:
        image_path: The path to the source image.
        balls: A list of Ball objects to be cropped and saved. The objects in
               this list will be modified in-place.
        output_dir: The directory where the cropped ball images will be saved.
    """
    Debugger.log(f"🖼️ Cropping and saving {len(balls)} balls from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        Debugger.error(f"❌ Failed to load image from {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    img_height, img_width = image.shape[:2]

    for i, ball in enumerate(balls, start=1):
        center_x, center_y = map(int, ball.center)
        # Use a slightly larger radius for a better crop
        radius = int(ball.radius * 1.3)

        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(img_width, center_x + radius)
        y2 = min(img_height, center_y + radius)

        region_of_interest = image[y1:y2, x1:x2]
        if region_of_interest.size == 0:
            Debugger.warn(f"⚠️ Empty ROI for ball at center {ball.center}, skipping.")
            continue

        filename = f"ball_{i}_{center_x}_{center_y}.png"
        ball_path = get_output_path(filename, sub_dir="balls")
        cv2.imwrite(ball_path, region_of_interest)

        # This mutation of the input object is not ideal.
        ball.single_ball_path = ball_path
        Debugger.log(f"💾 Saved ball #{i} → {ball_path}")

    Debugger.log("✅ Finished cropping all balls.")
