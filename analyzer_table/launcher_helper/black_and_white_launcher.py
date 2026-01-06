#!/usr/bin/env python3
"""
A launcher script for running the black and white ball detection pipeline.

This script takes an image path as input, runs the full ball detection
pipeline from the 'mark_balls_v4' module, and prints the results.
"""

import sys
from pathlib import Path
from typing import List

import cv2

# Add the 'black_white_detect' directory to the system path to allow imports.
# Note: This is generally not a good practice. Using packages and relative
# imports is preferred.
sys.path.append(str(Path(__file__).resolve().parent / "black_white_detect"))

from analyzer_table.black_white_detect.mark_balls_v4 import detect_balls_full_pipeline
from analyzer_table.launcher_helper.json_models import Ball


def run_ball_detection(image_path: str) -> List[Ball]:
    """
    Takes an image path as input and returns a list of detected Ball objects.

    This function loads an image, invokes the core detection pipeline, and
    returns the consolidated list of detected balls.

    Note:
        The function currently calls the detection pipeline twice with different
        parameters and concatenates the results. The purpose of the second call
        is unclear and should be investigated.

    Args:
        image_path: The path to the image file to be analyzed.

    Returns:
        A list of Ball objects detected in the image.

    Raises:
        FileNotFoundError: If the image file cannot be found at the given path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image file not found at: {image_path}")

    # Run the detection pipeline.
    # The first call is standard, the second has an unclear purpose (green?).
    balls_pass_one = detect_balls_full_pipeline(image_path)
    balls_pass_two = detect_balls_full_pipeline(image_path, True)
    all_detected_balls = balls_pass_one + balls_pass_two

    print(f"✅ Detected {len(all_detected_balls)} balls in total:")
    for i, ball in enumerate(all_detected_balls, start=1):
        print(f"  {i:02d}. Center={ball.center}, Radius={ball.radius}")

    return all_detected_balls


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python black_and_white_launcher.py <image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    try:
        detected_balls = run_ball_detection(input_image_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
