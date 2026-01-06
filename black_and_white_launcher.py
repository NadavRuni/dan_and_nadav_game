"""
A launcher script for running a legacy version of the ball detection algorithm.

Warning:
    This file is one of two files named 'black_and_white_launcher.py' in this
    project. This duplication is a major source of confusion and should be
    resolved. This version appears to be older or different from the one in
    'analyzer_table/launcher_helper/'.
"""

import sys
from pathlib import Path

import cv2

# This is not a recommended practice for managing imports.
sys.path.append(str(Path(__file__).resolve().parent / "black_white_detect"))

# These imports are likely incorrect if this file is run from the root directory
# without further sys.path manipulation.
from black_white_detect.mark_balls_v4 import detect_balls_as_dataclasses
from json_models import Ball


def run_ball_detection(image_path: str) -> list[Ball]:
    """
    Takes an image, creates a binary mask, and runs ball detection.

    Args:
        image_path: The path to the image file.

    Returns:
        A list of detected Ball objects.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image file not found at: {image_path}")

    # Create a binary mask based on blue/green colors (table felt)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = (80, 40, 30)
    upper_blue = (125, 255, 255)
    mask_felt = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_inv = cv2.bitwise_not(mask_felt)
    _, mask_bin = cv2.threshold(mask_inv, 127, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run the detection function
    balls: list[Ball] = detect_balls_as_dataclasses(mask_bin, gray)

    print(f"✅ Detected {len(balls)} balls:")
    for i, b in enumerate(balls, start=1):
        print(f"  {i:02d}. Center={b.center}, Radius={b.radius}")

    return balls


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python black_and_white_launcher.py <image_path>")
        sys.exit(1)

    image_path_arg = sys.argv[1]
    try:
        detected_balls = run_ball_detection(image_path_arg)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
