"""
Detects the rectangular boundaries of a pool table in an image.

This module uses computer vision techniques to find the main horizontal and
vertical lines in an image, which are then used to define the rectangle
representing the pool table's playing surface.
"""

from typing import Optional, Union

import cv2
import numpy as np

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Rectangle

# --- Constants for Computer Vision Parameters ---
# These values may require tuning for different lighting or camera conditions.

# Gaussian Blur parameters
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA_X = 1.2

# Canny Edge Detection parameters
CANNY_LOWER_THRESHOLD = 50
CANNY_UPPER_THRESHOLD = 150

# Hough Line Transform parameters
HOUGH_RHO_RESOLUTION = 1  # pixels
HOUGH_THETA_RESOLUTION = np.pi / 180  # radians
HOUGH_VOTE_THRESHOLD = 120
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 30

# Angle thresholds for classifying lines as horizontal or vertical
HORIZONTAL_ANGLE_THRESHOLD = 10  # degrees
VERTICAL_ANGLE_RANGE = (80, 100)  # degrees


def find_table_rectangle(image_source: Union[str, np.ndarray]) -> Optional[Rectangle]:
    """
    Finds the largest rectangle in an image, likely corresponding to the pool table.

    The process involves these steps:
    1.  Convert the image to grayscale and apply a Gaussian blur.
    2.  Use the Canny edge detector to find edges.
    3.  Use the Hough Line Transform to detect straight lines in the edge map.
    4.  Filter lines into horizontal and vertical groups based on their angle.
    5.  Determine the outermost lines from each group to form the rectangle.

    Args:
        image_source: Either a file path to an image or a numpy array
                      representing an image in BGR format.

    Returns:
        A Rectangle object defining the table boundaries, or None if a rectangle
        cannot be reliably detected.
    """
    if isinstance(image_source, str):
        Debugger.log(f"🖼  Loading image from path: {image_source}")
        image = cv2.imread(image_source)
        if image is None:
            Debugger.error(f"❌ Failed to load image from {image_source}")
            return None
    else:
        image = image_source
        Debugger.log("🧠 Received image object directly for rectangle detection")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(
        gray_image, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X
    )
    edges = cv2.Canny(blurred_image, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD)

    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO_RESOLUTION,
        theta=HOUGH_THETA_RESOLUTION,
        threshold=HOUGH_VOTE_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        Debugger.warn("⚠️ No lines detected in image — cannot find rectangle.")
        return None

    horizontals = []
    verticals = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < HORIZONTAL_ANGLE_THRESHOLD:
            horizontals.append((y1 + y2) / 2)
        elif VERTICAL_ANGLE_RANGE[0] < abs(angle) < VERTICAL_ANGLE_RANGE[1]:
            verticals.append((x1 + x2) / 2)

    if len(horizontals) < 2 or len(verticals) < 2:
        Debugger.warn("⚠️ Not enough horizontal/vertical lines to define a rectangle.")
        return None

    horizontals.sort()
    verticals.sort()

    top_y = int(horizontals[0])
    bottom_y = int(horizontals[-1])
    left_x = int(verticals[0])
    right_x = int(verticals[-1])

    Debugger.log(
        f"✅ Rectangle edges found: left={left_x}, right={right_x}, "
        f"top={top_y}, bottom={bottom_y}"
    )

    return Rectangle(
        top_left=(left_x, top_y),
        top_right=(right_x, top_y),
        bottom_left=(left_x, bottom_y),
        bottom_right=(right_x, bottom_y),
    )
