"""
Refines pocket locations by detecting circles within their cropped images.

This module processes a list of Pocket objects, each with a path to a cropped
image of the pocket area. It uses the Hough Circle Transform to find the precise
circular opening of the pocket in the image. The pocket's center coordinates
and radius are then updated based on this more accurate detection.
"""

import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from analyzer_table.detect_ball.Debugger import Debugger
from const_numbers import get_crop_half_size
from game_class.C_pocket import Pocket

# --- Internal Helper Functions ---


def _load_pocket_image(pocket: Pocket) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Loads the image for a single pocket and validates it.

    Args:
        pocket: The Pocket object to load the image for.

    Returns:
        A tuple containing the loaded image as a numpy array and its path.
        Returns (None, None) if the image does not exist or fails to load.
    """
    image_path = pocket.pocket_img_path
    if not os.path.exists(image_path):
        Debugger.log(f"⚠️ Image not found for pocket {pocket.id}: {image_path}")
        return None, None

    image = cv2.imread(image_path)
    if image is None:
        Debugger.log(f"❌ Failed to load pocket image: {image_path}")
        return None, None

    return image, image_path


def _detect_circle(image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Detects a circle in an image using the Hough Circle Transform.

    Args:
        image: The input image.

    Returns:
        A tuple (x, y, r) representing the center and radius of the detected
        circle, or None if no circle is found.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.medianBlur(gray_image, 5)

    # Note: These parameters are sensitive and may need tuning.
    circles = cv2.HoughCircles(
        image=gray_image,
        method=cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution
        minDist=10,  # Minimum distance between detected centers
        param1=100,  # Upper threshold for the internal Canny edge detector
        param2=15,  # Threshold for center detection
        minRadius=5,
        maxRadius=int(min(image.shape[:2]) / 2),
    )

    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        x, y, r = circles_int[0, 0]
        return int(x), int(y), int(r)

    return None


def _draw_and_save_circle_visualization(
    image: np.ndarray, circle_data: Optional[Tuple[int, int, int]], image_path: str
) -> str:
    """
    Draws the detected circle on the image and saves it for debugging.

    A new file is created with the suffix '_cycle'.

    Args:
        image: The original image to draw on.
        circle_data: A tuple (x, y, r) of the circle to draw.
        image_path: The path to the original image.

    Returns:
        The path to the newly saved image with the visualization.
    """
    base, ext = os.path.splitext(image_path)
    new_path = f"{base}_cycle{ext}"

    if circle_data:
        x, y, r = circle_data
        # Draw the circle circumference in green
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        # Draw the circle center in red
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
        Debugger.log(f"✅ Circle drawn at ({x},{y}), radius={r}")
    else:
        Debugger.log("⚠️ No circle detected — saving original image for review.")

    cv2.imwrite(new_path, image)
    Debugger.log(f"💾 Saved circle detection result to: {new_path}")
    return new_path


# --- Main Function ---


def mark_pocket_circles(
    all_pockets: List[Pocket], crop_half_size: int = get_crop_half_size()
) -> None:
    """
    Iterates through pockets, detects their circular opening, and updates them.

    For each pocket, this function loads its cropped image, attempts to find a
    circle, and then updates the pocket's global coordinates and radius based
    on the find. It modifies the Pocket objects in the input list directly.

    Args:
        all_pockets: A list of Pocket objects to be processed.
        crop_half_size: The half-size of the original crop area, used to
                          calculate the global coordinates.
    """
    Debugger.log("🎱 Starting pocket circle marking and coordinate refinement...")

    for pocket in all_pockets:
        image, path = _load_pocket_image(pocket)
        if image is None:
            continue

        circle_data = _detect_circle(image)
        new_path = _draw_and_save_circle_visualization(image, circle_data, path)
        pocket.pocket_img_path = new_path

        if circle_data:
            local_x, local_y, local_r = circle_data

            # The top-left corner of the original crop box
            origin_cx, origin_cy = pocket.center
            crop_x1 = max(0, origin_cx - crop_half_size)
            crop_y1 = max(0, origin_cy - crop_half_size)

            # The new global center is the crop origin + the local circle center
            global_x = int(crop_x1 + local_x)
            global_y = int(crop_y1 + local_y)

            # Adjust radius for a more conservative estimate
            adjusted_radius = int(local_r * 0.8)

            pocket.pocket_img_cordinates_on_table = (global_x, global_y)
            pocket.radius = adjusted_radius

            Debugger.log(
                f"📍 Pocket {pocket.id}: local=({local_x},{local_y}), "
                f"crop_origin=({crop_x1},{crop_y1}) → "
                f"new_global_center=({global_x},{global_y})"
            )
        else:
            # If no circle is found, fall back to the original center
            pocket.pocket_img_cordinates_on_table = pocket.center
            Debugger.log(
                f"⚠️ No circle found for pocket {pocket.id} — "
                f"using original center: {pocket.center}"
            )

    Debugger.log("✅ Finished marking pocket circles and updating coordinates.")
