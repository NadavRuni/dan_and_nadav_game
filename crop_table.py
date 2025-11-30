"""
Performs a perspective warp on an image to get a top-down view of the table.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import requests
import shutil

from analyzer_table.launcher_helper.json_models import Rectangle
from output_utils import get_output_path


def _download_image_if_needed(image_path: str, output_dir: Path) -> Path:
    """Downloads an image if the path is a URL, otherwise returns the local path."""
    if image_path.startswith("http"):
        filename = Path(image_path).name
        local_path = output_dir / filename
        print(f"[DEBUG] Downloading image from URL: {image_path}")
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"[DEBUG] Saved image to: {local_path}")
        return local_path
    return Path(image_path)


def crop_image_by_rectangle(
    rectangle: Rectangle,
    image_path: str,
    output_dir: Path,
    display_size: Dict[str, float],
    original_size: Dict[str, float],
) -> Tuple[str, Rectangle]:
    """
    Crops and performs a perspective warp on an image based on a rectangle.

    This function takes a rectangle defined on a (potentially scaled-down)
    display image, scales the coordinates to the original image size, and then
    applies a perspective transform to get a straightened, top-down view of
    the table.

    Args:
        rectangle: The Rectangle object with coordinates from the display image.
        image_path: The path or URL to the original image.
        output_dir: The directory to save downloaded and cropped images.
        display_size: A dictionary {'width': w, 'height': h} of the display area.
        original_size: A dictionary {'width': w, 'height': h} of the original image.

    Returns:
        A tuple containing:
        - The path to the newly created cropped and warped image.
        - A new Rectangle object that corresponds to the dimensions of the
          new warped image.
    """
    print("[DEBUG] Starting crop_image_by_rectangle")
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = _download_image_if_needed(image_path, output_dir)

    image = cv2.imread(str(local_path))
    if image is None:
        raise FileNotFoundError(f"❌ Could not load image: {local_path}")

    # Calculate scaling factors
    scale_x = original_size["width"] / display_size["width"]
    scale_y = original_size["height"] / display_size["height"]
    print(f"[DEBUG] Scaling factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # Scale the input rectangle points to the original image dimensions
    source_points = np.float32(
        [
            [rectangle.top_left[0] * scale_x, rectangle.top_left[1] * scale_y],
            [rectangle.top_right[0] * scale_x, rectangle.top_right[1] * scale_y],
            [rectangle.bottom_right[0] * scale_x, rectangle.bottom_right[1] * scale_y],
            [rectangle.bottom_left[0] * scale_x, rectangle.bottom_left[1] * scale_y],
        ]
    )

    # Calculate the dimensions of the new warped image
    width_a = np.linalg.norm(source_points[2] - source_points[3])
    width_b = np.linalg.norm(source_points[1] - source_points[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(source_points[1] - source_points[2])
    height_b = np.linalg.norm(source_points[0] - source_points[3])
    max_height = int(max(height_a, height_b))
    print(f"[DEBUG] Target crop size: {max_width}x{max_height}")

    # Define the destination points for the top-down perspective
    destination_points = np.float32(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ]
    )

    # Compute the perspective transform matrix and apply it
    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_image = cv2.warpPerspective(
        image, perspective_matrix, (max_width, max_height)
    )

    # Save the result
    cropped_path = get_output_path(f"cropped_{Path(image_path).stem}.jpeg")
    cv2.imwrite(cropped_path, warped_image)
    print(f"[DEBUG] Cropped image saved to: {cropped_path}")

    # Create a new rectangle that represents the entire warped image
    new_rectangle = Rectangle(
        top_left=(0, 0),
        top_right=(max_width - 1, 0),
        bottom_left=(0, max_height - 1),
        bottom_right=(max_width - 1, max_height - 1),
    )
    return cropped_path, new_rectangle
