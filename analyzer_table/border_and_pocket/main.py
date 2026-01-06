"""
Main pipeline for processing the table border from an image.

This module orchestrates the process of creating a binary mask of the table,
detecting and cropping out any surrounding white border, and generating debug
visualizations of the process.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# Add project root to allow sibling imports. This is not a recommended practice.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analyzer_table.black_white_detect.detect_balls_and_pockets import (
    create_felt_mask,
)
from analyzer_table.border_and_pocket.crop_table import (
    crop_to_content_by_projection,
)
from analyzer_table.launcher_helper.json_models import Rectangle
from output_utils import get_output_path


def _save_border_debug_images(
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    crop_coords: dict,
    cropped_original_photo: np.ndarray,
) -> str:
    """Saves a set of debug images related to the border crop."""
    top, left, right, bottom = (
        crop_coords["top"],
        crop_coords["left"],
        crop_coords["right"],
        crop_coords["bottom"],
    )

    # Save the cropped version of the original photo
    cropped_original_path = get_output_path(
        "debug_original_cropped.jpg", sub_dir="border_and_pocket"
    )
    cv2.imwrite(cropped_original_path, cropped_original_photo)

    # Draw rectangle on a copy of the original image
    debug_image = original_image.copy()
    cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 3)
    debug_border_path = get_output_path(
        "debug_border_crop.jpg", sub_dir="border_and_pocket"
    )
    cv2.imwrite(debug_border_path, debug_image)

    # Draw rectangle on a copy of the mask
    debug_mask_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(debug_mask_image, (left, top), (right, bottom), (0, 255, 0), 3)
    debug_mask_path = get_output_path(
        "debug_mask_crop.jpg", sub_dir="border_and_pocket"
    )
    cv2.imwrite(debug_mask_path, debug_mask_image)

    print(f"  - Saved debug images to 'output/border_and_pocket/'")
    return cropped_original_path


def process_image_for_border(
    image_path: str,
) -> Tuple[str, np.ndarray, np.ndarray, Optional[Rectangle], str]:
    """
    Loads an image, creates a binary mask, removes a white border, and saves results.

    Args:
        image_path: The path to the source image.

    Returns:
        A tuple containing:
        - Path to the final processed (and possibly cropped) mask.
        - The original binary mask (before cropping).
        - The cropped original color photo (or a copy of the original if no crop).
        - A Rectangle object representing the final table area.
        - Path to the cropped original photo.
    """
    print(f"--- Starting Border Removal Process for: {image_path} ---")

    # 1. Create binary mask from the original image
    print("Step 1: Creating binary mask from the image...")
    try:
        _, binary_mask, original_image = create_felt_mask(image_path)
    except (FileNotFoundError, Exception) as e:
        print(f"  - ERROR during mask creation: {e}")
        return "", np.array([]), np.array([]), None, ""

    # 2. Try to remove the white border using projection profiling
    print("Step 2: Checking for and removing white border...")
    processed_mask, crop_coords = crop_to_content_by_projection(binary_mask)
    final_mask_path = get_output_path(
        "final_processed_mask.jpg", sub_dir="border_and_pocket"
    )

    cropped_original_photo = original_image.copy()
    rectangle_obj = None
    cropped_photo_path = ""

    # 3. Handle case where a border was detected and cropped
    if crop_coords:
        print("  - Border detected and cropped!")
        top, bottom = crop_coords["top"], crop_coords["bottom"]
        left, right = crop_coords["left"], crop_coords["right"]

        # Crop the original photo to match the cropped mask
        cropped_original_photo = original_image[top : bottom + 1, left : right + 1]

        # Create a Rectangle object representing the new, full image area
        new_height, new_width = cropped_original_photo.shape[:2]
        rectangle_obj = Rectangle(
            top_left=(0, 0),
            top_right=(new_width, 0),
            bottom_left=(0, new_height),
            bottom_right=(new_width, new_height),
        )

        cropped_photo_path = _save_border_debug_images(
            original_image, binary_mask, crop_coords, cropped_original_photo
        )

    else:
        print("  - No removable border detected. Mask remains unchanged.")
        height, width = original_image.shape[:2]
        rectangle_obj = Rectangle(
            top_left=(0, 0),
            top_right=(width, 0),
            bottom_left=(0, height),
            bottom_right=(width, height),
        )
        cropped_photo_path = get_output_path(
            "debug_original_uncropped.jpg", sub_dir="border_and_pocket"
        )
        cv2.imwrite(cropped_photo_path, cropped_original_photo)

    cv2.imwrite(final_mask_path, processed_mask)
    print(f"  - Final mask saved to: {final_mask_path}")
    print("--- Process Finished ---")

    return (
        final_mask_path,
        binary_mask,  # Return original mask for context
        cropped_original_photo,
        rectangle_obj,
        cropped_photo_path,
    )


if __name__ == "__main__":
    # Example usage
    image_to_process = "uploads/70ace7279_test2_wall_shot.jpeg"
    _, _, _, detected_rect, _ = process_image_for_border(image_to_process)
    if detected_rect:
        print(f"Process complete. Detected Rectangle: {detected_rect}")
