# analyzer_table/border_and_pocket/main.py

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# --- Setup sys.path ---
# Add project root to allow sibling imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports from project ---
from analyzer_table.black_white_detect.detect_balls_and_pockets import crate_mask_table
from analyzer_table.border_and_pocket.crop_table import remove_white_border
from output_utils import get_output_path
from analyzer_table.launcher_helper.json_models import Rectangle  # Import Rectangle


def process_image_for_border(
    image_path: str,
) -> tuple[str, np.ndarray, np.ndarray, Rectangle | None]:
    """
    Loads an image, creates a binary mask, removes a white border if present,
    and saves the results and debug images.
    Returns:
        A tuple of (final_mask_output_path, binary_mask, cropped_original_photo, rectangle_obj).
        rectangle_obj will be None if no border was cropped.
    """
    print(f"--- Starting Border Removal Process for: {image_path} ---")

    # 1. Create binary mask from the original image
    print("Step 1: Creating binary mask from the image...")
    try:
        mask_output_path, binary_mask, original_image = crate_mask_table(image_path)
        print(f"  - Successfully created mask. Shape: {binary_mask.shape}")
        print(f"  - Mask saved to: {mask_output_path}")
    except FileNotFoundError as e:
        print(f"  - ERROR: {e}")
        return "", np.array([]), np.array([]), None
    except Exception as e:
        print(f"  - An unexpected error occurred during mask creation: {e}")
        return "", np.array([]), np.array([]), None

    cropped_original_photo = original_image.copy()
    rectangle_obj = None

    # 2. Try to remove the white border
    print("Step 2: Checking for and removing white border...")
    processed_mask, crop_coords = remove_white_border(binary_mask)
    final_mask_output_path = get_output_path(
        "final_processed_mask.jpg", sub_dir="border_and_pocket"
    )

    # 3. Check if the mask was changed
    if crop_coords:
        print(f"  - Border detected and cropped!")
        print(f"  - Original mask shape: {binary_mask.shape}")
        print(f"  - Final mask shape:    {processed_mask.shape}")

        # Save the new, cropped mask
        cv2.imwrite(final_mask_output_path, processed_mask)
        print(f"  - Saved cropped mask to: {final_mask_output_path}")

        # Draw the cropping rectangle on the original image for debugging
        print("  - Generating debug images with crop rectangle...")
        debug_image = original_image.copy()
        top, bottom = crop_coords["top"], crop_coords["bottom"]
        left, right = crop_coords["left"], crop_coords["right"]

        # Save the original photo, cropped by the result
        cropped_original_photo = original_image[top : bottom + 1, left : right + 1]
        output_debug_original_cropped_path = get_output_path(
            "debug_original_cropped.jpg", sub_dir="border_and_pocket"
        )
        cv2.imwrite(output_debug_original_cropped_path, cropped_original_photo)
        print(
            f"  - Saved debug original photo cropped to: {output_debug_original_cropped_path}"
        )

        # Create Rectangle object with coordinates relative to the cropped image
        cropped_height, cropped_width = cropped_original_photo.shape[:2]
        rectangle_obj = Rectangle(
            top_left=(0, 0),
            top_right=(cropped_width, 0),
            bottom_left=(0, cropped_height),
            bottom_right=(cropped_width, cropped_height),
        )

        # Draw on original image
        cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 3)
        output_debug_path = get_output_path(
            "debug_border_crop.jpg", sub_dir="border_and_pocket"
        )
        cv2.imwrite(output_debug_path, debug_image)
        print(f"  - Saved debug image with crop rectangle to: {output_debug_path}")

        # Draw on mask image
        debug_mask_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_mask_image, (left, top), (right, bottom), (0, 255, 0), 3)
        output_debug_mask_path = get_output_path(
            "debug_mask_crop.jpg", sub_dir="border_and_pocket"
        )
        cv2.imwrite(output_debug_mask_path, debug_mask_image)
        print(f"  - Saved debug mask with crop rectangle to: {output_debug_mask_path}")

    else:
        print("  - No removable border detected. Mask remains unchanged.")
        height, width = original_image.shape[:2]
        rectangle_obj = Rectangle(top_left=(0,0), top_right=(width,0), bottom_left=(0,height), bottom_right=(width,height))
        # Save the original mask as the final mask
        cv2.imwrite(final_mask_output_path, processed_mask)
        print(f"  - Unchanged mask saved to: {final_mask_output_path}")

    print("--- Process Finished ---")
    return final_mask_output_path, binary_mask, cropped_original_photo, rectangle_obj


if __name__ == "__main__":
    image_path = "uploads/70ace7279_test2_wall_shot.jpeg"
    final_mask_path, original_binary_mask, cropped_original, detected_rectangle = (
        process_image_for_border(image_path)
    )
    if detected_rectangle:
        print(f"Detected Rectangle: {detected_rectangle}")
