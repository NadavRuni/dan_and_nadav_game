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
from analyzer_table.crop_table import remove_white_border
from output_utils import get_output_path

def process_image_for_border(image_path: str):
    """
    Loads an image, creates a binary mask, removes a white border if present,
    and saves the results and debug images.
    """
    print(f"--- Starting Border Removal Process for: {image_path} ---")

    # 1. Create binary mask from the original image
    print("Step 1: Creating binary mask from the image...")
    try:
        _, binary_mask, original_image = crate_mask_table(image_path)
        print(f"  - Successfully created mask. Shape: {binary_mask.shape}")
    except FileNotFoundError as e:
        print(f"  - ERROR: {e}")
        return
    except Exception as e:
        print(f"  - An unexpected error occurred during mask creation: {e}")
        return

    # 2. Try to remove the white border
    print("Step 2: Checking for and removing white border...")
    processed_mask = remove_white_border(binary_mask)

    final_mask = None
    final_mask_output_path = get_output_path("final_processed_mask.jpg", sub_dir="border_and_pocket")
    was_cropped = False

    # 3. Check if the mask was changed
    if processed_mask.shape != binary_mask.shape:
        print(f"  - Border detected and cropped!")
        print(f"  - Original mask shape: {binary_mask.shape}")
        print(f"  - Cropped mask shape:  {processed_mask.shape}")
        final_mask = processed_mask
        was_cropped = True

        # 4b. Draw the cropping rectangle on the original image for debugging
        print("  - Generating debug image with crop rectangle...")
        black_pixels = np.where(binary_mask == 0)
        if black_pixels[0].size > 0:
            top = np.min(black_pixels[0])
            bottom = np.max(black_pixels[0])
            left = np.min(black_pixels[1])
            right = np.max(black_pixels[1])

            debug_image = original_image.copy()
            # Draw a bright green rectangle on the debug image
            cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 3)

            output_debug_path = get_output_path("debug_border_crop.jpg", sub_dir="border_and_pocket")
            cv2.imwrite(output_debug_path, debug_image)
            print(f"  - Saved debug image with crop rectangle to: {output_debug_path}")
        else:
            print("  - WARNING: Could not draw debug rectangle because no black pixels were found in the original mask.")
    else:
        print("  - No removable white border was detected. Mask remains unchanged.")
        final_mask = binary_mask
    
    # 4a. Always save the final (cropped or original) mask
    if final_mask is not None:
        cv2.imwrite(final_mask_output_path, final_mask)
        print(f"  - Final processed mask saved to: {final_mask_output_path}")

    print("--- Process Finished ---")


if __name__ == "__main__":
    image_path = "uploads/70ace7279_test2_wall_shot.jpeg"
    process_image_for_border(image_path)
