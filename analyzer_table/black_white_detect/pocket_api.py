"""
High-level API for the pocket detection pipeline.

This module provides a single entry point to run the full pocket detection
process, which includes creating a table mask, processing the border, and
identifying pocket locations.
"""

from analyzer_table.black_white_detect.detect_balls_and_pockets import (
    create_felt_mask,
    find_pockets_from_mask,
)
from analyzer_table.border_and_pocket.main import process_image_for_border
from analyzer_table.launcher_helper.json_models import Rectangle
from const_numbers import set_detected_pockets


def pocket_detection_api(
    original_image_path: str,
) -> tuple[str, str, Rectangle]:
    """
    Runs the full pipeline to detect pocket locations from an image.

    This pipeline involves several major steps:
    1.  Processing the image to identify and crop to the table border.
    2.  Creating a binary mask of the felt to isolate balls and pockets.
    3.  Finding pocket contours from the mask.
    4.  Storing the detected pockets in the application's global state.

    Args:
        original_image_path: The path to the source image of the pool table.

    Returns:
        A tuple containing:
        - Path to the debug image showing pockets on the original image.
        - Path to the cropped image of the table.
        - The Rectangle object defining the cropped area.
    """
    # Note: The initial call to create_felt_mask seems redundant as its
    # results are not used. The mask is recreated inside process_image_for_border.
    # This suggests a potential for simplification in the pipeline.

    # This function call has the primary side effect of producing the cropped
    # image and the final mask needed for pocket detection.
    (
        final_mask_path,
        final_binary_mask,
        cropped_original_image,
        table_rectangle,
        cropped_photo_path,
    ) = process_image_for_border(original_image_path)

    print("Finding corner pockets from mask...")
    (
        detected_pockets,
        debug_mask_path,
        original_with_pockets_path,
    ) = find_pockets_from_mask(final_mask_path, table_rectangle, cropped_original_image)

    print("Pocket detection process completed.")
    # This reliance on global state is a significant architectural issue.
    set_detected_pockets(detected_pockets)

    print(f"Detected Pockets: {len(detected_pockets)}")
    print(f"Debug Mask Path: {debug_mask_path}")
    print(f"Original with Pockets Path: {original_with_pockets_path}")
    print(f"Cropped Photo Path: {cropped_photo_path}")

    return original_with_pockets_path, cropped_photo_path, table_rectangle
