from analyzer_table.border_and_pocket.main import process_image_for_border
from analyzer_table.black_white_detect.detect_balls_and_pockets import (
    crate_mask_table,
    find_corner_pockets_from_mask,
)
from analyzer_table.launcher_helper.json_models import Rectangle
from const_numbers import set_detected_pockets


def pocket_detection_api(original_image_path: str) -> tuple[str, str, Rectangle]:

    # First, get the initial mask and original image to pass to find_corner_pockets_from_mask
    _, initial_binary_mask, initial_original_image = crate_mask_table(
        original_image_path
    )

    # Call process_image_for_border for its side effects (saving debug images).
    (
        final_mask_output_path,
        binary_mask,
        output_debug_original_cropped,
        rect,
        crop_photo_path,
    ) = process_image_for_border(original_image_path)

    print("Finding corner pockets from mask...")
    (
        detected_pockets,
        debug_mask_path,
        original_with_pockets_path,
    ) = find_corner_pockets_from_mask(
        final_mask_output_path, binary_mask, output_debug_original_cropped, rect
    )

    print("Test completed.")
    print("Detected Pockets:", detected_pockets)
    set_detected_pockets(detected_pockets)
    print("Debug Mask Path:", debug_mask_path)
    print("Original with Pockets Path:", original_with_pockets_path)
    print("Cropped Photo Path:", crop_photo_path)
    return original_with_pockets_path, crop_photo_path, rect
