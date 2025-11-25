import sys
from pathlib import Path

# --- Setup sys.path ---
# Add project root to allow sibling imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from analyzer_table.border_and_pocket.main import process_image_for_border
from analyzer_table.black_white_detect.detect_balls_and_pockets import (
    crate_mask_table,
    detect_only_pockets_and_draw,
    find_corner_pockets_from_mask,
)

if __name__ == "__main__":
    original_image_path = "uploads/everything_white_like.jpeg"

    # First, get the initial mask and original image to pass to find_corner_pockets_from_mask
    _, initial_binary_mask, initial_original_image = crate_mask_table(
        original_image_path
    )

    # Call process_image_for_border for its side effects (saving debug images).
    final_mask_output_path, binary_mask, output_debug_original_cropped, rect = (
        process_image_for_border(original_image_path)
    )

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
    print("Debug Mask Path:", debug_mask_path)
    print("Original with Pockets Path:", original_with_pockets_path)
