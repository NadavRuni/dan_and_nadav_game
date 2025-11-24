import cv2
from analyzer_table.black_white_detect.detect_balls_and_pockets import (
    detect_pockets_pipeline,
)
from output_utils import get_output_path  # Needed for MASK_OUTPUT_PATH access

# Assuming this is the input image for testing
original_image_path = "output/debug/cropped_63a578eaa_test3_b2b.jpeg"
original_image = cv2.imread(original_image_path)

if original_image is None:
    print(f"❌ Could not read input image: {original_image_path}")
else:
    # Call the pipeline to generate the mask and detect pockets
    detect_pockets_pipeline(original_image)

    # Manually reconstruct the path as detect_pockets_pipeline doesn't return it
    print(
        f"Mask image saved to: {get_output_path('01_pocket_mask.jpg', sub_dir='black_white_detect')}"
    )
