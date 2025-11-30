"""
Pocket Location Definition and Image Extraction.

This module provides a function to define the six standard pocket locations
based on the geometry of the table rectangle. It then crops and saves an image
of each pocket area for further analysis or debugging.
"""

import os
from typing import List

import cv2

from analyzer_table.launcher_helper.json_models import Rectangle
from const_numbers import (
    get_crop_half_size,
    get_use_predicted_pockets,
    get_detected_pockets,
)
from game_class.C_pocket import Pocket
from output_utils import get_output_path


def extract_pocket_images_from_rectangle(
    image_path: str,
    rectangle: Rectangle,
    output_dir: str = "out/pockets",
    crop_half_size: int = get_crop_half_size(),
) -> List[Pocket]:
    """
    Defines 6 pocket locations from a table rectangle and saves their images.

    If the global 'use_predicted_pockets' flag is set, this function will
    return a pre-existing list of pockets. Otherwise, it calculates the six
    standard pocket locations (corners and middles) from the provided
    rectangle, crops a square region of interest (ROI) around each, saves the
    ROI as an image file, and returns a list of Pocket objects.

    Args:
        image_path: The path to the full table image.
        rectangle: A Rectangle object defining the table boundaries.
        output_dir: The directory where pocket images will be saved.
        crop_half_size: The half-width/height of the square to crop around
                          each pocket center.

    Returns:
        A list of six Pocket objects with their locations and image paths.

    Raises:
        ValueError: If the image file cannot be loaded.
    """
    # This check relies on global state and is not a recommended pattern.
    if get_use_predicted_pockets():
        print("✅ Using previously detected pockets.")
        pockets = get_detected_pockets()
        return pockets if pockets is not None else []

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Failed to load image from {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    image_height, image_width = image.shape[:2]

    # Calculate the center points of the middle pockets
    top_middle_point = (
        int((rectangle.top_left[0] + rectangle.top_right[0]) / 2),
        int((rectangle.top_left[1] + rectangle.top_right[1]) / 2),
    )
    bottom_middle_point = (
        int((rectangle.bottom_left[0] + rectangle.bottom_right[0]) / 2),
        int((rectangle.bottom_left[1] + rectangle.bottom_right[1]) / 2),
    )

    pocket_positions = {
        "TL": rectangle.top_left,
        "TM": top_middle_point,
        "TR": rectangle.top_right,
        "BL": rectangle.bottom_left,
        "BM": bottom_middle_point,
        "BR": rectangle.bottom_right,
    }

    pocket_list: List[Pocket] = []
    for i, (name, center) in enumerate(pocket_positions.items()):
        center_x, center_y = center
        x1 = max(0, center_x - crop_half_size)
        y1 = max(0, center_y - crop_half_size)
        x2 = min(image_width, center_x + crop_half_size)
        y2 = min(image_height, center_y + crop_half_size)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        region_of_interest = image[y1:y2, x1:x2]
        if region_of_interest.size == 0:
            print(f"⚠️ Skipping pocket '{name}' – ROI is empty (is it near an edge?)")
            continue

        filename = f"{name}.png"
        output_path = get_output_path(filename, sub_dir="pockets")
        cv2.imwrite(output_path, region_of_interest)

        pocket = Pocket(
            id=i,
            center=(center_x, center_y),
            radius=crop_half_size,
            pocket_img_path=output_path,
            pocket_img_cordinates_on_table=(center_x, center_y),
            location=name,
        )
        pocket_list.append(pocket)

    print(f"✅ Created and saved images for {len(pocket_list)} pockets.")
    return pocket_list
