"""
Crops a binary mask to its content area by analyzing projection profiles.

This module provides a more advanced alternative to simple border removal by
calculating horizontal and vertical projection profiles to find the main
content area of a binary mask.
"""

from typing import Tuple, Optional, Dict

import cv2
import numpy as np


def find_continuous_segment_from_profile(
    projection: np.ndarray, threshold_ratio: float = 0.5
) -> Tuple[int, int]:
    """
    Finds the start and end of the largest continuous segment above a threshold.

    Args:
        projection: A 1D numpy array representing a pixel projection profile.
        threshold_ratio: A ratio of the max projection value to determine the
                         threshold for what is considered 'content'.

    Returns:
        A tuple containing the start and end indices of the segment.
    """
    if np.max(projection) == 0:
        return 0, 0

    threshold = np.max(projection) * threshold_ratio
    above_threshold = projection > threshold

    if not np.any(above_threshold):
        return 0, 0

    # np.argmax returns the index of the first True value.
    start_index = np.argmax(above_threshold)
    # Flipping the array and using argmax finds the first True from the end.
    end_index = (len(above_threshold) - 1) - np.argmax(np.flip(above_threshold))

    return start_index, end_index


def crop_to_content_by_projection(
    binary_mask: np.ndarray, profile_threshold: float = 0.5, edge_margin: int = 5
) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """
    Detects table boundaries using pixel intensity projection profiles and crops.

    This function inverts the mask to make the table area white, calculates
    horizontal and vertical projection profiles, and finds the largest continuous
    segment in each profile to determine the bounding box of the table.

    Args:
        binary_mask: A 2D numpy array where the table is black (0).
        profile_threshold: Ratio of the max projection value to consider a
                           row/col as part of the table segment.
        edge_margin: Margin to check if the detected table touches the image
                     edges. If it does, no crop is performed.

    Returns:
        A tuple containing:
        - The cropped mask.
        - A dictionary with the crop coordinates, or None if no crop was made.
    """
    img_height, img_width = binary_mask.shape

    # Invert mask so table is white (255) for projection calculation
    table_mask = cv2.bitwise_not(binary_mask)

    # Calculate horizontal and vertical projection profiles
    horizontal_projection = np.sum(table_mask, axis=1)
    vertical_projection = np.sum(table_mask, axis=0)

    # Find the largest continuous segment in each profile
    top, bottom = find_continuous_segment_from_profile(
        horizontal_projection, profile_threshold
    )
    left, right = find_continuous_segment_from_profile(
        vertical_projection, profile_threshold
    )

    # Validation: If the detected segment is too small or touches the edges,
    # assume no border needs to be cropped.
    is_too_small = (bottom - top) < 10 or (right - left) < 10
    touches_edge = (
        top <= edge_margin
        or left <= edge_margin
        or bottom >= img_height - edge_margin
        or right >= img_width - edge_margin
    )

    if is_too_small or touches_edge:
        return binary_mask, None

    crop_coords = {"top": top, "bottom": bottom, "left": left, "right": right}
    cropped_mask = binary_mask[top : bottom + 1, left : right + 1]

    return cropped_mask, crop_coords
