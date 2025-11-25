import cv2
import numpy as np


def remove_white_border(binary_mask: np.ndarray) -> np.ndarray:
    """
    Removes a surrounding white border from a binary image mask.
    If a connected white border is found on all four edges of the image, the function
    crops the image to the bounding box of the inner black pixels. Otherwise, it
    returns the original image.
    Args:
        binary_mask: A 2D numpy array representing the binary image,
                     where 255 is white and 0 is black. The input is expected
                     to be the result of a morphological closing operation.
    Returns:
        A 2D numpy array of the cropped mask if a white border is detected and removed,
        otherwise returns the original mask.
    """
    height, width = binary_mask.shape

    # A border is considered present if the outermost pixel layer is entirely white.
    is_top_border = np.all(binary_mask[0, :] == 255)
    is_bottom_border = np.all(binary_mask[height - 1, :] == 255)
    is_left_border = np.all(binary_mask[:, 0] == 255)
    is_right_border = np.all(binary_mask[:, width - 1] == 255)

    if not (is_top_border and is_bottom_border and is_left_border and is_right_border):
        # If any edge is not purely white, it's not a continuous rectangular
        # border as described. Return the original mask as is.
        return binary_mask

    # Find the bounding box of all non-white (i.e., black) pixels.
    # This effectively finds the content area inside the white border.
    black_pixels = np.where(binary_mask == 0)

    if black_pixels[0].size == 0:
        # The image is entirely white. There is no content to crop to.
        return binary_mask

    # Determine the extents of the black pixel area.
    top = np.min(black_pixels[0])
    bottom = np.max(black_pixels[0])
    left = np.min(black_pixels[1])
    right = np.max(black_pixels[1])

    # Crop the image to the determined bounding box.
    # The slice is inclusive of the start and exclusive of the end, so +1 is needed.
    cropped_mask = binary_mask[top : bottom + 1, left : right + 1]

    return cropped_mask
