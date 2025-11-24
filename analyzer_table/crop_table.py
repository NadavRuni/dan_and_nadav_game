import cv2
import numpy as np
import unittest
from typing import Tuple, Optional, Dict

def find_continuous_segment_from_profile(projection: np.ndarray, threshold_ratio: float = 0.5) -> Tuple[int, int]:
    """
    Finds the start and end indices of the largest continuous segment of values
    above a certain threshold in a 1D projection profile.
    """
    if np.max(projection) == 0:
        return 0, 0
        
    threshold = np.max(projection) * threshold_ratio
    above_threshold = projection > threshold
    
    if not np.any(above_threshold):
        return 0, 0

    # Find the indices of the first and last 'True' values.
    # np.argmax returns the index of the first True.
    start_index = np.argmax(above_threshold)
    # Flipping the array and using argmax finds the first True from the end.
    end_index = (len(above_threshold) - 1) - np.argmax(np.flip(above_threshold))
    
    return start_index, end_index


def remove_white_border(binary_mask: np.ndarray,
                        profile_threshold: float = 0.5,
                        edge_margin: int = 5) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """
    Detects table boundaries using pixel intensity projection profiles.
    Args:
        binary_mask: A 2D numpy array where the table is black (0).
        profile_threshold: Ratio of the max projection value to consider a row/col as part of the table segment.
        edge_margin: Margin to check if the detected table touches the image edges. If it does, no crop is performed.
    Returns:
        A tuple of (cropped_mask, crop_coordinates_dict | None).
    """
    img_h, img_w = binary_mask.shape

    # Invert mask so table is white (255) for projection calculation
    table_mask = cv2.bitwise_not(binary_mask)

    # Calculate horizontal and vertical projection profiles
    h_proj = np.sum(table_mask, axis=1)
    v_proj = np.sum(table_mask, axis=0)

    # Find the largest continuous segment in each profile
    top, bottom = find_continuous_segment_from_profile(h_proj, profile_threshold)
    left, right = find_continuous_segment_from_profile(v_proj, profile_threshold)
    
    # Validation: If the detected segment is too small or touches the edges, assume no border to crop
    if (top <= edge_margin or left <= edge_margin or
        bottom >= img_h - edge_margin or right >= img_w - edge_margin or
        (bottom - top) < 10 or (right - left) < 10): # Avoid tiny crops
        return binary_mask, None

    crop_coords = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    cropped_mask = binary_mask[top:bottom + 1, left:right + 1]

    return cropped_mask, crop_coords


class TestRemoveWhiteBorder(unittest.TestCase):

    def test_projection_crop_with_border(self):
        """A white border around a black table should be cropped."""
        # 70x80 image, all white
        img = np.full((70, 80), 255, dtype=np.uint8)
        # Add a 50x60 black table inside, with 10px margin
        img[10:60, 10:70] = 0 # top, bottom | left, right
        
        processed_mask, coords = remove_white_border(img)

        self.assertIsNotNone(coords)
        self.assertEqual(processed_mask.shape, (50, 60))
        self.assertEqual(coords['top'], 10)
        self.assertEqual(coords['left'], 10)
        self.assertEqual(coords['bottom'], 59)
        self.assertEqual(coords['right'], 69)

    def test_projection_no_border(self):
        """A black table touching the image edge should not be cropped."""
        img = np.full((70, 80), 255, dtype=np.uint8)
        # Black table touches top and left edge
        img[0:50, 0:60] = 0
        
        processed_mask, coords = remove_white_border(img, edge_margin=1)
        
        self.assertIsNone(coords)
        np.testing.assert_array_equal(processed_mask, img)

    def test_projection_with_internal_noise(self):
        """Noise inside the table should not affect the projection crop."""
        img = np.full((100, 100), 255, dtype=np.uint8)
        # Add a black table with a 10px margin
        img[10:90, 10:90] = 0
        # Add a white dot (noise) inside the black table
        img[50, 50] = 255
        
        processed_mask, coords = remove_white_border(img)
        
        self.assertIsNotNone(coords)
        # The projection profile should be robust enough to ignore the single pixel drop.
        self.assertEqual(coords['top'], 10)
        self.assertEqual(coords['bottom'], 89)
        self.assertEqual(coords['left'], 10)
        self.assertEqual(coords['right'], 89)

    def test_all_black_image(self):
        """An all-black image should not be cropped."""
        img = np.zeros((100, 100), dtype=np.uint8)
        processed_mask, coords = remove_white_border(img)
        self.assertIsNone(coords)
        np.testing.assert_array_equal(processed_mask, img)


if __name__ == '__main__':
    unittest.main()