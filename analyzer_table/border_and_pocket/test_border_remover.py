import unittest
import numpy as np
import cv2  # OpenCV will be used for creating test data

# Assuming the function is in a file named 'border_remover.py' inside 'analyzer_table'
from analyzer_table.border_and_pocket.border_remover import remove_white_border


class TestRemoveWhiteBorder(unittest.TestCase):

    def test_with_white_border(self):
        """
        Test case for an image that has a clear white border around a black rectangle.
        The border should be cropped.
        """
        # Create a 100x100 black image
        inner_area = np.zeros((100, 100), dtype=np.uint8)
        # Add some white "pockets/balls" inside
        inner_area[20:30, 40:50] = 255
        inner_area[70:80, 60:70] = 255

        # Add a 10-pixel white border around it
        bordered_image = cv2.copyMakeBorder(
            inner_area, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
        )

        # Apply the function
        processed_mask = remove_white_border(bordered_image)

        # The result should be the original 100x100 inner area
        self.assertEqual(processed_mask.shape, (100, 100))
        # Verify that the content of the processed mask is identical to the inner area
        np.testing.assert_array_equal(processed_mask, inner_area)

    def test_without_white_border(self):
        """
        Test case for an image where black pixels touch the edges.
        The image should be returned unchanged.
        """
        # Create a 100x100 image that is mostly black but touches the edges
        no_border_image = np.zeros((100, 100), dtype=np.uint8)
        # Add some white shapes
        no_border_image[20:80, 20:80] = 255
        # Make sure top-left corner is black, breaking the "all-white border" condition
        no_border_image[0, 0] = 0

        # Apply the function
        processed_mask = remove_white_border(no_border_image)

        # The shape and content should be identical to the input
        self.assertEqual(processed_mask.shape, no_border_image.shape)
        np.testing.assert_array_equal(processed_mask, no_border_image)

    def test_all_white_image(self):
        """
        Test case for an image that is entirely white.
        It has a "border" but no content to crop to. It should be returned as is.
        """
        all_white_image = np.full((100, 100), 255, dtype=np.uint8)

        # Apply the function
        processed_mask = remove_white_border(all_white_image)

        # The shape and content should be identical to the input
        self.assertEqual(processed_mask.shape, all_white_image.shape)
        np.testing.assert_array_equal(processed_mask, all_white_image)

    def test_content_touching_one_edge_from_inside(self):
        """
        Test case where the border is present, but an internal black shape
        extends to touch the inside edge of the border.
        """
        # Create a 100x100 black image, with one part touching the top edge
        inner_area = np.zeros((100, 100), dtype=np.uint8)
        inner_area[0, 50] = 0  # Black pixel at the top edge of the inner area

        # Add a 10-pixel white border
        bordered_image = cv2.copyMakeBorder(
            inner_area, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
        )

        # Apply the function
        processed_mask = remove_white_border(bordered_image)

        # The result should be cropped correctly to the 100x100 inner area
        self.assertEqual(processed_mask.shape, (100, 100))
        np.testing.assert_array_equal(processed_mask, inner_area)


if __name__ == "__main__":
    unittest.main()
