"""
Common utilities for ball scoring and image manipulation.

This module provides a collection of helper functions used across the different
scoring tests, including color space conversion, image masking, normalization,
and score calculation.
"""

import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from analyzer_table.launcher_helper.json_models import Ball


def clamp_0_100(value: float) -> float:
    """
    Clamps a numerical value to the inclusive range [0, 100].

    Args:
        value: The input number.

    Returns:
        The value clamped to the range 0 to 100.
    """
    return float(max(0.0, min(100.0, value)))


def norm_0_100(value: float, min_val: float, max_val: float) -> float:
    """
    Normalizes a value from a given range to the range [0, 100].

    Args:
        value: The value to normalize.
        min_val: The minimum of the input range.
        max_val: The maximum of the input range.

    Returns:
        The normalized value on a scale of 0 to 100. Returns 0 if max_val
        equals min_val to avoid division by zero.
    """
    if max_val == min_val:
        return 0.0
    return clamp_0_100(100.0 * (value - min_val) / (max_val - min_val))


def to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Converts an image from BGR to HSV color space.

    Args:
        image: The input image in BGR format.

    Returns:
        The converted image in HSV format.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def get_circle_mask(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None,
    padding: int = 0,
) -> Optional[np.ndarray]:
    """
    Creates a binary circle mask for a given image.

    If center and radius are not provided, it creates a circle in the center
    of the image for backward compatibility. This is not recommended.

    Args:
        image: The image for which to create the mask.
        center: The (x, y) coordinates of the circle's center.
        radius: The radius of the circle.
        padding: Additional padding to add to the radius.

    Returns:
        A binary mask (0s and 255s) of the same size as the input image,
        with the circle area filled. Returns None if the input image is None.
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if center is None or radius is None:
        # Backward compatibility for old calls: creates a circle in the center.
        center_x, center_y = w // 2, h // 2
        # A "safe" radius of ~45% of the smallest dimension.
        mask_radius = int(0.45 * min(h, w))
    else:
        center_x, center_y = int(center[0]), int(center[1])
        mask_radius = int(max(0, radius))

    mask_radius = int(mask_radius + max(0, padding))
    # Ensure center coordinates are within image bounds
    center_x = max(0, min(w - 1, center_x))
    center_y = max(0, min(h - 1, center_y))

    if mask_radius > 0:
        cv2.circle(mask, (center_x, center_y), int(mask_radius), 255, thickness=-1)

    return mask


def get_ball_circle_mask(
    image: np.ndarray, ball: Ball, padding: int = 0
) -> Optional[np.ndarray]:
    """
    Creates a circle mask based on a Ball object's center and radius.

    This is the recommended way to create a mask for a ball in scoring tests.

    Args:
        image: The image for which to create the mask.
        ball: The Ball object providing the center and radius.
        padding: Additional padding to add to the radius.

    Returns:
        A binary mask representing the ball's area.
    """
    return get_circle_mask(
        image, center=ball.center, radius=ball.radius, padding=padding
    )


def get_annulus_mask(
    shape: Tuple[int, int], center: Tuple[int, int], r_inner: float, r_outer: float
) -> np.ndarray:
    """
    Creates a binary mask of a ring (annulus).

    Args:
        shape: The (height, width) of the mask to create.
        center: The (x, y) center of the ring.
        r_inner: The inner radius of the ring.
        r_outer: The outer radius of the ring.

    Returns:
        A binary mask with the ring shape.
    """
    dummy_image = np.zeros(shape, dtype=np.uint8)
    outer_circle = get_circle_mask(dummy_image, center, r_outer)
    inner_circle = get_circle_mask(dummy_image, center, r_inner)
    # Subtract inner from outer to get the ring
    return cv2.bitwise_and(outer_circle, cv2.bitwise_not(inner_circle))


def get_ball_image(ball: Ball) -> Optional[np.ndarray]:
    """
    Loads the image for a ball and caches it as a dynamic '_cached_img' attribute.

    Subsequent calls for the same ball object will return the cached image
    instead of reading from the disk again.

    Args:
        ball: The ball object with a 'single_ball_path'.

    Returns:
        The loaded image as a numpy array, or None if the path is invalid.
    """
    # This use of dynamic attributes (monkey-patching) is not recommended.
    cached_image = getattr(ball, "_cached_img", None)
    if cached_image is not None:
        return cached_image

    path = ball.single_ball_path
    if not path or not os.path.exists(path):
        setattr(ball, "_cached_img", None)
        return None

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    setattr(ball, "_cached_img", image)
    return image


def clear_ball_image(ball: Ball) -> None:
    """
    Removes the cached image from a ball object to free up memory.

    Args:
        ball: The ball object to clear the cache from.
    """
    if hasattr(ball, "_cached_img"):
        delattr(ball, "_cached_img")


def _calculate_white_score_average(ball: Ball) -> float:
    """
    Calculates the weighted average score for a ball being the white ball.

    Args:
        ball: A ball object that has been scored.

    Returns:
        The weighted average score. Returns 0.0 if scoring data is missing.
    """
    if not hasattr(ball, "color_score") or not ball.color_score:
        return 0.0
    w_scores = ball.color_score.white_score
    if not w_scores:
        return 0.0

    # These weights should ideally be stored in a configuration file.
    weights = {
        "W1": 0.65,
        "W2": 0.05,
        "W3": 0.15,
        "W4": 0.015,
        "W5": 0.05,
    }

    weighted_sum = (
        float(w_scores.white_score_test_1) * weights["W1"]
        + float(w_scores.white_score_test_2) * weights["W2"]
        + float(w_scores.white_score_test_3) * weights["W3"]
        + float(w_scores.white_score_test_4) * weights["W4"]
        + float(w_scores.white_score_test_5) * weights["W5"]
    )
    return weighted_sum


def _calculate_black_score_average(ball: Ball) -> float:
    """
    Calculates the average score for a ball being the black ball.

    Args:
        ball: A ball object that has been scored.

    Returns:
        The average score. Returns 0.0 if scoring data is missing.
    """
    if not hasattr(ball, "color_score") or not ball.color_score:
        return 0.0
    b_scores = ball.color_score.black_score
    if not b_scores:
        return 0.0

    b_vec = [
        float(b_scores.black_score_test_1),
        float(b_scores.black_score_test_2),
        float(b_scores.black_score_test_3),
        float(b_scores.black_score_test_4),
        float(b_scores.black_score_test_5),
    ]
    return sum(b_vec) / len(b_vec) if b_vec else 0.0


def assert_scored(balls: List[Ball]) -> None:
    """
    Asserts that all balls in a list have been properly scored.

    This is a debugging utility to ensure that the scoring pipeline has run
    correctly before proceeding with color classification.

    Args:
        balls: A list of Ball objects.

    Raises:
        AssertionError: If any ball is missing its color score attributes.
    """

    for i, ball in enumerate(balls, 1):
        cs = getattr(ball, "color_score", None)
        assert cs is not None, (
            f"[Ball {i}] is missing 'color_score' attribute. "
            f"Did you run score_balls()?"
        )

        white_score = getattr(cs, "white_score", None)
        black_score = getattr(cs, "black_score", None)
        assert white_score is not None, f"[Ball {i}] is missing 'white_score'"
        assert black_score is not None, f"[Ball {i}] is missing 'black_score'"

        for test_num in range(1, 6):
            attr_name = f"white_score_test_{test_num}"
            assert hasattr(
                white_score, attr_name
            ), f"[Ball {i}] is missing '{attr_name}'"

        for test_num in range(1, 6):
            attr_name = f"black_score_test_{test_num}"
            assert hasattr(
                black_score, attr_name
            ), f"[Ball {i}] is missing '{attr_name}'"
