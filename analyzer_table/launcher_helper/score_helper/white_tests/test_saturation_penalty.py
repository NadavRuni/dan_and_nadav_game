import numpy as np
import cv2
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import (
    get_ball_image,
    clamp_0_100,
    get_circle_mask,
)


def run(ball: Ball) -> float:
    """
    W5 (מתוקן): מעניש כדורים לפי אחוז הפיקסלים הצבעוניים.
    גרסה לא אגרסיבית (בוטל מקדם הענישה).
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = get_circle_mask(img)
    total_pixels = np.count_nonzero(mask)
    if total_pixels == 0:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[..., 1]
    v_channel = hsv[..., 2]

    s_pixels = s_channel[mask == 255]
    v_pixels = v_channel[mask == 255]

    SATURATION_THRESHOLD = 60
    VALUE_THRESHOLD = 50

    colorful_mask = (s_pixels > SATURATION_THRESHOLD) & (v_pixels > VALUE_THRESHOLD)
    colorful_pixel_count = np.sum(colorful_mask)

    color_ratio = colorful_pixel_count / total_pixels

    # --- התיקון ---
    # הציון הוא פשוט האחוז ההפוך של הפיקסלים הצבעוניים
    # בוטל penalty_factor = 10.0
    score = (1.0 - color_ratio) * 100.0

    return clamp_0_100(score)
