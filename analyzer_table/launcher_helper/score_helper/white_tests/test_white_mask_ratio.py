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
    W3 (חדש): בודק את אחוז הפיקסלים שהם א-כרומטיים (רוויה נמוכה).
    מתעלם מצללים כהים (V < 50) ומתמקד בכמה מהכדור הוא "חסר צבע".
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    ball_mask = get_circle_mask(img)
    if np.count_nonzero(ball_mask) == 0:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[..., 1]
    v_channel = hsv[..., 2]

    # ניקח רק פיקסלים שהם על הכדור ואינם בצל עמוק
    pixels_on_ball_mask = ball_mask == 255
    pixels_not_dark_mask = v_channel > 50  # התעלם מצללים

    relevant_pixels_mask = pixels_on_ball_mask & pixels_not_dark_mask
    total_relevant_pixels = np.count_nonzero(relevant_pixels_mask)

    if total_relevant_pixels == 0:
        return 0.0  # הכדור כולו שחור

    # מתוך הפיקסלים הרלוונטיים, נספור כמה הם "חסרי צבע"
    SATURATION_THRESHOLD = 40  # סף רוויה מקסימלי (סובלני יותר מ-30)

    low_sat_mask = s_channel < SATURATION_THRESHOLD
    good_pixels_mask = relevant_pixels_mask & low_sat_mask

    good_pixel_count = np.count_nonzero(good_pixels_mask)

    score = (good_pixel_count / total_relevant_pixels) * 100.0
    return clamp_0_100(score)
