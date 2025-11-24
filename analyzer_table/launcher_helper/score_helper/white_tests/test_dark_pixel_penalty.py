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
    W5 (חדש): מעניש כדורים שיש להם פיקסלים כהים (כמו פסים או מספרים).
    משתמש בערוץ L (בהירות) מ-Lab.
    מתעלם מהשתקפויות (שהן פיקסלים בהירים).
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = get_circle_mask(img)
    if np.count_nonzero(mask) == 0:
        return 0.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_channel = lab[..., 0]  # ערוץ L (בהירות 0-255)

    l_pixels = l_channel[mask == 255]
    if l_pixels.size == 0:
        return 0.0  # אין פיקסלים

    # סף בהירות: כל פיקסל מתחת ל-180 (מ-255) נחשב "כהה"
    # (סף של ~70% בהירות)
    BRIGHTNESS_THRESHOLD = 180

    dark_pixel_count = np.sum(l_pixels < BRIGHTNESS_THRESHOLD)
    total_pixel_count = l_pixels.size

    # חשב את אחוז הפיקסלים הכהים
    dark_ratio = dark_pixel_count / total_pixel_count

    # הציון הוא הפוך: ככל שיש פחות פיקסלים כהים, הציון גבוה יותר
    score = (1.0 - dark_ratio) * 100.0

    return clamp_0_100(score)
