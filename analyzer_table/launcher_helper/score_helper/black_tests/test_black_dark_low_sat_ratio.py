import cv2
import numpy as np
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import (
    get_ball_image,
    clamp_0_100,
)


def _circular_mask(shape, shrink=0.85):
    h, w = shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.5 * shrink)
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def run(ball: Ball) -> float:
    """
    ⚫ Dark+Low-Sat Ratio:
    יחס פיקסלים שגם V נמוך *וגם* S נמוך. שחור אמיתי לרוב יעמוד בזה;
    כדורים צבעוניים – לרוב לא (S גבוה), ולבן – V גבוה.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = _circular_mask(img.shape, 0.85)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.uint8)
    v = hsv[..., 2].astype(np.uint8)

    # ספי "כהה" ו"רוויה נמוכה" (כוונון קל לפי הדאטה שלך)
    V_MAX = 95
    S_MAX = 70

    roi = mask > 0
    if not np.any(roi):
        return 0.0

    good = (v <= V_MAX) & (s <= S_MAX) & roi
    ratio = float(np.count_nonzero(good)) / float(np.count_nonzero(roi))

    # מיפוי: 0.15 → 0 ; 0.55 → 100  (מגדיל פערים לשחורים טובים)
    lo, hi = 0.15, 0.55
    if ratio <= lo:
        score = 0.0
    elif ratio >= hi:
        score = 100.0
    else:
        t = (ratio - lo) / (hi - lo)
        score = t * 100.0

    return clamp_0_100(float(score))
