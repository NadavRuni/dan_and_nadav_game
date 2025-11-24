import cv2, numpy as np
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


def run(ball):
    img = get_ball_image(ball)
    if img is None:
        return 0.0
    mask = _circular_mask(img.shape, 0.85)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    a = a[mask > 0].astype(np.float32) - 128.0  # מרכז ל-0
    b = b[mask > 0].astype(np.float32) - 128.0
    if a.size == 0:
        return 0.0

    # מרחק מהניטרליות (0,0). ננרמל בקירוב.
    dist = np.sqrt(a * a + b * b).mean()  # ככל שקטן → יותר ניטרלי
    # 0 → 100 ; 40 → 0 (קאליברציה גסה)
    neutral = max(0.0, 1.0 - (dist / 40.0)) * 100.0

    # משקול לפי בהירות (V), כדי למנוע אפור כהה
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2][mask > 0].astype(np.float32)
    mean_v = float(v.mean()) if v.size else 0.0
    v_weight = min(1.0, mean_v / 160.0)  # מעל 160 כמעט לא מגביל

    mean_s = float(
        hsv[..., 1][mask > 0].astype(np.float32).mean()
        if np.count_nonzero(mask)
        else 0.0
    )
    low_sat_weight = 1.0 - min(1.0, mean_s / 140.0)  # יותר נמוך S → משקל גבוה
    score = neutral * v_weight * (0.7 + 0.3 * low_sat_weight)
    return clamp_0_100(score)
