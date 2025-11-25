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
    ⚫ Trimmed Darkness:
    ממוצע ערוץ V אחרי חיתוך 5% תחתון ו-20% עליון (מסיר היילייטים),
    ואז ננרמל הפוך: כהה יותר → ציון גבוה יותר.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = _circular_mask(img.shape, 0.85)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    vals = v[mask > 0]
    if vals.size < 20:
        return 0.0

    lo = np.percentile(vals, 5)
    hi = np.percentile(vals, 80)  # חותכים את ה-20% הבהירים ביותר
    clipped = vals[(vals >= lo) & (vals <= hi)]
    if clipped.size == 0:
        return 0.0

    mean_v = float(np.mean(clipped))

    # מיפוי חד יותר: mean_v<=55 → ~100 ; mean_v>=110 → ~0
    if mean_v <= 55:
        score = 100.0
    elif mean_v >= 110:
        score = 0.0
    else:
        # מיפוי ליניארי הפוך בתחום
        t = (mean_v - 55.0) / (110.0 - 55.0)
        score = (1.0 - t) * 100.0

    return clamp_0_100(float(score))
