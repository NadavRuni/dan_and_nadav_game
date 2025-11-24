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
    ⚫ Low-Percentile Darkness:
    מודד את פרקטיל 20% של V; נמוך=שחור. עמיד להיילייטים מקומיים.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = _circular_mask(img.shape, 0.85)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    vals = v[mask > 0]
    if vals.size == 0:
        return 0.0

    p10 = float(np.percentile(vals, 10))

    # מיפוי חד יותר: p10<=45 → ~100 ; p10>=95 → ~0, עם "ברך" באמצע
    if p10 <= 45:
        score = 100.0
    elif p10 >= 95:
        score = 0.0
    else:
        # עקומה לוגיסטית עדינה להגדלת רזולוציה באזור 60–80
        k = 0.12
        x0 = 70.0
        t = 1.0 / (1.0 + np.exp(k * (p10 - x0)))
        score = t * 100.0

    return clamp_0_100(float(score))
