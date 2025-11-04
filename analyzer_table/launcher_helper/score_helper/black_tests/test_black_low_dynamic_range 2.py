import cv2
import numpy as np
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import get_ball_image, clamp_0_100

def _circular_mask(shape, shrink=0.85):
    h, w = shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.5 * shrink)
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m

def run(ball: Ball) -> float:
    """
    ⚫ Internal Dynamic Range:
    מודד p95(V)-p5(V); קטן → שחור חלק. גדול → כנראה לא שחור.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = _circular_mask(img.shape, 0.85)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[...,2].astype(np.float32)
    vals = v[mask > 0]
    if vals.size == 0:
        return 0.0

    p5  = float(np.percentile(vals, 5))
    p95 = float(np.percentile(vals, 95))
    dyn = max(0.0, p95 - p5)

    # מיפוי: dyn<=20 → 100, dyn>=80 → 0  (כוונן לפי הדאטה)
    lo, hi = 20.0, 80.0
    t = 1.0 - np.clip((dyn - lo) / (hi - lo), 0.0, 1.0)
    score = clamp_0_100(t * 100.0)
    return score
