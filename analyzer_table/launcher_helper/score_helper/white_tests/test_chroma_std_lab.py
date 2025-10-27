# analyzer_table/launcher_helper/score_helper/white_tests/test_chroma_std_lab.py
import cv2, numpy as np
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
    ⚪ White - Chroma Uniformity (Lab a*, b* std)
    - ככל שהסטייה של r=√(a^2+b^2) קטנה יותר → הציון גבוה יותר.
    - שונה מהמבחן 'neutrality' (שמודד מרחק ממוצע), כאן מודדים *אחידות*.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    mask = _circular_mask(img.shape, 0.85)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[...,1].astype(np.float32) - 128.0
    b = lab[...,2].astype(np.float32) - 128.0
    r = np.sqrt(a*a + b*b)
    r_roi = r[mask > 0]
    if r_roi.size == 0:
        return 0.0

    std_r = float(np.std(r_roi))
    # מיפוי: std=0 → 100 ; std=20 → 0 (כוונן לפי הדאטה שלך)
    score = (1.0 - (std_r / 20.0)) * 100.0
    return clamp_0_100(score)
