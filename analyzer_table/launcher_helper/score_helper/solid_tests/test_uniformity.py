# analyzer_table/score_helper/solid_tests/test_uniformity.py
import cv2, numpy as np
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import (
    get_ball_image,
    clamp_0_100,
)


def run(ball: Ball) -> float:
    img = get_ball_image(ball)
    if img is None:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    std_sv = float(np.mean([np.std(s), np.std(v)]))  # 0..~80
    # נמוך=טוב → ממפה הופכית: std 0 => 100, std 64 => ~0
    return clamp_0_100(100.0 * (1.0 - (std_sv / 64.0)))
