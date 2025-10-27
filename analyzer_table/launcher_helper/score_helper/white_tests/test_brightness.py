import numpy as np
import cv2
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import get_ball_image, clamp_0_100

def run(ball: Ball) -> float:
    """
    מחשב ציון לפי הבהירות הממוצעת של הכדור (white_score_test_1)
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2].astype(np.float32)  # בהירות
    mean_brightness = float(np.mean(v_channel))  # ממוצע בהירות 0–255
    score = (mean_brightness / 255.0) * 100.0
    return clamp_0_100(score)
