import numpy as np
import cv2
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import (
    get_ball_image, 
    clamp_0_100, 
    get_circle_mask
)

def run(ball: Ball) -> float:
    """
    W_NEW: בודק נייטרליות על ציר כחול-צהוב (b-channel) במרחב Lab.
    כדור לבן אמיתי צריך להיות קרוב ל-128 (נייטרלי).
    כדור צהוב (אפילו בהיר) יהיה עם ערך b גבוה יותר.
    מעניק ציון גבוה ככל שהערך קרוב יותר ל-128.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0
    
    mask = get_circle_mask(img,ball.center, ball.radius)
    if np.count_nonzero(mask) == 0:
        return 0.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    b_channel = lab[..., 2] # ערוץ b
    
    # חשב ממוצע רק בתוך המסכה
    b_pixels = b_channel[mask == 255].astype(np.float32)
    mean_b_value = float(np.mean(b_pixels))
    
    # הערך הנייטרלי ב-8bit Lab הוא 128
    neutral_point = 128.0
    
    # חשב מרחק מהנייטרליות
    distance_from_neutral = abs(mean_b_value - neutral_point)
    
    # נרמול: נניח שסטייה של 20 היא כבר "לא לבן" (ציון 0)
    # סטייה של 2 (ערך 130) תיתן ציון 90
    # סטייה של 10 (ערך 138 - צהבהב) תיתן ציון 50
    score = (1.0 - (distance_from_neutral / 20.0)) * 100.0
    
    return clamp_0_100(score)