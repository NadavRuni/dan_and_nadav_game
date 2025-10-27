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
    טסט נייטרליות על ציר כחול-צהוב (b-channel) וגם ירוק-אדום (a-channel).
    כדור לבן אמיתי צריך להיות קרוב ל-128 (נייטרלי) בשני הערוצים.
    כדור צבעוני (כמו הירוק של 15) יקבל ציון נמוך.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0
    
    mask = get_circle_mask(img)
    if np.count_nonzero(mask) == 0:
        return 0.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab[..., 1] # ערוץ a (ירוק-אדום)
    b_channel = lab[..., 2] # ערוץ b (כחול-צהוב)
    
    # חשב ממוצע רק בתוך המסכה
    a_pixels = a_channel[mask == 255].astype(np.float32)
    b_pixels = b_channel[mask == 255].astype(np.float32)
    mean_a_value = float(np.mean(a_pixels))
    mean_b_value = float(np.mean(b_pixels))
    
    neutral_point = 128.0
    
    # חשב מרחק משולב מהנייטרליות
    distance_a = abs(mean_a_value - neutral_point)
    distance_b = abs(mean_b_value - neutral_point)
    total_distance = np.sqrt(distance_a**2 + distance_b**2) # מרחק אוקלידי
    
    # נרמול: סטייה משולבת של 25 היא כבר "לא לבן" (ציון 0)
    score = (1.0 - (total_distance / 25.0)) * 100.0
    
    return clamp_0_100(score)