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
    W5: מחשב ציון ע"י זיהוי קצוות פנימיים (נגד פסים/מספרים).
    גרסה קפדנית (חוזרים למקור)
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    
    inner_radius = int(radius * 0.85) 
    if inner_radius <= 0:
        return 100.0 

    inner_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(inner_mask, center, inner_radius, 255, -1)
    
    total_mask_pixels = np.count_nonzero(inner_mask)
    if total_mask_pixels == 0:
        return 100.0 

    # --- שינוי: חוזרים לספים הרגישים ---
    # Canny(100, 200) היה סלחני מדי
    edges = cv2.Canny(gray, 50, 150)
    
    internal_edges_map = cv2.bitwise_and(edges, edges, mask=inner_mask)
    internal_edge_pixels = np.count_nonzero(internal_edges_map)
    
    edge_density = internal_edge_pixels / total_mask_pixels
    
    # --- שינוי: חוזרים לענישה אגרסיבית ---
    # penalty_factor = 3.0 היה סלחני מדי
    penalty_factor = 5.0 
    score = (1.0 - (edge_density * penalty_factor)) * 100.0
    
    return clamp_0_100(score)