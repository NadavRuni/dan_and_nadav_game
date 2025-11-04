import numpy as np
import cv2
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import (
    get_ball_image, clamp_0_100, get_circle_mask
)

def _safe_pct(arr, q, default=0.0):
    return float(np.percentile(arr, q)) if arr.size else float(default)

def _inv_norm(x, lo, hi):
    """קטן=טוב ⇒ 100..0"""
    if hi <= lo: 
        return 0.0
    x = max(lo, min(hi, float(x)))
    return 100.0 * (1.0 - (x - lo) / (hi - lo))

def run(ball: Ball) -> float:
    """
    White detection (v7):
    - חלון צבע טהור (L*, a*, b*, S).
    - אין יותר התייחסות למרקם (std_L) שהיה פגום.
    - חלון רוויה (S) רחב יותר.
    - חלון b* (צהוב) נשאר.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r  = int(0.45 * min(h, w))
    if r <= 2:
        return 0.0

    # מסכות
    inner = get_circle_mask(img, center=(cx, cy), radius=int(r * 0.72))
    if inner is None:
        return 0.0

    # מרחבים
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    # דגימות
    inner_mask = (inner > 0)
    if not np.any(inner_mask):
        return 0.0

    inner_L  = L[inner_mask]
    inner_S  = S[inner_mask]
    inner_a  = a[inner_mask]
    inner_b  = b[inner_mask]

    # --- ערכי מדידה ---
    L_p80 = _safe_pct(inner_L, 80, 0.0)
    S_med = float(np.median(inner_S)) if inner_S.size else 255.0
    a_med = float(np.median(inner_a)) if inner_a.size else 0.0
    b_med = float(np.median(inner_b)) if inner_b.size else 0.0

    # --- הגדרת "חלונות" צבע ---
    PEAK_B = 45.0  # (w1 היה 47)
    PEAK_S = 100.0 # (w1=118, w3=89) - מרכזנו מחדש ונתן טווח רחב

    # 1) בהירות (L*):
    bright_score = clamp_0_100((L_p80 / 255.0) * 100.0)

    # 2) רוויה (S) - חלון סביב השיא
    # #############################################################
    # # שינוי: חלון רחב יותר סביב 100 כדי לכלול את w3 (S=89)
    # #############################################################
    dist_S = abs(S_med - PEAK_S)
    sat_score = _inv_norm(dist_S, lo=20.0, hi=90.0) # סובלנות של 20, נפילה עד 90

    # 3) ציון צבע (a*, b*)
    a_score = _inv_norm(abs(a_med), lo=2.0, hi=20.0) # נשאר
    dist_B = abs(b_med - PEAK_B)
    b_score = _inv_norm(dist_B, lo=5.0, hi=30.0) # נשאר
    neutral_score = 0.55 * a_score + 0.45 * b_score

    # 4) מרקם (std L*) - בוטל
    
    # ענישות - בוטלו
    glare_penalty = 0.0
    yellow_penalty = 0.0

    # #############################################################
    # # שינוי: משקלים (רק צבע ובהירות)
    # #############################################################
    w_neutral = 0.50  # (a* + b*)
    w_sat     = 0.40  # (S)
    w_bright  = 0.10  # (L*)
    
    score = (
        w_neutral * neutral_score +
        w_sat     * sat_score     +
        w_bright  * bright_score
    )



    return clamp_0_100(score)