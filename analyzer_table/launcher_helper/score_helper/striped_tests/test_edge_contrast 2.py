# analyzer_table/launcher_helper/score_helper/striped_tests/test_edge_contrast.py
import cv2, numpy as np
from analyzer_table.launcher_helper.json_models import Ball
from analyzer_table.launcher_helper.score_helper.common import get_ball_image, clamp_0_100


def run(ball: Ball) -> float:
    """
    מדד STRIPED חכם:
    מזהה דפוסי פסים באמצעות פילטרי גבור בכיוונים שונים ומודד 'בנדיות כיוונית' (anisotropy).
    כולל ROI פנימי, נרמול תאורה, ענישת בהירות־יתר, וקונטרסט יחסי.
    מחזיר ציון 0–100.
    """
    img = get_ball_image(ball)
    if img is None:
        return 0.0

    # --- שלב 1: ROI פנימי כדי להוציא את ההיקף והרקע ---
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.45)  # רדיוס פנימי
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(img, img, mask=mask)

    # --- שלב 2: נרמול תאורה + מעבר לגווני אפור ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- שלב 3: פילטרי גבור בכמה זוויות ---
    ksize = max(9, (min(h, w) // 8) | 1)
    sigmas = [ksize / 3.5, ksize / 2.5]
    lambdas = [max(4, min(h, w) // 6)]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    responses = []
    for th in thetas:
        for sig in sigmas:
            for lam in lambdas:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma=sig, theta=th,
                    lambd=lam, gamma=0.5, psi=0, ktype=cv2.CV_32F
                )
                resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
                responses.append(np.mean(np.abs(resp[mask > 0])))

    responses = np.array(responses, dtype=np.float32)
    if responses.size == 0:
        return 0.0

    # --- שלב 4: מדד אניזוטרופיה (כיווניות) ---
    # אם יש כיוון דומיננטי אחד, זה סימן לפסים
    max_r = float(np.max(responses))
    med_r = float(np.median(responses))
    anisotropy = max(0.0, (max_r - med_r) / (max_r + 1e-6))

    # --- שלב 5: קונטרסט יחסי ---
    roi_vals = gray[mask > 0].astype(np.float32)
    if roi_vals.size == 0:
        return 0.0
    rel_contrast = np.clip(np.std(roi_vals) / (np.mean(roi_vals) + 1e-6), 0, 1)

    # --- שלב 6: ענישת בהירות־יתר (כדור לבן חלק) ---
    mean_brightness = np.mean(roi_vals)
    bright_penalty = max(0.0, (mean_brightness - 200) / 55.0)  # 0–1

    # --- שלב 7: חישוב סופי ---
    score = (anisotropy * (0.6 + 0.4 * rel_contrast)) * (1.0 - 0.4 * bright_penalty) * 100.0
    return clamp_0_100(score)
