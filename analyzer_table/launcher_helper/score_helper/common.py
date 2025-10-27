# analyzer_table/score_helper/common.py
import os
import cv2
from analyzer_table.launcher_helper.json_models import Ball

def clamp_0_100(x: float) -> float:
    """מגביל ציון לטווח 0–100."""
    return float(max(0.0, min(100.0, x)))

def norm_0_100(val: float, min_v: float, max_v: float) -> float:
    """נרמול ערך מכל טווח לטווח 0–100."""
    if max_v == min_v:
        return 0.0
    return clamp_0_100(100.0 * (val - min_v) / (max_v - min_v))

def to_hsv(img):
    """המרה ל-HSV לעיבודי צבע נוחים יותר."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



def get_ball_image(ball: Ball):
    """
    טוען ומקַאש את התמונה בתוך ball._cached_img (אטריביוט דינמי).
    קריאה חוזרת לא תיגש לדיסק שוב.
    """
    img = getattr(ball, "_cached_img", None)
    if img is not None:
        return img

    path = ball.single_ball_path
    if not path or not os.path.exists(path):
        ball._cached_img = None
        return None

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    ball._cached_img = img
    return img

def clear_ball_image(ball: Ball):
    """ניקוי הקאש אם רוצים לשחרר זיכרון אחרי העיבוד."""
    if hasattr(ball, "_cached_img"):
        delattr(ball, "_cached_img")


def _white_avg(ball: Ball) -> float:
    """ מחשב את הציון המשוקלל של הכדור הלבן. """
    if not hasattr(ball, 'color_score') or not ball.color_score:
         return 0.0 # אין נתוני ניקוד
    w = ball.color_score.white_score
    if not w:
        return 0.0

    w_vec = [
        float(w.white_score_test_1), # W1
        float(w.white_score_test_2), # W2
        float(w.white_score_test_3), # W3
        float(w.white_score_test_4), # W4
        float(w.white_score_test_5), # W5
    ]

    # --- השתמש באותם משקלים שהגדרת ב-run_real_image.py ---
    weights = {
        'W1': 0.25,
        'W2': 0.20,
        'W3': 0.05,
        'W4': 0.35,
        'W5': 0.15,
    }

    weighted_sum = (
        w_vec[0] * weights['W1'] +
        w_vec[1] * weights['W2'] +
        w_vec[2] * weights['W3'] +
        w_vec[3] * weights['W4'] +
        w_vec[4] * weights['W5']
    )
    return weighted_sum

def _black_avg(ball: Ball) -> float:
    """ מחשב את הציון הממוצע (או משוקלל אם תרצה) של הכדור השחור. """
    if not hasattr(ball, 'color_score') or not ball.color_score:
         return 0.0
    b = ball.color_score.black_score
    if not b:
        return 0.0

    b_vec = [
        float(b.black_score_test_1),
        float(b.black_score_test_2),
        float(b.black_score_test_3),
        float(b.black_score_test_4),
        float(b.black_score_test_5),
    ]
    # כרגע מחזיר ממוצע פשוט, שנה למשוקלל אם הגדרת משקלים גם לשחור
    return sum(b_vec) / 5.0 if b_vec else 0.0