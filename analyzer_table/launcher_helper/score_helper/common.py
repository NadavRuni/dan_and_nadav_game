# analyzer_table/score_helper/common.py
import os
import cv2 
from analyzer_table.launcher_helper.json_models import Ball
from typing import List
import numpy as np





__all__ = [
    "clamp_0_100",
    "norm_0_100",
    "to_hsv",
    "get_circle_mask",
    "get_annulus_mask",
    "_white_avg",
    "_black_avg",
    "get_ball_image",
    "clear_ball_image",
    "assert_scored",
]

# analyzer_table/launcher_helper/score_helper/common.py
import os
import cv2
import numpy as np
from analyzer_table.launcher_helper.json_models import Ball
from typing import List, Tuple

# ... כל הקוד הקיים שלך ...

import numpy as np
import cv2
from typing import Tuple, Optional
# ... שאר הייבואים והקוד ...

def get_circle_mask(
    img,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None,
    padding: int = 0,
):
    """
    מחזיר מסכת מעגל (0/255) בגודל img.
    תאימות לאחור: אם center/radius לא נמסרו, ישתמש במרכז התמונה וברדיוס יחסי.
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if center is None or radius is None:
        # תאימות לאחור לקריאות ישנות: מעגל במרכז התמונה
        cx, cy = w // 2, h // 2
        # רדיוס "בטוח": ~45% מהמימד הקטן
        r = int(0.45 * min(h, w))
    else:
        cx, cy = int(center[0]), int(center[1])
        r = int(max(0, radius))

    r = int(r + max(0, padding))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    if r > 0:
        cv2.circle(mask, (cx, cy), int(r), 255, thickness=-1)

    return mask


def get_ball_circle_mask(img, ball, padding: int = 0):
    """
    מסכת מעגל לפי center/radius של Ball.
    שימוש מומלץ בבדיקות.
    """
    return get_circle_mask(img, center=ball.center, radius=ball.radius, padding=padding)



def get_annulus_mask(shape, center, r_inner, r_outer):
    """טבעת בין רדיוס פנימי לחיצוני כ־mask של 0/1."""
    outer = get_circle_mask(shape, center, r_outer, filled=True)
    inner = get_circle_mask(shape, center, r_inner, filled=True)
    return (outer & (1 - inner)).astype(np.uint8)


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
        'W1': 0.65,
        'W2': 0.05,
        'W3': 0.15,
        'W4': 0.015,
        'W5': 0.05,
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



def assert_scored(balls: List[Ball]) -> None:
    for i, b in enumerate(balls, 1):
        cs = getattr(b, "color_score", None)
        assert cs is not None, f"[Ball {i}] missing color_score (did you run score_balls?)"

        w = getattr(cs, "white_score", None)
        bl = getattr(cs, "black_score", None)
        assert w is not None, f"[Ball {i}] missing white_score"
        assert bl is not None, f"[Ball {i}] missing black_score"

        for attr in ["white_score_test_1","white_score_test_2","white_score_test_3","white_score_test_4","white_score_test_5"]:
            assert hasattr(w, attr), f"[Ball {i}] missing {attr}"
        for attr in ["black_score_test_1","black_score_test_2","black_score_test_3","black_score_test_4","black_score_test_5"]:
            assert hasattr(bl, attr), f"[Ball {i}] missing {attr}"
