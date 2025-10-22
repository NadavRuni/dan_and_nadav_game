from analyzer_table.detect_ball.analyzer_runner import run_full_analysis
from analyzer_table.detect_ball.merge_utils import mergeData
from analyzer_table.detect_ball.draw_utils import draw_balls_on_image
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.launcher_helper.black_and_white_launcher import run_ball_detection
from dataclasses import dataclass
from typing import Tuple, List, Optional
import cv2
import numpy as np
import os


@dataclass
class Ball:
    center: Tuple[int, int]
    radius: int


def analyze_ball_brightness(image_path: str, balls: List[Ball], output_dir: str = "out/balls") -> Tuple[Optional[Ball], Optional[Ball]]:
    """
    🎱 מזהה את הכדור הכי לבן והכי שחור בתמונה.
    שומר את כל הכדורים כתמונות נפרדות בתיקייה נתונה.
    בנוסף שומר את הכדור הלבן כ-white.png ואת השחור כ-black.png
    """
    Debugger.log(f"🧠 Analyzing {len(balls)} balls to find the whitest and darkest...")

    img = cv2.imread(image_path)
    if img is None:
        Debugger.error(f"❌ Failed to load image from {image_path}")
        return None, None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    os.makedirs(output_dir, exist_ok=True)

    whitest_ball, darkest_ball = None, None
    max_brightness, min_brightness = -1, 9999
    whitest_img, darkest_img = None, None
    Debugger.log(f"🗂️ Saving individual ball images to: {output_dir}")

    for i, ball in enumerate(balls, 1):
        cx, cy = map(int, ball.center)
        r = int(ball.radius * 1.3)

        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        v_channel = roi[:, :, 2]
        mean_brightness = np.mean(v_channel)

        ball_img = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_dir, f"ball_{i}.png"), ball_img)

        if mean_brightness > max_brightness:
            max_brightness = mean_brightness
            whitest_ball = ball
            whitest_img = ball_img.copy()
        if mean_brightness < min_brightness:
            min_brightness = mean_brightness
            darkest_ball = ball
            darkest_img = ball_img.copy()
    Debugger.log(f"⚪ Whitest ball brightness: {max_brightness:.2f}, ⚫ Darkest ball brightness: {min_brightness:.2f}")

    if whitest_img is not None:
        cv2.imwrite(os.path.join(output_dir, "white.png"), whitest_img)
        Debugger.log(f"✅ Saved whitest ball image to {os.path.join(output_dir, 'white.png')}")
    if darkest_img is not None:
        cv2.imwrite(os.path.join(output_dir, "black.png"), darkest_img)
        Debugger.log(f"✅ Saved darkest ball image to {os.path.join(output_dir, 'black.png')}")
    
    print("whitest_ball" , whitest_ball)
    print("darkest_ball" , darkest_ball)

    return whitest_ball, darkest_ball

from typing import List, Tuple, Optional

def full_analyzer_pipeline(image_path: str) -> Tuple[List[Ball], Optional[Ball], Optional[Ball]]:
    """
    🧩 פונקציה מרכזית שמריצה את כל תהליך הזיהוי והמיזוג.
    קלט:  נתיב לתמונה אחת.
    פלט:  (רשימת כל הכדורים, הכדור השחור, הכדור הלבן)
    """
    Debugger.log(f"🚀 Starting full analyzer pipeline for: {image_path}")

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    # שלב 1: ניתוח מלא
    sub_photos, main_photo = run_full_analysis(image_path)
    black_and_white_ball_list = run_ball_detection(image_path)
    table_rectangle = find_table_rectangle(image_path)

    if not sub_photos or not main_photo:
        Debugger.error("❌ Analysis failed or no data returned.")
        return [], None, None

    # שלב 2: מיזוג
    total_before_merge = len(main_photo.balls) + sum(len(p.balls) for p in sub_photos) + len(black_and_white_ball_list)
    merged_photo = mergeData(main_photo, sub_photos, black_and_white_ball_list, table_rectangle)
    total_after_merge = len(merged_photo.balls)
    Debugger.log(f"✅ Merged {total_before_merge} → {total_after_merge} balls")

    # שלב 3: ציור סופי
    output_final_path = os.path.join(out_dir, "final_detected.png")
    draw_balls_on_image(merged_photo, image_path, output_final_path, table_rectangle)
    Debugger.log(f"🖼️ Final image saved to {output_final_path}")

    # שלב 4: מיון וסיכום
    sorted_balls = sorted(merged_photo.balls, key=lambda b: b.center[0])
    Debugger.log(f"📦 Total unique balls: {len(sorted_balls)}")

    # שלב 5: זיהוי הכדור הלבן והשחור
    white_ball ,black_ball =analyze_ball_brightness(image_path, sorted_balls, os.path.join(out_dir, "balls"))


    if white_ball:
        Debugger.log(f"⚪ White ball found at {white_ball.center}")
    else:
        Debugger.warn("⚪ White ball not found")

    if black_ball:
        Debugger.log(f"⚫ Black ball found at {black_ball.center}")
    else:
        Debugger.warn("⚫ Black ball not found")

    return {
    "balls": sorted_balls,
    "black": {
        "x": float(black_ball.center[0]),
        "y": float(black_ball.center[1]),
    } if black_ball else None,
    "white": {
        "x": float(white_ball.center[0]),
        "y": float(white_ball.center[1]),
    } if white_ball else None,
    "table_box": {
        "top_left": table_rectangle.top_left,
        "top_right": table_rectangle.top_right,
        "bottom_left": table_rectangle.bottom_left,
        "bottom_right": table_rectangle.bottom_right,
    },
}



# ✅ דוגמה לשימוש:
if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), "input", "first.jpeg")
    result = full_analyzer_pipeline(input_path)
    print(f"Returned {len(result)} balls → {[(b.center, b.radius) for b in result[:5]]}")
