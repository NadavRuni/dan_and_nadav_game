from detect_ball.analyzer_runner import run_full_analysis
from detect_ball.merge_utils import mergeData
from detect_ball.draw_utils import draw_balls_on_image
from detect_ball.Debugger import Debugger
from detect_ball.detect_table import find_table_rectangle
from dataclasses import dataclass
from typing import Tuple, List, Optional
import cv2
import numpy as np
import os
import sys
from launcher_helper.black_and_white_launcher import run_ball_detection

# הוסף את הנתיב הראשי למודולים
sys.path.append(os.path.dirname(__file__))

GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


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

    for i, ball in enumerate(balls, 1):
        cx, cy = map(int, ball.center)
        r = int(ball.radius * 1.3)

        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)

        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            Debugger.warn(f"⚠️ Ball #{i} ROI empty, skipping.")
            continue

        v_channel = roi[:, :, 2]
        mean_brightness = np.mean(v_channel)

        Debugger.log(f"⚪ Ball #{i}: center=({cx},{cy}), r={r}, brightness={mean_brightness:.2f}")

        # שמירת תמונה של כל כדור
        ball_img = img[y1:y2, x1:x2]
        ball_path = os.path.join(output_dir, f"ball_{i}_({cx}_{cy})_r{r}.png")
        cv2.imwrite(ball_path, ball_img)

        if mean_brightness > max_brightness:
            max_brightness = mean_brightness
            whitest_ball = ball
            whitest_img = ball_img.copy()

        if mean_brightness < min_brightness:
            min_brightness = mean_brightness
            darkest_ball = ball
            darkest_img = ball_img.copy()

    # שמור את הכדור הכי לבן והשחור
    if whitest_img is not None:
        white_path = os.path.join(output_dir, "white.png")
        cv2.imwrite(white_path, whitest_img)
        Debugger.log(f"🏆 Whitest ball saved as: {white_path}")

    if darkest_img is not None:
        black_path = os.path.join(output_dir, "black.png")
        cv2.imwrite(black_path, darkest_img)
        Debugger.log(f"⚫ Darkest ball saved as: {black_path}")

    Debugger.log(f"🗂️ All ball images saved under: {output_dir}")

    return whitest_ball, darkest_ball


if __name__ == "__main__":
    Debugger.log("🚀 Starting full analyzer process")

    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "input")
    path_name = "first.jpeg"
    full_path = os.path.join(input_dir, path_name)

    # === שלב 1: ניתוח מלא ===
    sub_photos, main_photo = run_full_analysis(full_path)
    black_and_white_ball_list = run_ball_detection(full_path)
    table_rectangle = find_table_rectangle(full_path)

    if not sub_photos or not main_photo:
        Debugger.error("❌ Analysis failed or no data returned.")
    else:
        Debugger.warn("🏁 Analysis complete — merging data...")

        # === שלב 2: מיזוג ===
        total_before_merge = len(main_photo.balls) + sum(len(p.balls) for p in sub_photos) + len(black_and_white_ball_list)
        merged_photo = mergeData(main_photo, sub_photos,black_and_white_ball_list,  table_rectangle)
        total_after_merge = len(merged_photo.balls)
        merged_count = total_before_merge - total_after_merge

        # === שלב 3: ציור סופי ===
        output_final_path = os.path.join(base_dir, "out", "final_detected.png")
        os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
        draw_balls_on_image(merged_photo, full_path, output_final_path, table_rectangle)

        # === שלב 4: סיכום כדורים ===
        sorted_balls = sorted(merged_photo.balls, key=lambda b: b.center[0])

        print(f"\n{BOLD}{CYAN}========== ⚪ FINAL SUMMARY (sorted by X) =========={RESET}")
        print(f"Total detected balls (before merge): {total_before_merge}")
        print(f"Balls merged (duplicates removed):   {merged_count}")
        print(f"Unique balls remaining:              {total_after_merge}\n")

        for i, b in enumerate(sorted_balls, 1):
            print(f"⚪ Ball #{i}: center=({b.center[0]}, {b.center[1]}), radius={b.radius}")

        print(f"\n🖼️ Final image saved to: {output_final_path}")
        print(
            f"📐 Table rectangle: TL={table_rectangle.top_left}, "
            f"TR={table_rectangle.top_right}, BL={table_rectangle.bottom_left}, BR={table_rectangle.bottom_right}"
        )

        # === שלב 5: ניתוח בהירות/כהות ===
        whitest, darkest = analyze_ball_brightness(full_path, sorted_balls, os.path.join(base_dir, "out", "balls"))

        print(f"\n✨ Whitest ball:  center={whitest.center if whitest else None}, radius={whitest.radius if whitest else None}")
        print(f"⚫ Darkest ball:  center={darkest.center if darkest else None}, radius={darkest.radius if darkest else None}")

    print(f"\n{GREEN}{BOLD}✅ Done! Merged data successfully.{RESET}\n")
