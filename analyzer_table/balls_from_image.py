from output_utils import get_output_path
import json
from pathlib import Path
from analyzer_table.detect_ball.analyzer_runner import run_full_analysis
from analyzer_table.detect_ball.merge_utils import mergeData
from analyzer_table.detect_ball.draw_utils import draw_balls_on_image
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.launcher_helper.black_and_white_launcher import run_ball_detection
from typing import Tuple, List, Optional
import cv2
from analyzer_table.ball_type_score.build_suites import (
    build_white_suite,
    build_black_suite,
)
from analyzer_table.ball_type_score.run_scores import score_balls
from analyzer_table.launcher_helper.score_helper.common import (
    _white_avg,
    _black_avg,
    assert_scored,
)
import numpy as np
import os
from analyzer_table.launcher_helper.json_models import (
    Ball,
    table_pockets,
    AnalyzerResult,
)
from analyzer_table.ball_from_image_helper import crop_and_save_balls
from analyzer_table.launcher_helper.pocket.pocket_detect import analyze_table_pockets
from analyzer_table.launcher_helper.pocket.pocket_cycle import mark_pocket_circles
from analyzer_table.predict.models.predict import update_undefined_balls
from analyzer_table.table.table import confirm_or_correct_rectangle
from const_numbers import BASE_DIR, RECTANGLE_JSON_PATH
from analyzer_table.launcher_helper.json_models import Rectangle
from analyzer_table.table.rectangle import parse_rectangle_from_data

from analyzer_table.launcher_helper.score_helper.white_tests import test_is_white_model
from analyzer_table.launcher_helper.json_models import Ball_Color


def insert_black_and_white_balls(balls: List[Ball], black_ball: Ball, white_ball: Ball):
    for ball in balls:
        if ball.center == black_ball.center:
            ball.final_color = "black"
        elif ball.center == white_ball.center:
            ball.final_color = "white"


def analyze_ball_brightness(
    image_path: str, balls: List[Ball], output_dir: str = "out/balls"
) -> Tuple[Optional[Ball], Optional[Ball]]:
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
        cv2.imwrite(
            get_output_path(f"ball_{i}.png", sub_dir="balls_from_image"), ball_img
        )
        if mean_brightness > max_brightness:
            max_brightness = mean_brightness
            whitest_ball = ball
            whitest_img = ball_img.copy()
        if mean_brightness < min_brightness:
            min_brightness = mean_brightness
            darkest_ball = ball
            darkest_img = ball_img.copy()
    Debugger.log(
        f"⚪ Whitest ball brightness: {max_brightness:.2f}, ⚫ Darkest ball brightness: {min_brightness:.2f}"
    )
    if whitest_img is not None:
        cv2.imwrite(
            get_output_path("white.png", sub_dir="balls_from_image"), whitest_img
        )
        Debugger.log(
            f"✅ Saved whitest ball image to {get_output_path('white.png', sub_dir='balls_from_image')}"
        )
    if darkest_img is not None:
        cv2.imwrite(
            get_output_path("black.png", sub_dir="balls_from_image"), darkest_img
        )
        Debugger.log(
            f"✅ Saved darkest ball image to {get_output_path('black.png', sub_dir='balls_from_image')}"
        )
    print("whitest_ball", whitest_ball)
    print("darkest_ball", darkest_ball)
    return whitest_ball, darkest_ball


from typing import List, Tuple, Optional


def full_analyzer_pipeline(image_path: str) -> AnalyzerResult:
    Debugger.log(f"🚀 Starting full analyzer pipeline for: {image_path}")
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    rect_path = Path(BASE_DIR / RECTANGLE_JSON_PATH)
    if rect_path.exists():
        try:
            with open(rect_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[DEBUG] Loaded existing data: {data}")
            table_rectangle = parse_rectangle_from_data(data)
        except Exception as e:
            print(f"[DEBUG] Failed to read existing file ({e}), creating empty one.")
            data = {}
            table_rectangle = find_table_rectangle(image_path)
    else:
        print("[DEBUG] File not found — creating new rectangle.")
        data = {}
        table_rectangle = find_table_rectangle(image_path)
    sub_photos, main_photo = run_full_analysis(image_path)
    black_and_white_ball_list = run_ball_detection(image_path)
    if table_rectangle is None:
        Debugger.error("❌ Table rectangle not confirmed or selected.")
        return []
    all_pocket: table_pockets = analyze_table_pockets(image_path, table_rectangle)
    Debugger.log(f"path : {all_pocket.pockets_img_paths}")
    if not sub_photos or not main_photo:
        Debugger.error("❌ Analysis failed or no data returned.")
        return [], None, None
    total_before_merge = (
        len(main_photo.balls)
        + sum(len(p.balls) for p in sub_photos)
        + len(black_and_white_ball_list)
    )
    merged_photo = mergeData(
        main_photo, sub_photos, black_and_white_ball_list, table_rectangle
    )
    total_after_merge = len(merged_photo.balls)
    Debugger.log(f"✅ Merged {total_before_merge} → {total_after_merge} balls")
    output_final_path = os.path.join(out_dir, "final_detected.png")
    draw_balls_on_image(
        merged_photo,
        image_path,
        output_final_path,
        table_rectangle,
        all_pockets=all_pocket,
    )
    Debugger.log(f"🖼️ Final image saved to {output_final_path}")
    sorted_balls = sorted(merged_photo.balls, key=lambda b: b.center[0])
    Debugger.log(f"📦 Total unique balls: {len(sorted_balls)}")
    crop_and_save_balls(image_path, sorted_balls)
    Debugger.log(f"✂️ Cropped and saved individual ball images.")
    for i, ball in enumerate(sorted_balls, 1):
        Debugger.log(
            f"   - Ball #{i}: Center={ball.center}, Radius={ball.radius}, Color={ball.final_color} "
        )
        Debugger.log(f"path to ball image: {ball.single_ball_path}")
    white_suite = build_white_suite()
    black_suite = build_black_suite()
    score_balls(sorted_balls, white_suite, black_suite)
    assert_scored(sorted_balls)
######################################################################################################
   # whitest_ball = max(sorted_balls, key=_white_avg, default=None)
    candidates = []
    for ball in sorted_balls:
        legacy_score = _white_avg(ball)
        candidates.append((ball, legacy_score))
    
    # מיון מהגבוה לנמוך לפי הציון הישן
    candidates.sort(key=lambda x: x[1], reverse=True)

    whitest_ball = None
    
    if candidates:
        # המקום הראשון כרגע (לפי הלוגיקה הישנה)
        top_ball, top_score = candidates[0]
        
        # 2. איסוף כל ה"מתמודדים" שנמצאים בהפרש קטן מ-10 מהפסגה
        contenders = []
        for ball, score in candidates:
            diff = top_score - score
            if diff < 10:
                contenders.append(ball)
            else:
                # בגלל שהרשימה ממויינת, ברגע שעברנו את ה-10 אין טעם להמשיך
                break
        
        # 3. ההכרעה
        if len(contenders) == 1:
            # אין שוויון - המנצח הישן לוקח
            whitest_ball = top_ball
            Debugger.log(f"✅ Clear winner (Legacy): Score {top_score:.1f}")
            
        else:
            # יש שוויון בין כמה כדורים! המודל נכנס לפעולה
            Debugger.log(f"⚔️ Multi-Tie detected! {len(contenders)} candidates within 10 points range.")
            
            best_ml_ball = None
            best_ml_score = -1.0
            
            for ball in contenders:
                path = ball.single_ball_path
                current_ml_score = 0.0
                
                if path and os.path.exists(path):
                    # בדיקת המודל
                    current_ml_score = test_is_white_model.get_white_score(cv2.imread(path))
                
                Debugger.log(f"   - Candidate at {ball.center}: Legacy={_white_avg(ball):.1f} | ML Conf={current_ml_score:.4f}")
                
                if current_ml_score > best_ml_score:
                    best_ml_score = current_ml_score
                    best_ml_ball = ball
            
            Debugger.log(f"   🏆 ML Chose ball at {best_ml_ball.center} with conf {best_ml_score:.4f}")
            whitest_ball = best_ml_ball
    # ======================================================
######################################################################################################
    print("Whitest ball analysis:")

    for ball in sorted_balls:
        score = _white_avg(ball)
        print(f"{ball.center}: white score = {score}")
    blackest_ball = max(sorted_balls, key=_black_avg, default=None)
    if whitest_ball and os.path.exists(whitest_ball.single_ball_path):
        whitest_img = cv2.imread(whitest_ball.single_ball_path, cv2.IMREAD_COLOR)
        if whitest_img is not None:
            cv2.imwrite(get_output_path("white.png", sub_dir="balls"), whitest_img)
            Debugger.log(
                f"✅ Saved whitest ball image to {get_output_path('white.png', sub_dir='balls')}"
            )
    if blackest_ball and os.path.exists(blackest_ball.single_ball_path):
        blackest_img = cv2.imread(blackest_ball.single_ball_path, cv2.IMREAD_COLOR)
        if blackest_img is not None:
            cv2.imwrite(get_output_path("black.png", sub_dir="balls"), blackest_img)
            Debugger.log(
                f"✅ Saved blackest ball image to {get_output_path('black.png', sub_dir='balls')}"
            )
    insert_black_and_white_balls(sorted_balls, blackest_ball, whitest_ball)
    update_undefined_balls(sorted_balls)
    if whitest_ball:
        Debugger.log(f"⚪ White ball found at {whitest_ball.center}")
    else:
        Debugger.warn("⚪ White ball not found")
    if blackest_ball:
        Debugger.log(f"⚫ Black ball found at {blackest_ball.center}")
    else:
        Debugger.warn("⚫ Black ball not found")
    analyzerResult = AnalyzerResult(
        Pockets=all_pocket,
        balls=sorted_balls,
        black=blackest_ball,
        white=whitest_ball,
    )
    Debugger.log("✅ Full analyzer pipeline completed successfully.")
    return analyzerResult


if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), "input", "first.jpeg")
    result = full_analyzer_pipeline(input_path)
    print(
        f"Returned {len(result)} balls → {[(b.center, b.radius) for b in result[:5]]}"
    )
