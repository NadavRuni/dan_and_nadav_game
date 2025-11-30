"""
Main orchestrator for the full image analysis pipeline.

This module contains the primary function, `full_analyzer_pipeline`, which
coordinates all steps of the pool table analysis process, from detecting the
table and balls to classifying ball colors and packaging the final result.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from analyzer_table.ball_from_image_helper import crop_and_save_balls
from analyzer_table.ball_type_score.build_suites import (
    build_white_suite,
    build_black_suite,
)
from analyzer_table.ball_type_score.run_scores import score_balls
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.detect_ball.analyzer_runner import run_full_analysis
from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.detect_ball.draw_utils import draw_balls_on_image
from analyzer_table.detect_ball.merge_utils import mergeData
from analyzer_table.launcher_helper.black_and_white_launcher import run_ball_detection
from analyzer_table.launcher_helper.json_models import (
    Ball,
    AnalyzerResult,
    Rectangle,
)
from analyzer_table.launcher_helper.pocket.pocket_cycle import mark_pocket_circles
from analyzer_table.launcher_helper.pocket.pocket_detect import (
    extract_pocket_images_from_rectangle,
)
from analyzer_table.launcher_helper.score_helper.common import (
    _calculate_white_score_average,
    _calculate_black_score_average,
    assert_scored,
)
from analyzer_table.launcher_helper.score_helper.white_tests import (
    test_is_white_model,
)
from analyzer_table.predict.models.predict import update_undefined_balls
from analyzer_table.table.rectangle import parse_rectangle_from_data
from const_numbers import BASE_DIR, RECTANGLE_JSON_PATH
from game_class.C_pocket import Pocket
from output_utils import get_output_path


def _insert_black_and_white_balls(
    balls: List[Ball], black_ball: Optional[Ball], white_ball: Optional[Ball]
):
    """
    Sets the final_color for the white and black balls in the main list.

    Args:
        balls: The list of all detected balls.
        black_ball: The ball identified as the black ball.
        white_ball: The ball identified as the white ball.
    """
    if black_ball:
        for ball in balls:
            if ball.center == black_ball.center:
                ball.final_color = "black"
                break
    if white_ball:
        for ball in balls:
            if ball.center == white_ball.center:
                ball.final_color = "white"
                break


def _load_or_find_table_rectangle(image_path: str) -> Optional[Rectangle]:
    """
    Loads the table rectangle from a cached JSON file or detects it from the image.

    Args:
        image_path: The path to the image file.

    Returns:
        A Rectangle object for the table boundaries, or None if detection fails.
    """
    rect_path = Path(BASE_DIR / RECTANGLE_JSON_PATH)
    if not rect_path.exists():
        print("[DEBUG] Rectangle file not found — detecting new rectangle.")
        return find_table_rectangle(image_path)

    try:
        with open(rect_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[DEBUG] Loaded existing rectangle data: {data}")
        return parse_rectangle_from_data(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[DEBUG] Failed to read rectangle file ({e}), detecting new one.")
        return find_table_rectangle(image_path)


def _find_white_ball_with_ml_tiebreaker(
    ball_candidates: List[Ball],
) -> Optional[Ball]:
    """
    Finds the whitest ball from a list of candidates, using a machine learning
    model as a tie-breaker.

    Args:
        ball_candidates: A list of balls to be evaluated.

    Returns:
        The ball identified as the whitest, or None if the list is empty.
    """
    if not ball_candidates:
        return None

    # Create a list of (ball, legacy_score) tuples
    candidates_with_scores = [
        (ball, _calculate_white_score_average(ball)) for ball in ball_candidates
    ]
    # Sort from highest to lowest legacy score
    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

    top_ball, top_score = candidates_with_scores[0]

    # Collect all contenders within a 10-point range of the top score
    contenders = [
        ball for ball, score in candidates_with_scores if (top_score - score) < 10
    ]

    if len(contenders) == 1:
        Debugger.log(f"✅ Clear white ball winner (Legacy Score): {top_score:.1f}")
        return top_ball

    # If there's a tie, use the ML model to decide
    Debugger.log(
        f"⚔️ Tie detected! {len(contenders)} candidates within 10 points. "
        f"Using ML model for tie-breaking."
    )
    best_ml_ball = None
    best_ml_score = -1.0

    for ball in contenders:
        current_ml_score = 0.0
        if ball.single_ball_path and os.path.exists(ball.single_ball_path):
            image = cv2.imread(ball.single_ball_path)
            current_ml_score = test_is_white_model.get_white_score(image)

        Debugger.log(
            f"   - Candidate at {ball.center}: "
            f"Legacy={_calculate_white_score_average(ball):.1f} | ML_Conf={current_ml_score:.4f}"
        )
        if current_ml_score > best_ml_score:
            best_ml_score = current_ml_score
            best_ml_ball = ball

    if best_ml_ball:
        Debugger.log(
            f"   🏆 ML Chose ball at {best_ml_ball.center} "
            f"with confidence {best_ml_score:.4f}"
        )
    return best_ml_ball


def full_analyzer_pipeline(image_path: str) -> AnalyzerResult:
    """
    Executes the complete pool table analysis pipeline on a given image.

    This function orchestrates the following steps:
    1. Detects or loads the table rectangle.
    2. Runs multiple ball detection algorithms.
    3. Merges results from all detection pipelines.
    4. Draws detected items on a final output image for visualization.
    5. Crops and saves individual ball images for classification.
    6. Scores balls for color (white, black, solid, striped).
    7. Identifies the white and black balls, using an ML model for tie-breaking.
    8. Uses a prediction model to classify any remaining undefined balls.
    9. Packages all results into an AnalyzerResult object.

    Args:
        image_path: The path to the image to be analyzed.

    Returns:
        An AnalyzerResult object containing all detected pockets, balls,
        and the identified black and white balls.
    """
    Debugger.log(f"🚀 Starting full analyzer pipeline for: {image_path}")

    # Step 1: Detect or load table rectangle
    table_rectangle = _load_or_find_table_rectangle(image_path)
    if table_rectangle is None:
        Debugger.error("❌ Table rectangle could not be determined. Aborting.")
        return AnalyzerResult()

    # Step 2: Run various ball detection pipelines
    sub_photos, main_photo = run_full_analysis(image_path)
    black_and_white_ball_list = run_ball_detection(image_path)
    all_pockets: List[Pocket] = extract_pocket_images_from_rectangle(
        image_path, table_rectangle
    )
    Debugger.log(f"Pocket image paths: {[p.pocket_img_path for p in all_pockets]}")

    # Step 3: Merge results
    if not sub_photos or not main_photo:
        Debugger.error("❌ Ball analysis failed or no data returned. Aborting.")
        return AnalyzerResult(pockets=all_pockets)

    total_before_merge = (
        len(main_photo.balls)
        + sum(len(p.balls) for p in sub_photos)
        + len(black_and_white_ball_list)
    )
    merged_photo = mergeData(
        main_photo, sub_photos, black_and_white_ball_list, table_rectangle
    )
    Debugger.log(f"✅ Merged {total_before_merge} → {len(merged_photo.balls)} balls")
    sorted_balls = sorted(merged_photo.balls, key=lambda b: b.center[0])
    Debugger.log(f"📦 Total unique balls: {len(sorted_balls)}")

    # Step 4: Save ball crops and draw debug image
    output_final_path = get_output_path("final_detected.png", sub_dir="pipeline")
    draw_balls_on_image(
        merged_photo, image_path, output_final_path, table_rectangle, all_pockets
    )
    Debugger.log(f"🖼️ Final debug image saved to {output_final_path}")

    crop_and_save_balls(image_path, sorted_balls)
    Debugger.log("✂️ Cropped and saved individual ball images.")

    # Step 5: Score balls for each color type
    score_balls(sorted_balls, build_white_suite(), build_black_suite())
    assert_scored(sorted_balls)

    # Step 6: Identify white and black balls
    whitest_ball = _find_white_ball_with_ml_tiebreaker(sorted_balls)
    blackest_ball = max(sorted_balls, key=_calculate_black_score_average, default=None)

    # Step 7: Update ball classifications
    _insert_black_and_white_balls(sorted_balls, blackest_ball, whitest_ball)
    update_undefined_balls(sorted_balls)

    # Final logging and packaging
    if whitest_ball:
        Debugger.log(f"⚪ White ball found at {whitest_ball.center}")
        if os.path.exists(whitest_ball.single_ball_path):
            cv2.imwrite(
                get_output_path("white.png", sub_dir="balls"),
                cv2.imread(whitest_ball.single_ball_path),
            )
    else:
        Debugger.warn("⚪ White ball not found")

    if blackest_ball:
        Debugger.log(f"⚫ Black ball found at {blackest_ball.center}")
        if os.path.exists(blackest_ball.single_ball_path):
            cv2.imwrite(
                get_output_path("black.png", sub_dir="balls"),
                cv2.imread(blackest_ball.single_ball_path),
            )
    else:
        Debugger.warn("⚫ Black ball not found")

    analyzer_result = AnalyzerResult(
        pockets=all_pockets,
        balls=sorted_balls,
        black=blackest_ball,
        white=whitest_ball,
    )
    Debugger.log("✅ Full analyzer pipeline completed successfully.")
    return analyzer_result


if __name__ == "__main__":
    # Example usage
    default_image_path = os.path.join(os.path.dirname(__file__), "input", "first.jpeg")
    result = full_analyzer_pipeline(default_image_path)
    print(
        f"Pipeline returned {len(result.balls)} balls. "
        f"First 5: {[(b.center, b.radius) for b in result.balls[:5]]}"
    )

# Note: The `analyze_ball_brightness` function previously in this file was a
# duplicate of logic handled by the scoring suites and has been removed to
# reduce redundancy. The core logic is now encapsulated within the scoring
# and ML tie-breaking functions.
