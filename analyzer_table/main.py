"""
Command-line entry point for running the full ball detection pipeline.

This script orchestrates the entire analysis process, including running
different detection algorithms, merging the results, and identifying the white
and black balls.
"""

import os
import sys
from pathlib import Path
from typing import List

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.detect_ball.analyzer_runner import run_full_analysis
from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.detect_ball.draw_utils import draw_balls_on_image
from analyzer_table.detect_ball.merge_utils import mergeData
from analyzer_table.launcher_helper.black_and_white_launcher import run_ball_detection
from analyzer_table.launcher_helper.detect_ball_color import (
    find_whitest_and_darkest_balls,
)
from analyzer_table.launcher_helper.json_models import Ball

# Add project root to allow sibling imports. This is not a recommended practice.
sys.path.append(os.path.dirname(__file__))

# --- Constants for terminal colors ---
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def main_pipeline(image_path: str):
    """
    Runs the main analysis pipeline on a given image path.

    Args:
        image_path: The full path to the image to be analyzed.
    """
    Debugger.log("🚀 Starting full analyzer process")

    # Run the different analysis components
    sub_photos, main_photo = run_full_analysis(image_path)
    black_and_white_ball_list = run_ball_detection(image_path)
    table_rectangle = find_table_rectangle(image_path)

    if not sub_photos or not main_photo:
        Debugger.error("❌ Analysis failed or no data returned.")
        return

    Debugger.warn("🏁 Analysis complete — merging data...")
    total_before_merge = (
        len(main_photo.balls)
        + sum(len(p.balls) for p in sub_photos)
        + len(black_and_white_ball_list)
    )

    # Merge results from all detection passes
    merged_photo = mergeData(
        main_photo, sub_photos, black_and_white_ball_list, table_rectangle
    )
    total_after_merge = len(merged_photo.balls)
    merged_count = total_before_merge - total_after_merge

    # Create and save the final visualization image
    output_final_path = Path(__file__).resolve().parent / "out" / "final_detected.png"
    output_final_path.parent.mkdir(parents=True, exist_ok=True)
    draw_balls_on_image(
        merged_photo, image_path, str(output_final_path), table_rectangle
    )

    # Print a summary of the results
    sorted_balls = sorted(merged_photo.balls, key=lambda b: b.center[0])
    print(f"\n{BOLD}{CYAN}========== ⚪ FINAL SUMMARY (sorted by X) =========={RESET}")
    print(f"Total detected balls (before merge): {total_before_merge}")
    print(f"Balls merged (duplicates removed):   {merged_count}")
    print(f"Unique balls remaining:              {total_after_merge}\n")

    for i, ball in enumerate(sorted_balls, 1):
        print(
            f"⚪ Ball #{i}: center=({ball.center[0]}, {ball.center[1]}), "
            f"radius={ball.radius}"
        )

    print(f"\n🖼️ Final image saved to: {output_final_path}")
    if table_rectangle:
        print(
            f"📐 Table rectangle: TL={table_rectangle.top_left}, "
            f"TR={table_rectangle.top_right}, BL={table_rectangle.bottom_left}, "
            f"BR={table_rectangle.bottom_right}"
        )

    # Find and print the whitest and darkest balls
    output_balls_dir = Path(__file__).resolve().parent / "out" / "balls"
    whitest, darkest = find_whitest_and_darkest_balls(
        image_path, sorted_balls, str(output_balls_dir)
    )
    print(
        f"\n✨ Whitest ball:  center={whitest.center if whitest else 'N/A'}, "
        f"radius={whitest.radius if whitest else 'N/A'}"
    )
    print(
        f"⚫ Darkest ball:  center={darkest.center if darkest else 'N/A'}, "
        f"radius={darkest.radius if darkest else 'N/A'}"
    )
    print(f"\n{GREEN}{BOLD}✅ Done! Merged data successfully.{RESET}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    # Default image for testing
    image_name = "first.jpeg"
    full_image_path = base_dir / "input" / image_name
    main_pipeline(str(full_image_path))
