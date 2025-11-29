import os
import sys
import json
import shutil
import matplotlib.pyplot as plt

# ===== Make local imports work no matter where you run from =====
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ===============================================================

from const_numbers import *
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_draw import draw_table
from game_class.C_gameAnalayzer import GameAnalayzer
from game_class.C_lineDrawer import LineDrawer
from const_numbers import OUTPUT_JSON_PATH, OUTPUT_IMAGE_PATH
from game_class.C_bestShot import BestShot
from game_class.C_bestShotBallToBall import BestShotBallToBall
from game_class.C_bestShot_use_wall import BestWallShot


def load_analysis(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_to_table(x: float, length: float) -> float:
    """גזירה לגבולות השולחן תוך שמירה על רדיוס הכדור."""
    return max(get_ball_radius(), min(length - get_ball_radius(), x))


def build_table_from_analysis(analysis: dict):
    # מידות התמונה בפיקסלים (ל־fallback)
    width_px = float(analysis.get("table_size_px", {}).get("width_px", 1.0))
    height_px = float(analysis.get("table_size_px", {}).get("height_px", 1.0))
    set_height_px(height_px)
    set_width_px(width_px)

    # upside-down pockets conversion
    # because the detected pockets are in image coordinates (0,0 top-left)
    # set_set_detected_pockets_to_upside_downside()

    # סקלת פיקסלים → יחידות משחק (ל־fallback)
    sx = get_table_length() / max(1.0, width_px)
    sy = get_table_width() / max(1.0, height_px)

    # אם קיימת הומוגרפיה/קואורדינטות מנורמלות מה-pipeline — נעדיף אותן
    # table_uv: u,v ב-[0..1] כאשר u משמאל לימין, v מלמעלה למטה (Top-Left origin)
    has_uv = False

    balls = []
    print(analysis.get("balls", []))

    for b in analysis.get("balls", []):
        btype = b.get("type", "other")
        bid = b.get("index")

        x_game = y_game = None

        # ---- עדיפות 4 (חדשה): center_px בלבד ----
        if (x_game is None or y_game is None) and "center_px" in b:
            # נשלוף את מיקום הכדור במערכת הפיקסלים
            cx = float(b["center_px"]["x"])
            cy = float(b["center_px"]["y"])

            # if we are using pre defined pockets, we dont need to convert nothing
            pockets = analysis.get("pockets", {})
            if pockets and "BL" in pockets and "TR" in pockets:

                x_game = cx

                y_game = cy

            else:

                # נשתמש בהיפוך פשוט של הציר האנכי
                x_game = clamp_to_table(cx * sx, get_table_length())
                y_game = clamp_to_table((height_px - cy) * sy, get_table_width())

        # אם עדיין אין ערכים — דלג על הכדור
        if x_game is None or y_game is None:
            continue

        balls.append(
            Ball(
                ball_id=bid,
                x_cord=x_game,
                y_cord=y_game,
                ball_type=btype,
                radius=get_ball_radius(),
            )
        )

    # אפשרי: לוג קטן כדי להבין באיזה נתיב השתמשנו
    if has_uv:
        print("[build] used table_uv (rectified coords)")
    else:
        print("[build] used pixel deltas fallback (sx/sy)")

    return Table(get_table_length(), get_table_width(), balls)


def start_build_table_from_img():

    analysis = load_analysis(OUTPUT_JSON_PATH)
    print(f"Loaded analysis from {OUTPUT_JSON_PATH}")

    table = build_table_from_analysis(analysis)
    print(f"Built table with {len(table.balls)} balls from {OUTPUT_JSON_PATH}")

    # draw_table(table)

    game = GameAnalayzer(table)
    print("Analyzing best shot...")
    print(table.pockets)
    best_shot = game.find_best_overall_shot(get_ball_type())

    # ציור
    if best_shot:
        if len(best_shot) > 0:
            print("best shot is:", best_shot[0])
        if len(best_shot) > 1:
            print("second best shot is:", best_shot[1])
        if len(best_shot) > 2:
            print("third best shot is:", best_shot[2])

        p, lines = draw_table(table, best_shot=best_shot[0])

        plot_output_dir = BASE_DIR / "out" / "debug" / "plot"
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        plot_file_path = plot_output_dir / "table_simulation.png"
        plt.savefig(plot_file_path)
        plt.close(p)  # Close the figure to free up memory

        line_drawer = LineDrawer(OUTPUT_JSON_PATH, best_shot[0], OUTPUT_IMAGE_PATH)
        line_drawer.show_contact_hit()  # need to be first before draw_lines
        if isinstance(best_shot[0], BestWallShot):
            print("Drawing wall-based shot lines...")
            line_drawer.draw_lines_with_wall(
                (
                    best_shot[0].point_with_the_wall[0],
                    best_shot[0].point_with_the_wall[1],
                )
            )
        elif isinstance(best_shot[0], BestShotBallToBall):
            print("Drawing ball-to-ball shot lines...")
            line_drawer.draw_combo_lines()
        else:
            line_drawer.draw_lines()
    else:
        print("No best shot found. Saving original image as output.")
        # Optional: Save the image without any lines
        analysis = load_analysis(OUTPUT_JSON_PATH)
        if analysis.get("image_path"):
            from shutil import copyfile

            copyfile(analysis["image_path"], OUTPUT_IMAGE_PATH)
        return "false"


if __name__ == "__main__":
    start_build_table_from_img()
