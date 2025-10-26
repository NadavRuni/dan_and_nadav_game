import os
import sys
import json

# ===== Make local imports work no matter where you run from =====
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ===============================================================

from const_numbers import TABLE_LENGTH, TABLE_WIDTH, BALL_RADIUS
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
    return max(BALL_RADIUS, min(length - BALL_RADIUS, x))


def build_table_from_analysis(analysis: dict):
    # מידות התמונה בפיקסלים (ל־fallback)
    width_px = float(analysis.get("table_size_px", {}).get("width_px", 1.0))
    height_px = float(analysis.get("table_size_px", {}).get("height_px", 1.0))

    # סקלת פיקסלים → יחידות משחק (ל־fallback)
    sx = TABLE_LENGTH / max(1.0, width_px)
    sy = TABLE_WIDTH / max(1.0, height_px)

    # אם קיימת הומוגרפיה/קואורדינטות מנורמלות מה-pipeline — נעדיף אותן
    # table_uv: u,v ב-[0..1] כאשר u משמאל לימין, v מלמעלה למטה (Top-Left origin)
    has_uv = False

    balls = []
    next_id = 1
    used_ids = set([0, 8])
    print("[build] building table from analysis...")
    print (analysis.get("balls", []))

    for b in analysis.get("balls", []):
        btype = b.get("type", "other")
        bid = b.get("index")

        x_game = y_game = None

                # ---- עדיפות 4 (חדשה): center_px בלבד ----
        if (x_game is None or y_game is None) and "center_px" in b:
            # נשלוף את מיקום הכדור במערכת הפיקסלים
            cx = float(b["center_px"]["x"])
            cy = float(b["center_px"]["y"])

            # --- נחשב סקלות המרה אם לא חושבו קודם ---
            # נשתמש תחילה בגבולות מהכיסים אם קיימים (מדויק יותר)
            pockets = analysis.get("pockets", {})
            if pockets and "BL" in pockets and "TR" in pockets:
                x_min = pockets["BL"]["x"]
                y_min = pockets["TR"]["y"]
                x_max = pockets["TR"]["x"]
                y_max = pockets["BL"]["y"]

                table_width_px = x_max - x_min
                table_height_px = y_max - y_min

                sx = TABLE_LENGTH / max(1.0, table_width_px)
                sy = TABLE_WIDTH / max(1.0, table_height_px)

                # נחשב מיקום יחסי לפי גבולות השולחן
                x_game = clamp_to_table((cx - x_min) * sx, TABLE_LENGTH)
                # ציר Y בפיקסלים הפוך, לכן נשתמש ב־(y_max - cy)
                y_game = clamp_to_table((y_max - cy) * sy, TABLE_WIDTH)

                print(f"[build] using pocket-based mapping for ball id={bid}")
            else:
                # fallback — אם אין מידע על כיסים
                sx = TABLE_LENGTH / max(1.0, width_px)
                sy = TABLE_WIDTH / max(1.0, height_px)

                # נשתמש בהיפוך פשוט של הציר האנכי
                x_game = clamp_to_table(cx * sx, TABLE_LENGTH)
                y_game = clamp_to_table((height_px - cy) * sy, TABLE_WIDTH)

                print(f"[build] using fallback image-size mapping for ball id={bid}")


        # אם עדיין אין ערכים — דלג על הכדור
        if x_game is None or y_game is None:
            print(f"[build] warning: skipping ball id={bid} due to missing position")
            continue

        balls.append(
            Ball(
                ball_id=bid,
                x_cord=x_game,
                y_cord=y_game,
                ball_type=btype,
                radius=BALL_RADIUS,
            )
        )
        print (f"[build] added ball id={bid}, type={btype}, pos=({x_game:.1f}, {y_game:.1f}")
        

    # אפשרי: לוג קטן כדי להבין באיזה נתיב השתמשנו
    if has_uv:
        print("[build] used table_uv (rectified coords)")
    else:
        print("[build] used pixel deltas fallback (sx/sy)")

    return Table(TABLE_LENGTH, TABLE_WIDTH, balls)


def start_build_table_from_img():

    analysis = load_analysis(OUTPUT_JSON_PATH)
    print(f"Loaded analysis from {OUTPUT_JSON_PATH}")
    
    table = build_table_from_analysis(analysis)
    print(f"Built table with {len(table.balls)} balls from {OUTPUT_JSON_PATH}")
    draw_table(table)

    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot("solid")
    if len(best_shot) > 0:
        print("best shot is:", best_shot[0])
    if len(best_shot) > 1:
        print("second best shot is:", best_shot[1])
    if len(best_shot) > 2:
        print("third best shot is:", best_shot[2])

    # ציור
    p, lines = draw_table(table, best_shot=best_shot[0])
    line_drawer = LineDrawer(OUTPUT_JSON_PATH, best_shot[0], OUTPUT_IMAGE_PATH)
    line_drawer.show_contact_hit() # need to be first before draw_lines
    if (isinstance(best_shot[0],BestWallShot)):
        print("Drawing wall-based shot lines...")
        line_drawer.draw_lines_with_wall(
            (best_shot[0].point_with_the_wall[0], best_shot[0].point_with_the_wall[1])
        )
    else :
        line_drawer.draw_lines()


if __name__ == "__main__":
    start_build_table_from_img()
