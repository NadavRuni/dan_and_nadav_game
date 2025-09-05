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

    for b in analysis.get("balls", []):
        btype = b.get("type", "other")
        bid = b.get("index")

        x_game = y_game = None

        # ---- עדיפות 1: table_uv (הכי מדויק כי כבר תוקנה פרספקטיבה) ----
        uv = b.get("table_uv")
        if uv is not None and "u" in uv and "v" in uv:
            has_uv = True
            u = float(uv["u"])
            v = float(uv["v"])
            # מערכת המשחק היא Bottom-Left, וה-uv הגיעו עם Top-Left ⇒ להפוך את v
            x_game = u * TABLE_LENGTH
            y_game = (1.0 - v) * TABLE_WIDTH
            x_game = clamp_to_table(x_game, TABLE_LENGTH)
            y_game = clamp_to_table(y_game, TABLE_WIDTH)

        # ---- עדיפות 2: rel_to_BL_px (מה-pipeline החדש; Δ מ-BL בפיקסלים) ----
        if x_game is None or y_game is None:
            rel = b.get("rel_to_BL_px")
            if rel is not None and ("x" in rel and "y" in rel):
                x_px = float(rel["x"])
                y_px = float(rel["y"])
                x_game = clamp_to_table(x_px * sx, TABLE_LENGTH)
                y_game = clamp_to_table(y_px * sy, TABLE_WIDTH)

        # ---- עדיפות 3: x_px/y_px (תמיכה לאחור ב-JSON הישן) ----
        if (x_game is None or y_game is None) and ("x_px" in b and "y_px" in b):
            x_px = float(b["x_px"])
            y_px = float(b["y_px"])
            x_game = clamp_to_table(x_px * sx, TABLE_LENGTH)
            y_game = clamp_to_table(y_px * sy, TABLE_WIDTH)

        # ---- עדיפות 4: center_px + origin_px (נבנה Δ עצמאית אם צריך) ----
        if (x_game is None or y_game is None) and (
            "center_px" in b and "origin_px" in analysis
        ):
            cx = float(b["center_px"]["x"])
            cy = float(b["center_px"]["y"])
            ox = float(analysis["origin_px"]["x"])
            oy = float(analysis["origin_px"]["y"])
            x_px = cx - ox
            y_px = oy - cy  # ציר Y הפוך בפיקסלים → למעלה חיובי
            x_game = clamp_to_table(x_px * sx, TABLE_LENGTH)
            y_game = clamp_to_table(y_px * sy, TABLE_WIDTH)

        # אם עדיין אין ערכים — דלג על הכדור
        if x_game is None or y_game is None:
            continue

        # מזהה/טיפוס
        if btype != "white" and btype != "black":
            used_ids.add(bid)
            btype = "solid"

        balls.append(
            Ball(
                ball_id=bid,
                x_cord=x_game,
                y_cord=y_game,
                ball_type=btype,
                radius=BALL_RADIUS,
            )
        )

    # אפשרי: לוג קטן כדי להבין באיזה נתיב השתמשנו
    if has_uv:
        print("[build] used table_uv (rectified coords)")
    else:
        print("[build] used pixel deltas fallback (sx/sy)")

    return Table(TABLE_LENGTH, TABLE_WIDTH, balls)


def start_build_table_from_img():

    analysis = load_analysis(OUTPUT_JSON_PATH)
    table = build_table_from_analysis(analysis)
    print(f"Built table with {len(table.balls)} balls from {OUTPUT_JSON_PATH}")
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
    line_drawer.draw_lines()


if __name__ == "__main__":
    start_build_table_from_img()
