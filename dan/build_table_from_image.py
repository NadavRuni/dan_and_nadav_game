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

ANALYSIS_JSON_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/analysis-table-12.json"

def load_analysis(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_table_from_analysis(analysis: dict):
    # קרא מידות בפיקסלים (נפילה לגיבוי אם חסר)
    width_px  = float(analysis.get("table_size_px", {}).get("width_px", 1.0))
    height_px = float(analysis.get("table_size_px", {}).get("height_px", 1.0))

    # סקלת פיקסלים→יחידות משחק
    sx = TABLE_LENGTH / max(1.0, width_px)
    sy = TABLE_WIDTH  / max(1.0, height_px)

    balls = []
    next_id = 1
    used_ids = set([0, 8])

    for b in analysis["balls"]:
        x_game = max(BALL_RADIUS, min(TABLE_LENGTH - BALL_RADIUS, float(b["x_px"]) * sx))
        y_game = max(BALL_RADIUS, min(TABLE_WIDTH  - BALL_RADIUS, float(b["y_px"]) * sy))
        btype = b.get("type", "other")

        if btype == "white":
            bid = 0
        elif btype == "black":
            bid = 8        
        else:
            while next_id in used_ids or next_id == 8:
                next_id += 1
            bid = next_id
            used_ids.add(bid)
            next_id += 1
            btype="solid" 

        balls.append(Ball(ball_id=bid, x_cord=x_game, y_cord=y_game, ball_type=btype, radius=BALL_RADIUS))

    return Table(TABLE_LENGTH, TABLE_WIDTH, balls)

if __name__ == "__main__":
    analysis = load_analysis(ANALYSIS_JSON_PATH)
    table = build_table_from_analysis(analysis)
    print(f"Built table with {len(table.balls)} balls from {ANALYSIS_JSON_PATH}")
    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot("solid")
    if len(best_shot) > 0:
        print("best shot is:", best_shot[0])
    if len(best_shot) > 1:
        print("second best shot is:", best_shot[1])
    if len(best_shot) > 2:
        print("third best shot is:", best_shot[2])

    # ציור
    draw_table(table, best_shot=best_shot[0])
    draw_table(table)

