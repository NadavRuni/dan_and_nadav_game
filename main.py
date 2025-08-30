from game_class.C_table import Table
from game_class.C_draw import *
from const_numbers import *
from game_class.C_ball import *
from game_class.C_calc import *
from game_class.C_bestShot import *
from dan.build_table_from_image import build_table_from_image
from dan.pipe_Line import IMAGE_PATH

from game_class.C_gameAnalayzer import *
import random


def main():
    x_white, y_white = TABLE_LENGTH / 2, TABLE_WIDTH / 2
    white = Ball(0, x_white, y_white, "white", BALL_RADIUS)

    x_black = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
    y_black = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)
    black = Ball(8, x_black, y_black, "black", BALL_RADIUS)

    balls = []
    colors = [
        ("red", "solid"),
        ("blue", "striped"),
        ("green", "solid"),
        ("yellow", "striped"),
        ("orange", "solid"),
        ("purple", "striped"),
        ("brown", "solid"),
        ("pink", "striped"),
        ("cyan", "solid"),
        ("magenta", "striped"),
        ("lime", "solid"),
        ("teal", "striped"),
        ("gold", "solid"),
        ("silver", "striped"),
    ]

    for i, (color, ball_type) in enumerate(colors, start=1):
        x = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
        y = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)
        balls.append(Ball(str(i), x, y, ball_type, BALL_RADIUS))

    # יצירת שולחן עם 16 כדורים
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black] + balls)

    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot("striped")
    if len(best_shot) > 0:
        print("best shot is:", best_shot[0])
    if len(best_shot) > 1:
        print("second best shot is:", best_shot[1])
    if len(best_shot) > 2:
        print("third best shot is:", best_shot[2])

    # ציור
    draw_table(table, best_shot=best_shot[0])


def dan():
    
    tbl = build_table_from_image(IMAGE_PATH)
    draw_table(tbl)


if __name__ == "__main__":
   # main()
    dan()

