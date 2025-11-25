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
    x_white, y_white = get_table_length() / 2, get_table_width() / 2
    white = Ball(0, x_white, y_white, "white", get_ball_radius())

    x_black = random.uniform(
        get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
    )
    y_black = random.uniform(
        get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
    )
    black = Ball(8, x_black, y_black, "black", get_ball_radius())

    balls = []
    colors = [
        ("red", "solid"),
        ("blue", "solid"),
        ("green", "solid"),
        ("yellow", "solid"),
        ("orange", "solid"),
        ("purple", "solid"),
        ("brown", "striped"),
        ("pink", "striped"),
        ("cyan", "striped"),
        ("magenta", "striped"),
        ("lime", "solid"),
        ("teal", "striped"),
        ("gold", "solid"),
        ("silver", "striped"),
    ]

    for i, (color, ball_type) in enumerate(colors, start=1):
        x = random.uniform(
            get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
        )
        y = random.uniform(
            get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
        )
        balls.append(Ball(str(i), x, y, ball_type, get_ball_radius()))

    # יצירת שולחן עם 16 כדורים
    table = Table(get_table_length(), get_table_width(), [white, black] + balls)
    draw_table(table)

    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot("striped")
    if len(best_shot) > 0:
        print("best shot is:", best_shot[0])
        draw_table(table, best_shot=best_shot[0])
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
