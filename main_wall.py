from game_class.C_table import Table
from game_class.C_draw import *
from const_numbers import *
from game_class.C_ball import *
from game_class.C_calc import *
from game_class.C_bestShot import *
from game_class.C_gameAnalayzer import *
import math


def main():
    # White ball in the center
    x_white, y_white = TABLE_LENGTH / 2, TABLE_WIDTH / 2

    # Choose your ball type
    my_type = "striped"  # or "solid"
    my_id = 1

    # Vector from center to pocket 0 (0,0)
    dx = 0 - x_white
    dy = 0 - y_white
    dist = math.hypot(dx, dy)

    # Normalize direction
    dir_x, dir_y = dx / dist, dy / dist

    # Bigger spacing between balls
    spacing = BALL_RADIUS * 20
    white = Ball(0, 100, 111, "white", BALL_RADIUS)

    # Place black ball first after the white
    black = Ball(8, 220, 40, "black", BALL_RADIUS)

    # Place your ball further along the same line
    x_my = x_white + dir_x * spacing * 2 - 20
    y_my = y_white + dir_y * spacing * 2
    my_ball = Ball(my_id, 80, 50, my_type, BALL_RADIUS)

    # --- Extra balls with fixed coordinates ---
    ball_a = Ball(2, 275, 135, "solid", BALL_RADIUS)  # כדור סוליד
    ball_b = Ball(3, 145, 130, "solid", BALL_RADIUS)  # כדור פסים
    ball_b2 = Ball(9, 156, 135, "solid", BALL_RADIUS)  # כדור פסים

    ball_c = Ball(4, 275, 15, "solid", BALL_RADIUS)  # עוד כדור סוליד
    ball_d = Ball(5, 135, 15, "solid", BALL_RADIUS)  # עוד כדור פסים
    ball_e = Ball(6, 155, 15, "solid", BALL_RADIUS)  # עוד כדור פסים
    ball_f = Ball(7, 12, 15, "solid", BALL_RADIUS)  # עוד כדור פסים

    # Create table with only 3 balls
    table = Table(
        TABLE_LENGTH,
        TABLE_WIDTH,
        [
            white,
            ball_f,
            black,
            my_ball,
            ball_a,
            ball_b,
            ball_c,
            ball_d,
            ball_e,
            ball_b2,
        ],
    )
    draw_table(table)

    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot(my_type)
    if len(best_shot) > 0:
        print("best shot is:", best_shot[0])
        draw_table(table, best_shot=best_shot[0])
    if len(best_shot) > 1:
        print("second best shot is:", best_shot[1])
    if len(best_shot) > 2:
        print("third best shot is:", best_shot[2])


if __name__ == "__main__":
    main()
