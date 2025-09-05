from game_class.C_table import Table
from game_class.C_draw  import *
from const_numbers import *
from game_class.C_ball import *
from game_class.C_calc import *
from game_class.C_bestShot import *
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
        x = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
        y = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)
        balls.append(Ball(str(i), x, y, ball_type, BALL_RADIUS))

    # יצירת שולחן עם 16 כדורים
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black] + balls)
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





if __name__ == "__main__":
    main()
