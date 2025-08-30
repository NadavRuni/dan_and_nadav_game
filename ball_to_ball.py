from game_class.C_table import Table
from game_class.C_draw import draw_table
from const_numbers import *
from game_class.C_ball import Ball
from game_class.C_gameAnalayzer import GameAnalayzer


def main():
    # כאן אתה מגדיר את כל הכדורים שאתה רוצה על השולחן

    white = Ball(0, 180,61, "white", BALL_RADIUS)  # כדור לבן
    black = Ball(8, 200, 70, "black", BALL_RADIUS)  # כדור שחור במיקום שבחרת

    balls = [
        Ball(1, 35, 30, "solid", BALL_RADIUS),
        Ball(2, 9, 134, "striped", BALL_RADIUS),
        Ball(3, 100, 55, "solid", BALL_RADIUS),
        Ball(5, 135, 35, "solid", BALL_RADIUS),
        Ball(6, 279, 28, "striped", BALL_RADIUS),
        Ball(7, 102, 92, "solid", BALL_RADIUS),
        Ball(9, 215, 50, "striped", BALL_RADIUS),
        Ball(10, 230, 60, "solid", BALL_RADIUS),
        Ball(11, 80, 74, "striped", BALL_RADIUS),
        Ball(12, 255, 21, "solid", BALL_RADIUS),
    ]

    # יצירת שולחן עם הכדורים
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black] + balls)

    # ניתוח
    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot("striped")
    if not best_shot:
        # enter here the logic!!!!!!!!!!!!!!!
        print("[DEBUG] No valid shot found.")
        print("Trying ball to ball shots...")
        best_shot = game.find_best_overall_shot_ball_to_ball("striped")
    if len(best_shot) > 0:
        print("best shot is:", best_shot[0])
    if len(best_shot) > 1:
        print("second best     shot is:", best_shot[1])
    if len(best_shot) > 2:
        print("third best shot is:", best_shot[2])

    # ציור
    if best_shot:
        draw_table(table, best_shot=best_shot[0])
    draw_table(table)


if __name__ == "__main__":
    main()
