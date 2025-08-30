import random
import os
import matplotlib.pyplot as plt
from game_class.C_table import Table
from game_class.C_draw import draw_table
from const_numbers import *
from game_class.C_ball import Ball
from game_class.C_gameAnalayzer import GameAnalayzer


def run_one_simulation(run_idx: int, out_dir: str):
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

    # יצירת שולחן עם כל הכדורים
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black] + balls)

    game = GameAnalayzer(table)
    best_shot = game.find_best_overall_shot("striped")

    if best_shot:
        print(f"[{run_idx}] best shot is:", best_shot[0])

    # ציור ושמירה לקובץ
    fig = draw_table(table, best_shot=best_shot[0] if best_shot else None)
    save_path = os.path.join(out_dir, f"table_{run_idx:02d}.png")
    fig.savefig(save_path)
    plt.close(fig)


def main():
    out_dir = "simulations"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, 21):  # 20 ריצות
        run_one_simulation(i, out_dir)

    print(f"\n✅ Done! Saved 20 images inside ./{out_dir}")


if __name__ == "__main__":
    main()
