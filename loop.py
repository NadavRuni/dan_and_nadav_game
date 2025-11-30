"""
Runs a simulation of the game analysis on randomly generated table layouts.

This script creates a series of random table setups and, for each one, runs
the full game analysis pipeline to find the best shot. The resulting table
state, with the suggested shot, is saved as an image.
"""

import os
import random

import matplotlib.pyplot as plt

from const_numbers import get_ball_radius, get_table_length, get_table_width
from game_class.C_ball import GameBall
from game_class.C_draw import draw_table
from game_class.C_gameAnalayzer import GameAnalayzer
from game_class.C_table import Table


def run_one_simulation(run_idx: int, out_dir: str) -> None:
    """
    Creates a single random table layout and runs the analysis on it.

    Args:
        run_idx: The index number of the current simulation run.
        out_dir: The directory where the output image will be saved.
    """
    # Create the white and black balls
    x_white, y_white = get_table_length() / 2, get_table_width() / 2
    white_ball = GameBall(0, x_white, y_white, "white", get_ball_radius())

    x_black = random.uniform(
        get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
    )
    y_black = random.uniform(
        get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
    )
    black_ball = GameBall(8, x_black, y_black, "black", get_ball_radius())

    # Create a list of colored balls
    balls = []
    ball_colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "brown",
        "pink",
        "cyan",
        "magenta",
        "lime",
        "teal",
        "gold",
        "silver",
    ]
    for i, color in enumerate(ball_colors, start=1):
        ball_type = "solid" if i <= 7 else "striped"
        x = random.uniform(
            get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
        )
        y = random.uniform(
            get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
        )
        balls.append(GameBall(i, x, y, ball_type, get_ball_radius()))

    # Create a table with all the balls
    table = Table(
        get_table_length(), get_table_width(), [white_ball, black_ball] + balls
    )

    game_analyzer = GameAnalayzer(table)
    best_shot_info = game_analyzer.find_best_overall_shot("striped")

    if best_shot_info:
        print(f"[{run_idx:02d}] Best shot found: {best_shot_info[0]}")

    # Draw the table and save it to a file
    fig = draw_table(table, best_shot=best_shot_info[0] if best_shot_info else None)
    save_path = os.path.join(out_dir, f"table_{run_idx:02d}.png")
    fig.savefig(save_path)
    plt.close(fig)


def main():
    """
    Runs a series of game analysis simulations and saves the results as images.
    """
    output_directory = "simulations"
    os.makedirs(output_directory, exist_ok=True)

    num_simulations = 20
    for i in range(1, num_simulations + 1):
        run_one_simulation(i, output_directory)

    print(f"\n✅ Done! Saved {num_simulations} images in ./{output_directory}")


if __name__ == "__main__":
    main()
