"""
A script for testing the game analysis with a predefined table layout,
with a focus on evaluating wall shots.
"""

import math

from const_numbers import get_ball_radius, get_table_length, get_table_width
from game_class.C_ball import GameBall
from game_class.C_gameAnalayzer import GameAnalayzer
from game_class.C_table import Table


def main() -> None:
    """
    Sets up a predefined table layout and runs the game analysis to find the
    best shots, including wall shots.
    """
    # Define a specific layout of balls on the table
    white_ball = GameBall(0, 100, 111, "white", get_ball_radius())
    black_ball = GameBall(8, 220, 40, "black", get_ball_radius())
    my_ball = GameBall(1, 80, 50, "striped", get_ball_radius())

    # Add a set of obstacle balls with fixed coordinates
    obstacle_balls = [
        GameBall(2, 275, 135, "solid", get_ball_radius()),
        GameBall(3, 145, 130, "solid", get_ball_radius()),
        GameBall(9, 156, 135, "solid", get_ball_radius()),
        GameBall(4, 275, 15, "solid", get_ball_radius()),
        GameBall(5, 135, 15, "solid", get_ball_radius()),
        GameBall(6, 155, 15, "solid", get_ball_radius()),
        GameBall(7, 12, 15, "solid", get_ball_radius()),
    ]

    # Create the table with all the balls
    table = Table(
        get_table_length(),
        get_table_width(),
        [white_ball, black_ball, my_ball] + obstacle_balls,
    )

    # Analyze the game state for the best shot for the 'striped' player
    game_analyzer = GameAnalayzer(table)
    best_shots = game_analyzer.find_best_overall_shot("striped")

    if best_shots:
        if len(best_shots) > 0:
            print("Best shot is:", best_shots[0])
        if len(best_shots) > 1:
            print("Second best shot is:", best_shots[1])
        if len(best_shots) > 2:
            print("Third best shot is:", best_shots[2])
    else:
        print("No valid shots were found.")

    # The drawing calls can be uncommented for visualization.
    # from game_class.C_draw import draw_table
    # if best_shots:
    #     draw_table(table, best_shot=best_shots[0])
    # else:
    #     draw_table(table)


if __name__ == "__main__":
    main()
