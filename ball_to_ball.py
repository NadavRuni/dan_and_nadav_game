"""
A script for testing the game analysis with a predefined table layout.

This script sets up a specific arrangement of balls on the table and then runs
the game analyzer to find the best possible shot, including considering
ball-to-ball combination shots.
"""

from const_numbers import get_ball_radius, get_table_length, get_table_width
from game_class.C_ball import GameBall
from game_class.C_gameAnalayzer import GameAnalayzer
from game_class.C_table import Table


def main() -> None:
    """
    Sets up a predefined table, runs the analysis, and prints the best shots.
    """
    # Define the positions of the white, black, and other balls
    white_ball = GameBall(0, 180, 61, "white", get_ball_radius())
    black_ball = GameBall(8, 200, 70, "black", get_ball_radius())

    other_balls = [
        GameBall(1, 35, 30, "solid", get_ball_radius()),
        GameBall(2, 9, 134, "striped", get_ball_radius()),
        GameBall(3, 100, 55, "solid", get_ball_radius()),
        GameBall(5, 135, 35, "solid", get_ball_radius()),
        GameBall(6, 279, 28, "striped", get_ball_radius()),
        GameBall(7, 102, 92, "solid", get_ball_radius()),
        GameBall(9, 215, 50, "striped", get_ball_radius()),
        GameBall(10, 230, 60, "solid", get_ball_radius()),
        GameBall(11, 80, 74, "striped", get_ball_radius()),
        GameBall(12, 255, 21, "solid", get_ball_radius()),
    ]

    # Create the table with all the balls
    table = Table(
        get_table_length(), get_table_width(), [white_ball, black_ball] + other_balls
    )

    # Analyze the game state
    game_analyzer = GameAnalayzer(table)
    best_shot = game_analyzer.find_best_overall_shot("striped")

    if not best_shot:
        print("[DEBUG] No direct valid shot found. Trying ball-to-ball shots...")
        best_shot = game_analyzer.find_best_overall_shot_ball_to_ball("striped")

    if best_shot:
        if len(best_shot) > 0:
            print("Best shot is:", best_shot[0])
        if len(best_shot) > 1:
            print("Second best shot is:", best_shot[1])
        if len(best_shot) > 2:
            print("Third best shot is:", best_shot[2])
    else:
        print("[DEBUG] No valid shots found, including ball-to-ball combinations.")

    # The drawing calls are commented out but can be enabled for visualization.
    # from game_class.C_draw import draw_table
    # if best_shot:
    #     print("Drawing best shot on table...")
    #     draw_table(table, best_shot=best_shot[0])
    # draw_table(table)


if __name__ == "__main__":
    main()
