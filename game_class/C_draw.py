"""
Provides utility functions for drawing the game state using matplotlib.

This module is used to create visual representations of the pool table,
including the balls, pockets, and calculated shot lines, for debugging and
simulation purposes.
"""

import math
import random
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from const_numbers import get_ball_radius, get_table_length, get_table_width
from game_class.C_ball import GameBall
from game_class.C_bestShot import BestShot
from game_class.C_bestShotBallToBall import BestShotBallToBall
from game_class.C_bestShot_use_wall import BestWallShot
from game_class.C_calc import Calculations
from game_class.C_line import Line
from game_class.C_pocket import Pocket
from game_class.C_table import Table


def _draw_table_border(ax, table: Table) -> None:
    """Draws the table border and sets up the plot area."""
    border = plt.Rectangle(
        (0, 0),
        table.length,
        table.width,
        linewidth=15,
        edgecolor="saddlebrown",
        facecolor="green",
    )
    ax.add_patch(border)
    ax.set_facecolor("green")
    ax.set_xlim(0, table.length)
    ax.set_ylim(0, table.width)
    ax.set_aspect("equal", adjustable="box")


def _draw_pockets(ax, pockets: List[Pocket]) -> None:
    """Draws the pockets on the table."""
    for pocket in pockets:
        pocket_circle = plt.Circle(
            pocket.center, pocket.radius, color="black", zorder=3
        )
        ax.add_patch(pocket_circle)


def _draw_balls(ax, balls: List[GameBall]) -> None:
    """Draws the balls on the table."""
    color_map = {"white": "white", "black": "black", "solid": "blue", "striped": "red"}
    for ball in balls:
        face_color = color_map.get(ball.type, "gray")
        circle = plt.Circle(
            (ball.x_cord, ball.y_cord),
            ball.radius,
            color=face_color,
            ec="black",
            zorder=4,
        )
        ax.add_patch(circle)
        text_color = "black" if face_color != "black" else "white"
        ax.text(
            ball.x_cord,
            ball.y_cord,
            str(ball.id),
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            zorder=5,
        )


def _draw_shot_lines(
    ax, best_shot: Optional[BestShot | BestShotBallToBall | BestWallShot]
) -> None:
    """Draws the lines for the calculated best shot."""
    if not best_shot or not best_shot.valid:
        return

    if isinstance(best_shot, BestWallShot):
        for i, ((x1, y1), (x2, y2)) in enumerate(best_shot.get_lines()):
            color = "blue" if i == 0 else ("orange" if i == 1 else "red")
            ax.plot(
                [x1, x2], [y1, y2], linestyle="-", color=color, linewidth=2, zorder=2
            )
    elif isinstance(best_shot, BestShotBallToBall):
        _draw_ball_to_ball_shot(ax, best_shot)
    else:  # Standard shot
        _draw_standard_shot(ax, best_shot)


def _draw_standard_shot(ax, best_shot: BestShot):
    """Draws lines for a standard two-ball shot."""
    ax.plot(
        [best_shot.target.x_cord, best_shot.pocket.center[0]],
        [best_shot.target.y_cord, best_shot.pocket.center[1]],
        linestyle="-",
        color="red",
        linewidth=2,
        zorder=2,
    )
    _draw_contact_line(ax, best_shot.white, best_shot.target, best_shot.pocket)


def _draw_ball_to_ball_shot(ax, best_shot: BestShotBallToBall):
    """Draws lines for a three-ball combination shot."""
    # Line from target to pocket
    ax.plot(
        [best_shot.target.x_cord, best_shot.pocket.center[0]],
        [best_shot.target.y_cord, best_shot.pocket.center[1]],
        linestyle="-",
        color="red",
        linewidth=2,
        zorder=2,
    )
    # Line from cue ball to helper ball (via contact point)
    _draw_contact_line(
        ax, best_shot.white, best_shot.target_helper, best_shot.target, color="blue"
    )
    # Line from helper ball to target ball (via contact point)
    _draw_contact_line(ax, best_shot.target_helper, best_shot.target, best_shot.pocket)


def _draw_contact_line(
    ax,
    cue_ball: GameBall,
    target_ball: GameBall,
    final_destination: Union[Pocket, GameBall],
    color: str = "orange",
) -> None:
    """Draws the path from the cue ball to the contact point on the target."""
    if isinstance(final_destination, GameBall):
        vx = final_destination.x_cord - target_ball.x_cord
        vy = final_destination.y_cord - target_ball.y_cord
    else:  # It's a Pocket
        vx = final_destination.center[0] - target_ball.x_cord
        vy = final_destination.center[1] - target_ball.y_cord

    norm = math.hypot(vx, vy)
    if norm == 0:
        return

    vx /= norm
    vy /= norm

    contact_x = target_ball.x_cord - vx * target_ball.radius
    contact_y = target_ball.y_cord - vy * target_ball.radius

    ax.plot(
        [cue_ball.x_cord, contact_x],
        [cue_ball.y_cord, contact_y],
        linestyle="--",
        color=color,
        linewidth=2,
        zorder=2,
    )
    ax.plot(contact_x, contact_y, "o", color="orange", markersize=6, zorder=6)


def draw_table(
    table: Table,
    lines: Optional[List[Line]] = None,
    best_shot: Optional[BestShot | BestShotBallToBall | BestWallShot] = None,
) -> plt.Figure:
    """
    Draws the entire table, including balls, pockets, and shot lines.

    Args:
        table: The Table object to draw.
        lines: An optional list of custom lines to draw.
        best_shot: An optional BestShot object to visualize.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_table_border(ax, table)
    _draw_pockets(ax, table.pockets)
    _draw_balls(ax, table.balls)

    if lines:
        for line in lines:
            (x1, y1), (x2, y2) = line.as_tuple()
            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle="--",
                color="black",
                linewidth=1.5,
                zorder=2,
            )

    _draw_shot_lines(ax, best_shot)

    return fig
