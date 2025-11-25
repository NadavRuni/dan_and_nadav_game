import matplotlib.pyplot as plt
import random
from const_numbers import *
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_line import Line
from game_class.C_pocket import Pocket
from game_class.C_calc import *
from typing import List, Optional
from game_class.C_bestShot import BestShot
from game_class.C_bestShotBallToBall import BestShotBallToBall
from game_class.C_bestShot_use_wall import BestWallShot


def draw_table(
    table: Table,
    lines: Optional[List[Line]] = None,
    best_shot: Optional[BestShot | BestShotBallToBall] = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    border = plt.Rectangle(
        (0, 0),
        table.length,
        table.width,
        linewidth=15,
        edgecolor="saddlebrown",
        facecolor="none",
    )
    ax.add_patch(border)
    ax.set_facecolor("green")
    ax.set_xlim(0, table.length)
    ax.set_ylim(0, table.width)
    ax.set_aspect("equal", adjustable="box")
    final_lines = []
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
            final_lines.append(((x1, y1), (x2, y2)))
    if best_shot:
        print(isinstance(best_shot, BestWallShot))
        if not isinstance(best_shot, BestWallShot) or best_shot.angle > 90:
            print("Best shot is BestWallShot draw the line later")
        elif isinstance(best_shot, BestShotBallToBall):
            ax.plot(
                [best_shot.target.x_cord, best_shot.pocket.center[0]],
                [best_shot.target.y_cord, best_shot.pocket.center[1]],
                linestyle="-",
                color="red",
                linewidth=2,
                zorder=2,
            )
            final_lines.append(
                (
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                    (best_shot.pocket.center[0], best_shot.pocket.center[1]),
                )
            )
            draw_contact_line(
                ax,
                best_shot.white,
                best_shot.target_helper,
                best_shot.target,
                color="blue",
            )
            final_lines.append(
                (
                    (best_shot.white.x_cord, best_shot.white.y_cord),
                    (best_shot.target_helper.x_cord, best_shot.target_helper.y_cord),
                )
            )
            draw_contact_line(
                ax, best_shot.target_helper, best_shot.target, best_shot.pocket
            )
            final_lines.append(
                (
                    (best_shot.target_helper.x_cord, best_shot.target_helper.y_cord),
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                )
            )
        else:
            ax.plot(
                [best_shot.target.x_cord, best_shot.pocket.center[0]],
                [best_shot.target.y_cord, best_shot.pocket.center[1]],
                linestyle="-",
                color="red",
                linewidth=2,
                zorder=2,
            )
            final_lines.append(
                (
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                    (best_shot.pocket.center[0], best_shot.pocket.center[1]),
                )
            )
            draw_contact_line(ax, best_shot.white, best_shot.target, best_shot.pocket)
            final_lines.append(
                (
                    (best_shot.white.x_cord, best_shot.white.y_cord),
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                )
            )
        if isinstance(best_shot, BestWallShot):
            for i, ((x1, y1), (x2, y2)) in enumerate(best_shot.get_lines()):
                color = "blue" if i == 0 else ("orange" if i == 1 else "red")
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linestyle="-",
                    color=color,
                    linewidth=2,
                    zorder=2,
                )
    for pocket in table.pockets:
        pocket_circle = plt.Circle(
            (pocket.center[0], pocket.center[1]), pocket.radius, color="black", zorder=3
        )
        ax.add_patch(pocket_circle)
    for ball in table.balls:
        if ball.type == "white":
            color = "white"
        elif ball.type == "black":
            color = "black"
        elif ball.type == "solid":
            color = "blue"
        elif ball.type == "striped":
            color = "red"
        else:
            color = "gray"
        circle = plt.Circle(
            (ball.x_cord, ball.y_cord), ball.radius, color=color, ec="black", zorder=4
        )
        ax.add_patch(circle)
        ax.text(
            ball.x_cord,
            ball.y_cord,
            str(ball.id),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=5,
        )
    return fig, final_lines


def draw_random_table():
    balls = []
    for i in range(16):
        if i == 0:
            btype = "white"
        elif i == 8:
            btype = "black"
        elif 1 <= i <= 7:
            btype = "solid"
        else:
            btype = "striped"
        x = random.uniform(
            get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
        )
        y = random.uniform(
            get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
        )
        balls.append(Ball(i, x, y, btype, get_ball_radius()))
    table = Table(get_table_length(), get_table_width(), balls)
    draw_table(table)


def draw_random_white_and_black(draw_line_between: bool = False, pocket_id: int = None):
    x_white = random.uniform(
        get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
    )
    y_white = random.uniform(
        get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
    )
    white = Ball(0, x_white, y_white, "white", get_ball_radius())
    x_black = random.uniform(
        get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2
    )
    y_black = random.uniform(
        get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2
    )
    black = Ball(8, x_black, y_black, "black", get_ball_radius())
    table = Table(get_table_length(), get_table_width(), [white, black])
    lines = []
    if draw_line_between:
        lines.append(Line(white, black))
    if pocket_id is not None and 0 <= pocket_id < len(table.pockets):
        lines.append(Line(black, table.pockets[pocket_id]))
    draw_table(table, lines)


def draw_white_center_black_to_corner():
    x_white = get_table_length() / 2
    y_white = get_table_width() / 2
    white = Ball(0, x_white, y_white, "white", get_ball_radius())
    bottom_right_pocket = None
    for p in Table(get_table_length(), get_table_width(), []).pockets:
        if p.center[0] == get_table_length() and p.center[1] == 0:
            bottom_right_pocket = p
            break
    if bottom_right_pocket is None:
        raise ValueError("לא נמצא חור בפינה הימנית תחתונה")
    x_black = (x_white + bottom_right_pocket.center[0]) / 2
    y_black = (y_white + bottom_right_pocket.center[1]) / 2
    black = Ball(8, x_black, y_black, "black", get_ball_radius())
    table = Table(get_table_length(), get_table_width(), [white, black])
    lines = [Line(black, p) for p in table.pockets]
    draw_table(table, lines)
    calc = Calculations(white, black, table.pockets)
    print("זוויות של הכדור השחור לכל החורים:")
    angles = calc.angle_to_pockets()
    print(angles)


def draw_contact_line(ax, white, black, pocket, color="orange"):
    print("Drawing contact line...")
    vx = pocket.center[0] - black.x_cord
    vy = pocket.center[1] - black.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm
    contact_x = black.x_cord - vx * black.radius
    contact_y = black.y_cord - vy * black.radius
    ax.plot(
        [white.x_cord, contact_x],
        [white.y_cord, contact_y],
        linestyle="--",
        color=color,
        linewidth=2,
        zorder=2,
    )
    print("Contact line drawn.[]")
    ax.plot(contact_x, contact_y, "o", color="orange", markersize=6, zorder=6)


def draw_contact_line_B2B(ax, white, target_helper, target, pocket):
    vx = pocket.center[0] - target.x_cord
    vy = pocket.center[1] - target.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm
    contact_x = target.x_cord - vx * target.radius
    contact_y = target.y_cord - vy * target.radius
    ax.plot(
        [target_helper.x_cord, contact_x],
        [target_helper.y_cord, contact_y],
        linestyle="--",
        color="blue",
        linewidth=2,
        zorder=2,
    )
    ax.plot(contact_x, contact_y, "o", color="orange", markersize=6, zorder=6)


def draw_ball_contact_view(white, target, pocket):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("green")
    margin = target.radius * 4
    ax.set_xlim(target.x_cord - margin, target.x_cord + margin)
    ax.set_ylim(target.y_cord - margin, target.y_cord + margin)
    type_to_color = {
        "white": "white",
        "black": "black",
        "solid": "blue",
        "striped": "red",
    }
    face_color = type_to_color.get(target.type, "gray")
    circle_black = plt.Circle(
        (target.x_cord, target.y_cord),
        target.radius,
        color=face_color,
        ec="white",
        zorder=3,
    )
    ax.add_patch(circle_black)
    ax.text(
        target.x_cord,
        target.y_cord,
        str(target.id),
        ha="center",
        va="center",
        fontsize=14,
        color="black" if face_color != "black" else "white",
        zorder=4,
    )
    vx = pocket.center[0] - target.x_cord
    vy = pocket.center[1] - target.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm
    contact_x = target.x_cord - vx * target.radius
    contact_y = target.y_cord - vy * target.radius
    if target.type == "black":
        ax.plot(contact_x, contact_y, "o", color="red", markersize=10, zorder=5)
    else:
        ax.plot(contact_x, contact_y, "o", color="black", markersize=10, zorder=5)
    circle_white = plt.Circle(
        (white.x_cord, white.y_cord), white.radius, color="white", ec="black", zorder=4
    )
    ax.add_patch(circle_white)
    ax.plot(
        [white.x_cord, contact_x],
        [white.y_cord, contact_y],
        linestyle="--",
        color="orange",
        linewidth=2,
        zorder=2,
    )
    plt.title("Zoom on Target Ball with Contact Point")
    return fig
