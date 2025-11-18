import matplotlib.pyplot as plt
import random
from const_numbers import *
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_line import Line
from game_class.C_pocket import Pocket
from game_class.C_calc import *
from typing import List, Optional
from game_class.C_bestShot import BestShot  # נניח שזה השם של המחלקה שלך
from game_class.C_bestShotBallToBall import BestShotBallToBall
from game_class.C_bestShot_use_wall import BestWallShot


def draw_table(
    table: Table,
    lines: Optional[List[Line]] = None,
    best_shot: Optional[BestShot | BestShotBallToBall] = None,
):

    fig, ax = plt.subplots(figsize=(10, 5))

    # ציור מסגרת השולחן
    border = plt.Rectangle(
        (0, 0),
        table.length,
        table.width,
        linewidth=15,
        edgecolor="saddlebrown",
        facecolor="none",
    )
    ax.add_patch(border)

    # רקע ירוק
    ax.set_facecolor("green")
    ax.set_xlim(0, table.length)
    ax.set_ylim(0, table.width)
    ax.set_aspect("equal", adjustable="box")

    # ציור קווים רגילים
    final_lines = []  # רשימה שתאחסן את כל הקווים

    # ציור קווים רגילים
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

    # ציור לפי best_shot (כולל wall shot אם קיים)
    if best_shot:
        print (isinstance(best_shot, BestWallShot))
        if not isinstance(best_shot, BestWallShot) or best_shot.angle>90 :
            print("Best shot is BestWallShot draw the line later")
        elif isinstance(best_shot, BestShotBallToBall):
            ax.plot(
                [best_shot.target.x_cord, best_shot.pocket.x_cord],
                [best_shot.target.y_cord, best_shot.pocket.y_cord],
                linestyle="-",
                color="red",
                linewidth=2,
                zorder=2,
            )
            final_lines.append(
                (
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                    (best_shot.pocket.x_cord, best_shot.pocket.y_cord),
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
                [best_shot.target.x_cord, best_shot.pocket.x_cord],
                [best_shot.target.y_cord, best_shot.pocket.y_cord],
                linestyle="-",
                color="red",
                linewidth=2,
                zorder=2,
            )
            final_lines.append(
                (
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                    (best_shot.pocket.x_cord, best_shot.pocket.y_cord),
                )
            )

            draw_contact_line(ax, best_shot.white, best_shot.target, best_shot.pocket)
            final_lines.append(
                (
                    (best_shot.white.x_cord, best_shot.white.y_cord),
                    (best_shot.target.x_cord, best_shot.target.y_cord),
                )
            )
        if (isinstance(best_shot, BestWallShot)):
            for i, ((x1, y1), (x2, y2)) in enumerate(best_shot.get_lines()):
                color = "blue" if i == 0 else ("orange" if i == 1 else "red")
                ax.plot(
                    [x1, x2], [y1, y2], linestyle="-", color=color, linewidth=2, zorder=2
                )

    # ציור חורים
    for pocket in table.pockets:
        pocket_circle = plt.Circle(
            (pocket.x_cord, pocket.y_cord), pocket.radius, color="black", zorder=3
        )
        ax.add_patch(pocket_circle)

    # ציור כדורים
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

    # plt.show()
    return fig, final_lines


def draw_random_table():
    """יוצר שולחן חדש עם כדורים מפוזרים אקראית ומצייר אותו"""
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

        x = random.uniform(get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2)
        y = random.uniform(get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2)

        balls.append(Ball(i, x, y, btype, get_ball_radius()))

    table = Table(get_table_length(), get_table_width(), balls)
    draw_table(table)


def draw_random_white_and_black(draw_line_between: bool = False, pocket_id: int = None):
    """יוצר שולחן עם שני כדורים (לבן ושחור) במיקומים אקראיים ומצייר אותו
    Args:
        draw_line_between (bool): אם True יצויר קו בין הכדור הלבן לכדור השחור
        pocket_id (int|None): אם ניתן, יצויר קו בין הכדור השחור לחור עם ה־id הנתון
    """

    # כדור לבן
    x_white = random.uniform(get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2)
    y_white = random.uniform(get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2)
    white = Ball(0, x_white, y_white, "white", get_ball_radius())

    # כדור שחור
    x_black = random.uniform(get_ball_radius() * 2, get_table_length() - get_ball_radius() * 2)
    y_black = random.uniform(get_ball_radius() * 2, get_table_width() - get_ball_radius() * 2)
    black = Ball(8, x_black, y_black, "black", get_ball_radius())

    # יצירת שולחן
    table = Table(get_table_length(), get_table_width(), [white, black])

    # מערך קווים לציור
    lines = []

    if draw_line_between:
        lines.append(Line(white, black))

    if pocket_id is not None and 0 <= pocket_id < len(table.pockets):
        lines.append(Line(black, table.pockets[pocket_id]))

    # ציור
    draw_table(table, lines)


def draw_white_center_black_to_corner():
    """מצייר שולחן עם כדור לבן במרכז
    וכדור שחור באמצע הדרך לעבר הפינה הימנית־תחתונה
    """

    # מיקום הלבן = מרכז
    x_white = get_table_length() / 2
    y_white = get_table_width() / 2
    white = Ball(0, x_white, y_white, "white", get_ball_radius())

    # חור בפינה הימנית־תחתונה
    bottom_right_pocket = None
    for p in Table(get_table_length(), get_table_width(), []).pockets:
        if p.x_cord == get_table_length() and p.y_cord == 0:  # ימין תחתון
            bottom_right_pocket = p
            break

    if bottom_right_pocket is None:
        raise ValueError("לא נמצא חור בפינה הימנית תחתונה")

    # מיקום השחור = אמצע הדרך בין מרכז→חור ימין תחתון
    x_black = (x_white + bottom_right_pocket.x_cord) / 2
    y_black = (y_white + bottom_right_pocket.y_cord) / 2
    black = Ball(8, x_black, y_black, "black", get_ball_radius())

    # שולחן
    table = Table(get_table_length(), get_table_width(), [white, black])

    # קווים: בין שחור→חור
    lines = [Line(black, p) for p in table.pockets]

    # ציור
    draw_table(table, lines)

    calc = Calculations(white, black, table.pockets)
    print("זוויות של הכדור השחור לכל החורים:")
    angles = calc.angle_to_pockets()
    print(angles)


def draw_contact_line(ax, white, black, pocket, color="orange"):
    print ("Drawing contact line...")
    # וקטור שחור→חור
    vx = pocket.x_cord - black.x_cord
    vy = pocket.y_cord - black.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm

    # נקודת מגע על היקף השחור (בכיוון מהחור לשחור)
    contact_x = black.x_cord - vx * black.radius
    contact_y = black.y_cord - vy * black.radius

    # ציור הקו מלבן → נקודת מגע
    ax.plot(
        [white.x_cord, contact_x],
        [white.y_cord, contact_y],
        linestyle="--",
        color=color,
        linewidth=2,
        zorder=2,
    )
    print("Contact line drawn.[]")
    # ציור נקודת המגע עצמה
    ax.plot(contact_x, contact_y, "o", color="orange", markersize=6, zorder=6)

def draw_contact_line_B2B(ax, white, target_helper, target, pocket):
    # וקטור שחור→חור
    vx = pocket.x_cord - target.x_cord
    vy = pocket.y_cord - target.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm

    # נקודת מגע על היקף השחור (בכיוון מהחור לשחור)
    contact_x = target.x_cord - vx * target.radius
    contact_y = target.y_cord - vy * target.radius

    # ציור הקו מלבן → נקודת מגע
    ax.plot(
        [target_helper.x_cord, contact_x],
        [target_helper.y_cord, contact_y],
        linestyle="--",
        color="blue",
        linewidth=2,
        zorder=2,
    )

    # ציור נקודת המגע עצמה
    ax.plot(contact_x, contact_y, "o", color="orange", markersize=6, zorder=6)


def draw_ball_contact_view(white, target, pocket):
    """
    מצייר גרף חדש שמתרכז רק בכדור השחור,
    ומראה עליו את נקודת הפגיעה הרצויה + הכדור הלבן.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("green")

    # קובעים גבולות - הכדור השחור במרכז והגדלה סביבו
    margin = target.radius * 4
    ax.set_xlim(target.x_cord - margin, target.x_cord + margin)
    ax.set_ylim(target.y_cord - margin, target.y_cord + margin)
    type_to_color = {
        "white": "white",
        "black": "black",
        "solid": "blue",  # תוכל לשנות לפי מה שמתאים לך
        "striped": "red",  # דוגמה
    }
    face_color = type_to_color.get(target.type, "gray")

    # ציור הכדור השחור
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

    # חישוב נקודת המגע (שחור→חור)
    vx = pocket.x_cord - target.x_cord
    vy = pocket.y_cord - target.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm
    contact_x = target.x_cord - vx * target.radius
    contact_y = target.y_cord - vy * target.radius

    # ציור נקודת הפגיעה (אדום קטן)
    if target.type == "black":
        ax.plot(contact_x, contact_y, "o", color="red", markersize=10, zorder=5)
    else:
        ax.plot(contact_x, contact_y, "o", color="black", markersize=10, zorder=5)

    # ציור הכדור הלבן (רק בשביל הכיוון)
    circle_white = plt.Circle(
        (white.x_cord, white.y_cord), white.radius, color="white", ec="black", zorder=4
    )
    ax.add_patch(circle_white)

    # קו מהלבן לנקודת המגע
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
