import matplotlib.pyplot as plt
import random
from const_numbers import *
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_line import Line
from game_class.C_pocket import Pocket
from game_class.C_calc import *
from typing import List, Optional
from game_class.C_bestShot  import BestShot  # נניח שזה השם של המחלקה שלך

def draw_table(table: Table, lines: Optional[List[Line]] = None, best_shot: Optional[BestShot] = None):
    fig, ax = plt.subplots(figsize=(10, 5))

    # ציור מסגרת השולחן
    border = plt.Rectangle(
        (0, 0), table.length, table.width,
        linewidth=15, edgecolor="saddlebrown", facecolor="none"
    )
    ax.add_patch(border)

    # רקע ירוק
    ax.set_facecolor("green")
    ax.set_xlim(0, table.length)
    ax.set_ylim(0, table.width)
    ax.set_aspect("equal", adjustable="box")

    # ציור קווים רגילים
    if lines:
        for line in lines:
            (x1, y1), (x2, y2) = line.as_tuple()
            ax.plot([x1, x2], [y1, y2],
                    linestyle="--", color="black", linewidth=1.5, zorder=2)

    # ציור לפי best_shot (כולל wall shot אם קיים)
    if best_shot:
        for i, ((x1, y1), (x2, y2)) in enumerate(best_shot.get_lines()):
            color = "blue" if i == 0 else ("orange" if i == 1 else "red")
            ax.plot([x1, x2], [y1, y2],
                    linestyle="-", color=color, linewidth=2, zorder=2)

    # ציור חורים
    for pocket in table.pockets:
        pocket_circle = plt.Circle(
            (pocket.x_cord, pocket.y_cord),
            pocket.radius,
            color="black",
            zorder=3
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
            (ball.x_cord, ball.y_cord),
            ball.radius,
            color=color,
            ec="black",
            zorder=4
        )
        ax.add_patch(circle)
        ax.text(ball.x_cord, ball.y_cord, str(ball.id),
                ha="center", va="center",
                fontsize=8, color="black", zorder=5)

    plt.show()



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

        x = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
        y = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)

        balls.append(Ball(i, x, y, btype, BALL_RADIUS))

    table = Table(TABLE_LENGTH, TABLE_WIDTH, balls)
    draw_table(table)

def draw_random_white_and_black(draw_line_between: bool = False, pocket_id: int = None):
    """יוצר שולחן עם שני כדורים (לבן ושחור) במיקומים אקראיים ומצייר אותו
    Args:
        draw_line_between (bool): אם True יצויר קו בין הכדור הלבן לכדור השחור
        pocket_id (int|None): אם ניתן, יצויר קו בין הכדור השחור לחור עם ה־id הנתון
    """

    # כדור לבן
    x_white = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
    y_white = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)
    white = Ball(0, x_white, y_white, "white", BALL_RADIUS)

    # כדור שחור
    x_black = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
    y_black = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)
    black = Ball(8, x_black, y_black, "black", BALL_RADIUS)

    # יצירת שולחן
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black])

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
    x_white = TABLE_LENGTH / 2
    y_white = TABLE_WIDTH / 2
    white = Ball(0, x_white, y_white, "white", BALL_RADIUS)

    # חור בפינה הימנית־תחתונה
    bottom_right_pocket = None
    for p in Table(TABLE_LENGTH, TABLE_WIDTH, []).pockets:
        if p.x_cord == TABLE_LENGTH and p.y_cord == 0:  # ימין תחתון
            bottom_right_pocket = p
            break

    if bottom_right_pocket is None:
        raise ValueError("לא נמצא חור בפינה הימנית תחתונה")

    # מיקום השחור = אמצע הדרך בין מרכז→חור ימין תחתון
    x_black = (x_white + bottom_right_pocket.x_cord) / 2
    y_black = (y_white + bottom_right_pocket.y_cord) / 2
    black = Ball(8, x_black, y_black, "black", BALL_RADIUS)

    # שולחן
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black])

    # קווים: בין שחור→חור
    lines = [Line(black, p) for p in table.pockets]

    # ציור
    draw_table(table, lines)

    calc = Calculations(white, black, table.pockets)
    print("זוויות של הכדור השחור לכל החורים:")
    angles = calc.angle_to_pockets()
    print(angles)

def draw_contact_line(ax, white, black, pocket):
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
        linestyle="--", color="orange", linewidth=2, zorder=2
    )

    # ציור נקודת המגע עצמה
    ax.plot(contact_x, contact_y, "o", color="orange", markersize=6, zorder=6)

def draw_ball_contact_view(white, black, pocket):
    """
    מצייר גרף חדש שמתרכז רק בכדור השחור,
    ומראה עליו את נקודת הפגיעה הרצויה + הכדור הלבן.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("green")

    # קובעים גבולות - הכדור השחור במרכז והגדלה סביבו
    margin = black.radius * 4
    ax.set_xlim(black.x_cord - margin, black.x_cord + margin)
    ax.set_ylim(black.y_cord - margin, black.y_cord + margin)

    # ציור הכדור השחור
    circle_black = plt.Circle(
        (black.x_cord, black.y_cord),
        black.radius,
        color="black",
        ec="white",
        zorder=3
    )
    ax.add_patch(circle_black)

    # חישוב נקודת המגע (שחור→חור)
    vx = pocket.x_cord - black.x_cord
    vy = pocket.y_cord - black.y_cord
    norm = math.hypot(vx, vy)
    vx /= norm
    vy /= norm
    contact_x = black.x_cord - vx * black.radius
    contact_y = black.y_cord - vy * black.radius

    # ציור נקודת הפגיעה (אדום קטן)
    ax.plot(contact_x, contact_y, "o", color="red", markersize=10, zorder=5)

    # ציור הכדור הלבן (רק בשביל הכיוון)
    circle_white = plt.Circle(
        (white.x_cord, white.y_cord),
        white.radius,
        color="white",
        ec="black",
        zorder=4
    )
    ax.add_patch(circle_white)

    # קו מהלבן לנקודת המגע
    ax.plot(
        [white.x_cord, contact_x],
        [white.y_cord, contact_y],
        linestyle="--", color="orange", linewidth=2, zorder=2
    )

    plt.title("Zoom on Target Ball with Contact Point")
    return fig, ax

