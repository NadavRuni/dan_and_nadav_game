from random import random
import matplotlib.pyplot as plt
import random
from const_numbers import *
from game_class.ball import Ball
from game_class.table import Table


def draw_table(table):
    fig, ax = plt.subplots(figsize=(10, 5))
    border = plt.Rectangle(
        (0, 0), table.length, table.width,
        linewidth=15, edgecolor="saddlebrown", facecolor="none"
    )
    ax.add_patch(border)

    # ציור שולחן (רקע ירוק)
    ax.set_facecolor("green")
    ax.set_xlim(0, table.length)
    ax.set_ylim(0, table.width)

    # ציור חורים
    pocket_radius = POCKET_RADIUS
    for (x, y) in table.pockets:
        pocket = plt.Circle((x, y), pocket_radius, color="black")
        ax.add_patch(pocket)

    # ציור כדורים
    ball_radius = BALL_RADIUS
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

        circle = plt.Circle((ball.x_cord, ball.y_cord), ball_radius, color=color, ec="black")
        ax.add_patch(circle)
        ax.text(ball.x_cord, ball.y_cord, str(ball.id), ha="center", va="center", fontsize=8, color="black")

    # הורדת צירים
    ax.axis("off")
    plt.show()
def draw_random_table():
    """יוצר שולחן חדש עם כדורים מפוזרים אקראית ומצייר אותו"""

    balls = []

    # כל הכדורים 1–15 + לבן (0) ושחור (8)
    for i in range(16):
        if i == 0:
            btype = "white"
        elif i == 8:
            btype = "black"
        elif 1 <= i <= 7:
            btype = "solid"
        else:
            btype = "striped"

        # מיקום אקראי בתוך השולחן
        x = random.uniform(POCKET_RADIUS + BALL_RADIUS, TABLE_LENGTH - POCKET_RADIUS - BALL_RADIUS)
        y = random.uniform(POCKET_RADIUS + BALL_RADIUS, TABLE_WIDTH - POCKET_RADIUS - BALL_RADIUS)

        balls.append(Ball(i, x, y, btype, BALL_RADIUS))

    # צור שולחן
    table = Table(TABLE_LENGTH, TABLE_WIDTH, balls)

    # צייר
    draw_table(table)


def draw_random_white_and_black():
    """יוצר שולחן עם שני כדורים (לבן ושחור) במיקומים אקראיים ומצייר אותו"""

    balls = []

    # כדור לבן
    x_white = random.uniform(POCKET_RADIUS + BALL_RADIUS, TABLE_LENGTH - POCKET_RADIUS - BALL_RADIUS)
    y_white = random.uniform(POCKET_RADIUS + BALL_RADIUS, TABLE_WIDTH - POCKET_RADIUS - BALL_RADIUS)
    white = Ball(0, x_white, y_white, "white", BALL_RADIUS)
    balls.append(white)

    # כדור שחור
    x_black = random.uniform(POCKET_RADIUS + BALL_RADIUS, TABLE_LENGTH - POCKET_RADIUS - BALL_RADIUS)
    y_black = random.uniform(POCKET_RADIUS + BALL_RADIUS, TABLE_WIDTH - POCKET_RADIUS - BALL_RADIUS)
    black = Ball(8, x_black, y_black, "black", BALL_RADIUS)
    balls.append(black)

    # יצירת שולחן עם שני הכדורים
    table = Table(TABLE_LENGTH, TABLE_WIDTH, balls)

    # ציור השולחן
    draw_table(table)