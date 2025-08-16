import matplotlib.pyplot as plt


def draw_table(table):
    fig, ax = plt.subplots(figsize=(10, 5))

    # ציור שולחן (רקע ירוק)
    ax.set_facecolor("green")
    ax.set_xlim(0, table.length)
    ax.set_ylim(0, table.width)

    # ציור חורים
    pocket_radius = 5
    for (x, y) in table.pockets:
        pocket = plt.Circle((x, y), pocket_radius, color="black")
        ax.add_patch(pocket)

    # ציור כדורים
    ball_radius = 4
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
