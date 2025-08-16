from game_class.table import Table
from game_class.draw  import *
from const_numbers import *


if __name__ == "__main__":
    # # צור שולחן חדש
    # table = Table(length=TABLE_LENGTH, width=TABLE_WIDTH)
    #
    # # הדפס מידע
    # print("📋 Table size:", table.length, "x", table.width)
    # print("🕳️ Pockets:", table.pockets)
    #
    # print("🎱 Balls on table:")
    # for ball in table.balls:
    #     print("   ", ball)
    #
    # # צייר את השולחן עם הכדורים
    # draw_table(table)
    #draw_random_table()

    draw_random_white_and_black()