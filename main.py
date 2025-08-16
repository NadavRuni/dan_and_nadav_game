from game_class.C_table import Table
from game_class.C_draw  import *
from const_numbers import *
from game_class.C_ball import *
from game_class.C_calc import *
from game_class.C_bestShot import *
from dan.build_table_from_image import *



def main():
    # יצירת כדורים
    x_white, y_white = TABLE_LENGTH / 2, TABLE_WIDTH / 2
    white = Ball(0, x_white, y_white, "white", BALL_RADIUS)

    x_black = random.uniform(BALL_RADIUS * 2, TABLE_LENGTH - BALL_RADIUS * 2)
    y_black = random.uniform(BALL_RADIUS * 2, TABLE_WIDTH - BALL_RADIUS * 2)
    black = Ball(8, x_black, y_black, "black", BALL_RADIUS)

    # יצירת שולחן עם שני כדורים
    table = Table(TABLE_LENGTH, TABLE_WIDTH, [white, black])

    bestShot = BestShot(white, black, table)


    print (bestShot)




    # ציור
    draw_table(table , best_shot=bestShot)

def dan():
    tbl = build_table_from_image(IMAGE_PATH)
    draw_table(tbl)


if __name__ == "__main__":
   # main()
    dan()
