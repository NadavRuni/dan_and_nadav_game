from game_class.table import Table
from game_class.draw  import draw_table

if __name__ == "__main__":
    # צור שולחן חדש
    table = Table(length=200, width=100)

    # הדפס מידע
    print("📋 Table size:", table.length, "x", table.width)
    print("🕳️ Pockets:", table.pockets)

    print("🎱 Balls on table:")
    for ball in table.balls:
        print("   ", ball)

    # צייר את השולחן עם הכדורים
    draw_table(table)


