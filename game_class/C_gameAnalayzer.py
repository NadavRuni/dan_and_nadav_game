from game_class.C_table import Table
from game_class.C_bestShot import BestShot
import math
from const_numbers import *

class GameAnalayzer:
    def __init__(self, table: Table):
        self.table = table

    def find_best_overall_shot(self, my_ball_type: str = "all") -> list[BestShot]:
        """
        מחשבת את שלושת המכות הכי טובות על פני כל הכדורים בשולחן.
        מחזירה רשימה ממוינת מהטובה ביותר לפחות.
        אם אין שלוש מכות חוקיות תחזיר כמה שיש.
        """
        table = self.table
        white = next(b for b in table.get_balls() if b.type == "white")
        all_shots: list[BestShot] = []

        if my_ball_type == "all":
            balls = table.get_balls()
        elif my_ball_type == "solid":
            balls = table.get_solid()
        elif my_ball_type == "striped":
            balls = table.get_striped()
        else:
            balls = table.get_black()

        for ball in balls:
            if ball.type == "white":
                continue
            if ball.type == "black" and len(table.get_balls()) > 2:
                # אפשר להכניס לוגיקה מתקדמת לפסים/מלאים
                continue
            if not self.has_clear_path(white, ball):
                print("between the white and ball number", ball.id, "dont have a free shot")
                continue

            shot = BestShot(white, ball, table)
            if not shot.valid:
                continue
            all_shots.append(shot)

        if all_shots:
            sorted_shots = sorted(
                [s for s in all_shots if s.score is not None],
                key=lambda s: s.score,
                reverse=True
            )
            print("✅ All valid shots:", ", ".join(
                [f"Ball {s.target.id} (score={s.score:.2f})" for s in sorted_shots]
            ))
            return sorted_shots[:3]  # שלושת המכות הכי טובות
        else:
            print("❌ No valid shots found.")
            return []

    def has_clear_path(self ,ball1, ball2) -> bool:
        """
        בודקת האם יש קו ישר בין ball1 ל-ball2 ללא הפרעה מכדורים אחרים.
        """

        for other in self.table.get_balls():
            if other.id in (ball1.id, ball2.id):
                continue

            dist = GameAnalayzer.point_line_distance(
                other.x_cord, other.y_cord,
                ball1.x_cord, ball1.y_cord,
                ball2.x_cord, ball2.y_cord
            )

            if dist <= other.radius + BALL_RADIUS + SAFE_DISTANCE:
                if GameAnalayzer.is_between(
                    other.x_cord, other.y_cord,
                    ball1.x_cord, ball1.y_cord,
                    ball2.x_cord, ball2.y_cord
                ):
                    return False
        return True

    @staticmethod
    def point_line_distance(px, py, x1, y1, x2, y2) -> float:
        """
        מחשבת את המרחק מנקודה (px,py) לקו המחבר בין (x1,y1) ל-(x2,y2).
        """
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D
        param = dot / len_sq if len_sq != 0 else -1

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = px - xx
        dy = py - yy
        return math.hypot(dx, dy)

    @staticmethod
    def is_between(px, py, x1, y1, x2, y2) -> bool:
        """
        בודקת אם נקודה (px,py) נמצאת בין שתי נקודות (x1,y1) ו-(x2,y2).
        """
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
