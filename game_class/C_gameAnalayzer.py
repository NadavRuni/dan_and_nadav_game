from game_class.C_table import Table
from game_class.C_bestShot import BestShot
import math
from const_numbers import *
from game_class.C_bestShotBallToBall import BestShotBallToBall

from game_class.C_calc_using_wall import CalculationsWithWall
from game_class.C_bestShot_use_wall import BestWallShot


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
                print(
                    "between the white and ball number",
                    ball.id,
                    "dont have a free shot",
                )
                continue

            shot = BestShot(white, ball, table)

            if not shot.valid or shot.score <= 1:
                continue
            all_shots.append(shot)

        if all_shots:
            sorted_shots = sorted(
                [s for s in all_shots if s.score is not None],
                key=lambda s: s.score,
                reverse=True,
            )
            print(
                "✅ All valid shots:",
                ", ".join(
                    [f"Ball {s.target.id} (score={s.score:.2f})" for s in sorted_shots]
                ),
            )
            return sorted_shots[:3]  # שלושת המכות הכי טובות
        else:
            return []  # need to add mor logic here

    def has_clear_path(self, ball1, ball2) -> bool:
        """
        בודקת האם יש מסלול פנוי בין שני כדורים (ball1 → ball2).
        - המסלול הוא מהיקף של ball1 עד היקף של ball2 (לא מרכז-למרכז).
        - אם כדור אחר מתקרב למסלול פחות מ-(other.radius + SAFE_DISTANCE) → חסימה.
        """
        EPS = 1e-6

        ax, ay = ball1.x_cord, ball1.y_cord
        bx, by = ball2.x_cord, ball2.y_cord

        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len < EPS:
            # כדורים כמעט באותו מקום – אין מסלול משמעותי
            return False

        # וקטור יחידה לאורך הקטע
        ux, uy = dx / seg_len, dy / seg_len

        # "קיצור" הקטע: מהיקף של ball1 עד היקף של ball2
        axp = ax + ux * ball1.radius
        ayp = ay + uy * ball1.radius
        bxp = bx - ux * ball2.radius
        byp = by - uy * ball2.radius

        dxp, dyp = bxp - axp, byp - ayp
        seg_len2 = dxp * dxp + dyp * dyp
        if seg_len2 < EPS:
            # אחרי קיצוץ הרדיוסים לא נשאר כמעט אורך מסלול
            return False
        return True

    @staticmethod
    def point_segment_distance(px, py, x1, y1, x2, y2):
        """מרחק מנקודה (px,py) לקטע בין (x1,y1) ל-(x2,y2)"""
        # וקטור מהתחלה לסוף
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)

        # הקרנה של p על הקטע
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # מגבילים לקטע בלבד
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.hypot(px - proj_x, py - proj_y)

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

    def find_best_overall_shot_ball_to_ball(
        self, my_ball_type: str = "all"
    ) -> list[BestShotBallToBall]:
        """
        מחשבת את שלושת המכות הכי טובות על פני כל הכדורים בשולחן.
        מחזירה רשימה ממוינת מהטובה ביותר לפחות.
        אם אין שלוש מכות חוקיות תחזיר כמה שיש.
        """
        table = self.table
        white = next(b for b in table.get_balls() if b.type == "white")
        all_shots: list[BestShot] = []

    def find_best_wall_shots(self, my_ball_type: str = "all") -> list[BestWallShot]:
        """
        מחפש מכות עם קיר (Wall shots) ומחזיר את שלושת הטובות ביותר.
        """
        print("enter to find_best_wall_shots")
        table = self.table
        white = next(b for b in table.get_balls() if b.type == "white")
        wall_shots: list[BestWallShot] = []

        if my_ball_type == "all":
            balls = table.get_balls()
        elif my_ball_type == "solid":
            balls = table.get_solid()
        elif my_ball_type == "striped":
            balls = table.get_striped()
        else:
            balls = table.get_black()

        for helper_ball in balls:
            for target_ball in balls:
                if helper_ball.id == target_ball.id:
                    continue
                if target_ball.type == "white" or helper_ball.type == "white":
                    continue
                if target_ball.type == "black" or helper_ball.type == "black":
                    # אפשר להכניס לוגיקה מתקדמת לפסים/מלאים
                    continue
                if not self.has_clear_path(white, helper_ball):
                    print(
                        "[B2B] between the white and ball number",
                        helper_ball.id,
                        "don't have a free shot",
                    )
                    continue
                print(
                    "[B2B] found a valid shot between the white and ball number",
                    helper_ball.id,
                    "try to B2B with",
                    target_ball.id,
                )
                shot = BestShotBallToBall(white, target_ball, helper_ball, table)
                if not shot.valid:
                    continue
                all_shots.append(shot)

        if all_shots:
            sorted_shots = sorted(
                [s for s in all_shots if s.score is not None],
                key=lambda s: s.score,
                reverse=True,
            )
            print(
                "✅ All valid shots:",
                ", ".join(
                    [f"Ball {s.target.id} (score={s.score:.2f})" for s in sorted_shots]
                ),
            )
            return sorted_shots[:3]  # שלושת המכות הכי טובות
        else:
            print("❌ No valid shots found.")
            return []
        for ball in balls:
            if ball.type == "white":
                continue

            calc = CalculationsWithWall(white, ball, table)
            for pocket in table.get_pockets():
                wall_shot = BestWallShot(calc, pocket)
                if wall_shot.valid:
                    wall_shots.append(wall_shot)

        print("wall_shot")
        print(wall_shot)

        if wall_shots:
            sorted_wall_shots = sorted(
                [s for s in wall_shots if s.score is not None],
                key=lambda s: s.score,
                reverse=True,
            )
            print(
                "✅ All valid wall shots:",
                ", ".join(
                    [
                        f"[WALL] Ball {s.target.id} (score={s.score:.2f})"
                        for s in sorted_wall_shots
                    ]
                ),
            )
            return sorted_wall_shots[:3]

        print("❌ No wall shots found either.")
        return []
