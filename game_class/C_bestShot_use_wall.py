import math
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_pocket import Pocket
from const_numbers import *
from .C_bestShot import BestShot
from .C_calc_using_wall import CalculationsWithWall


class BestWallShot(BestShot):
    def __init__(self, calc: CalculationsWithWall, pocket: Pocket):
        """
        BestShot with wall reflection.

        Args:
            calc (CalculationsWithWall): calculation object (holds white, target, table)
            pocket (Pocket): chosen pocket
        """
        super().__init__(calc.white, calc.target, calc.table)
        self.calc = calc
        self.pocket = pocket

        # β = from wall calculation
        beta_dict = calc.angle_to_pockets_use_wall()
        _, (dir_x, dir_y) = beta_dict.get(pocket.id, (0.0, (0.0, 0.0)))

        self.point_with_the_wall = (dir_x, dir_y)
        self.angle = self.angle_white_ball_wall(
            self.white, self.target, self.point_with_the_wall
        )
        self.valid = not self.has_obstacle_on_lines() and self.angle < 75
        if self.valid:
            self.score = self.score_shot()

    @staticmethod
    def point_segment_distance(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)

        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # clamp to segment
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def angle_white_ball_wall(
        self, white: Ball, target: Ball, center_wall: tuple[float, float]
    ) -> float:
        """
        מחזיר מילון של {pocket_id: angle} שבו הזווית היא ההפרש
        בין הכיוון לבן→מטרה (נחשב כ-0°) לבין הכיוון מטרה→כיס.
        """
        # the center_wall is the impact point on the wall
        # it need to be flip for this calculation

        # וקטור לבן→מטרה
        v1x = target.x_cord - white.x_cord
        v1y = target.y_cord - white.y_cord

        v2x = center_wall[0] - target.x_cord
        v2y = center_wall[1] - target.y_cord

        # dot & cross
        dot = v1x * v2x + v1y * v2y
        cross = v1x * v2y - v1y * v2x

        # זווית ברדיאנים
        angle_rad = math.atan2(cross, dot)

        # המרה למעלות
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def has_obstacle_on_lines(self) -> bool:
        """
        Check if there are any balls (excluding white & target)
        that block the path along the 3 lines, considering radius.
        """
        lines = self.get_lines()
        for ball in self.table.get_balls():
            if ball.id in (self.white.id, self.target.id):
                continue

            for (x1, y1), (x2, y2) in lines:
                dist = self.point_segment_distance(
                    ball.x_cord, ball.y_cord, x1, y1, x2, y2
                )
                if dist <= ball.radius + get_ball_radius() + get_safe_distance():
                    return True
        return False

    def get_lines(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Returns 3 lines:
          1. white → target
          2. target → wall point
          3. wall point → pocket
        """
        # if not self.valid or self.pocket is None:
        #     return []

        line_white_to_target = (
            (self.white.x_cord, self.white.y_cord),
            (self.target.x_cord, self.target.y_cord),
        )

        line_target_to_wall = (
            (self.target.x_cord, self.target.y_cord),
            self.point_with_the_wall,
        )

        line_wall_to_pocket = (
            self.point_with_the_wall,
            (self.pocket.center[0], self.pocket.center[1]),
        )

        return [line_white_to_target, line_target_to_wall, line_wall_to_pocket]

    def __repr__(self):
        base = super().__repr__()
        return (
            base + f" [WALL SHOT] "
            f"angle={self.angle:.1f}°, "
            f"pocket.id={self.pocket.id}, "
            f"impact_point={self.point_with_the_wall}"
        )

    import math

    def score_shot(self) -> float:
        """
        מחשב ציון בין -1 ל-50 עבור מכה:
        - זווית קטנה בין קו 1 ל-2 = יותר טוב
        - מרחק כולל קצר = יותר טוב
        """

        lines = self.get_lines()
        if len(lines) != 3:
            return -1

        def vector(p1, p2):
            return (p2[0] - p1[0], p2[1] - p1[1])

        def length(v):
            return math.hypot(v[0], v[1])

        def angle_between(v1, v2):
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = length(v1)
            norm2 = length(v2)
            if norm1 == 0 or norm2 == 0:
                return 180
            cos_theta = max(-1, min(1, dot / (norm1 * norm2)))
            return math.degrees(math.acos(cos_theta))

        # וקטורים לכל קו
        v1 = vector(*lines[0])  # white -> target
        v2 = vector(*lines[1])  # target -> wall
        v3 = vector(*lines[2])  # wall -> pocket

        # זווית בין הקו הראשון לשני
        angle1 = angle_between(v1, v2)

        # אם המכה לא הגיונית (זווית קיצונית)
        if angle1 > 170:  # כמעט ישר לקיר
            return -1

        # מרחק כולל
        dist_total = sum(length(vector(*line)) for line in lines)

        # ----------------
        # ניקוד לפי זווית
        # ----------------
        # 0° = הכי טוב, 170° = גרוע
        angle_score = max(0, (170 - angle1) / 170)  # מנורמל ל־0–1

        # ----------------
        # ניקוד לפי מרחק
        # ----------------
        # מניחים שמכה רגילה לא תהיה מעל 300 יחידות
        max_reasonable_dist = 300
        dist_score = max(0, 1 - (dist_total / max_reasonable_dist))  # קצר = יותר טוב

        # ----------------
        # משקלול
        # ----------------
        final_score = 0.7 * angle_score + 0.3 * dist_score  # בין 0 ל-1

        # מיפוי ל־[1, 50]
        score = 1 + final_score * 49
        return round(score, 2)
