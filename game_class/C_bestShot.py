import math
from game_class.C_table import Table
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket
from game_class.C_calc import Calculations


class BestShot:
    def __init__(self, white: Ball, target: Ball, table: Table):
        self.white = white
        self.target = target
        self.table = table

        # השתמש במחלקת Calculations כדי למצוא את הזווית הכי טובה
        calc = Calculations(white, target, table.pockets)
        best_pocket_id, best_angle = calc.min_abs_angle()

        # שמירת הנתונים
        self.pocket: Pocket = next(p for p in table.pockets if p.id == best_pocket_id)
        self.angle: float = best_angle

        # מרחקים
        self.dist_target_to_pocket = math.hypot(
            self.pocket.x_cord - target.x_cord,
            self.pocket.y_cord - target.y_cord
        )
        self.dist_white_to_target = math.hypot(
            target.x_cord - white.x_cord,
            target.y_cord - white.y_cord
        )

        # ציון סופי – יחושב בהמשך
        self.score: float | None = None

    def get_pocket(self) -> int:
        """מחזירה את ה־ID של הכיס שנבחר"""
        return self.pocket.id

    def get_pocket_and_angle(self) -> tuple[int, float]:
        """מחזירה tuple עם (pocket_id, angle)"""
        return self.pocket.id, self.angle

    def __repr__(self):
        return (f"BestShot(target_id={self.target.id}, "
                f"pocket_id={self.pocket.id}, "
                f"angle={self.angle:.2f}, "
                f"dist_white_target={self.dist_white_to_target:.2f}, "
                f"dist_target_pocket={self.dist_target_to_pocket:.2f}, "
                f"score={self.score})")
