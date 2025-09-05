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
        print("self.pocket ", self.pocket)
        print("this is beta dict: ", beta_dict)
        self.angle, (dir_x, dir_y) = beta_dict.get(pocket, (0.0, (0.0, 0.0)))

        self.point_with_the_wall = (dir_x, dir_y)
        self.valid = not self.has_obstacle_on_lines()

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
                if dist <= ball.radius + BALL_RADIUS + SAFE_DISTANCE:
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
            (self.pocket.x_cord, self.pocket.y_cord),
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
