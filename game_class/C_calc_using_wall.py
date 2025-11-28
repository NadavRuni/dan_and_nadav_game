import math
from typing import List
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket
from game_class.C_table import Table
from const_numbers import *
from game_class.C_calc import Calculations


class CalculationsWithWall(Calculations):
    def __init__(self, white: Ball, target: Ball, table: Table):
        super().__init__(white, target, table)
        self.distance_from_wall_dict = self.calculate_distance_from_walls()

    def calculate_distance_from_walls(self):
        """
        מחזיר מילון של מרחקים מהמרכז של הכדור לכל קיר:
        {"left": d1, "right": d2, "down": d3, "up": d4}
        """
        print("for ")
        return {
            "left": self.target.x_cord,  # ממרכז הכדור לקיר שמאלי (x=0)
            "right": self.table.length - self.target.x_cord,  # מהמרכז לקיר ימני
            "down": self.target.y_cord,  # מהמרכז לקיר תחתון (y=0)
            "up": self.table.width - self.target.y_cord,  # מהמרכז לקיר עליון
        }

    def angle_to_pockets_use_wall(self) -> dict[Pocket, float]:
        angle_using_wall_dict = {}
        for pocket in self.pockets:
            if pocket.id == 0 or pocket.id == 1 or pocket.id == 2 or pocket.id == 3:
                angle_using_wall_dict[pocket] = self.wall_shot_angle_to_pocket(pocket)
        return angle_using_wall_dict

    import math

    def wall_shot_angle_to_pocket(self, pocket: Pocket):
        """
        מחשבת את זווית הירייה כדי לפגוע בקיר העליון ואז בכיס.

        Args:
            ball_x, ball_y (float): מיקום הכדור
            pocket_x, pocket_y (float): מיקום הכיס
            get_table_width() (float): רוחב השולחן (y המקסימלי, כלומר מיקום הקיר העליון)

        Returns:
            angle_deg (float): הזווית במעלות (ביחס לציר ה־X)
            (impact_x, impact_y): נקודת הפגיעה בקיר
        """
        pocket_y = pocket.center[1]

        # 1. שיקוף הכיס ביחס לקיר העליון
        mirrored_pocket_y = 2 * get_table_width() - pocket_y

        match pocket.id:
            case 0:
                Q = mirrored_pocket_y - self.distance_from_wall_dict["down"]
                direction_from_side_wall = "left"
            case 1:
                Q = mirrored_pocket_y - self.distance_from_wall_dict["down"]
                direction_from_side_wall = "right"
            case 2:
                Q = self.distance_from_wall_dict["down"] + get_table_width()
                direction_from_side_wall = "right"
            case 3:
                Q = self.distance_from_wall_dict["down"] + get_table_width()
                direction_from_side_wall = "left"

        P = self.distance_from_wall_dict[direction_from_side_wall]
        theta_rad = math.atan(Q / P)  # זווית ברדיאנים
        theta_deg = abs(math.degrees(theta_rad))  # זווית במעלות, תמיד 0–90

        match pocket.id:
            # דמיון משולשים
            # sorry fot this :)

            case 0:
                impact_x = (get_table_width() * P) / Q
                impact_y = get_table_width()
                theta_deg = 180 - theta_deg
            case 1:
                impact_x = get_table_length() - ((get_table_width() * P) / Q)
                impact_y = get_table_width()
            case 2:
                impact_x = get_table_length() - ((get_table_width() * P) / Q)
                impact_y = 0
            case 3:
                impact_x = (get_table_width() * P) / Q
                impact_y = 0
                theta_deg = 180 - theta_deg

        return theta_deg, (impact_x, impact_y)

    def wall_shot_angle_to_pocket_1(self, pocket: Pocket):
        """
        מחשבת את זווית הירייה כדי לפגוע בקיר העליון ואז בכיס.

        Args:
            ball_x, ball_y (float): מיקום הכדור
            pocket_x, pocket_y (float): מיקום הכיס
            get_table_width() (float): רוחב השולחן (y המקסימלי, כלומר מיקום הקיר העליון)

        Returns:
            angle_deg (float): הזווית במעלות (ביחס לציר ה־X)
            (impact_x, impact_y): נקודת הפגיעה בקיר
        """
        ball_x = self.target.x_cord
        ball_y = self.target.y_cord
        pocket_x = pocket.center[0]
        pocket_y = pocket.center[1]

        # 1. שיקוף הכיס ביחס לקיר העליון
        mirrored_pocket_y = 2 * get_table_width() - pocket_y

        Q = mirrored_pocket_y - self.distance_from_wall_dict["down"]
        P = self.distance_from_wall_dict["right"]

        if P == 0:  # אנך
            return 90.0

        theta_rad = math.atan(Q / P)  # זווית ברדיאנים
        theta_deg = abs(math.degrees(theta_rad))  # זווית במעלות, תמיד 0–90

        # דמיון משולשים
        # sorry fot this :)

        impact_x = (get_table_width() * P) / Q

        return theta_deg, (impact_x, get_table_width())

    def min_abs_angle(self) -> tuple[int, float]:
        """
        מחזירה את החור עם הזווית הקטנה ביותר בערך מוחלט.
        אם אין חור חוקי → מחזירה NOT_FREE_SHOT.
        """
        angles = self.angle_to_pockets()

        # סינון ערכים שהם מספרים בלבד
        valid_angles = {
            pid: ang for pid, ang in angles.items() if isinstance(ang, (int, float))
        }

        if not valid_angles:
            return NOT_FREE_SHOT, NOT_FREE_SHOT

        return min(valid_angles.items(), key=lambda kv: abs(kv[1]))
