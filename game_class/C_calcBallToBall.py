import math
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket
from game_class.C_table import Table


class CalculationsBallToBall:
    def __init__(self, white: Ball, target: Ball, target_helper: Ball, table: Table):
        self.white = white
        self.target = target
        self.target_helper = target_helper
        self.pockets = table.get_pockets()
        self.balls = table.get_balls()
    import math
from typing import Tuple, Optional
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket
from game_class.C_table import Table
from const_numbers import *
# נניח שכאן קיימת המחלקה שביקשת קודם:
# class CalculationsBallToBall: ... (have_free_shot returns True iff only helper blocks)

class CalculationsBallToBall:
    def __init__(self, white: Ball, target: Ball, target_helper: Ball, table: Table):
        self.white = white
        self.target = target
        self.target_helper = target_helper
        self.pockets = table.get_pockets()
        self.balls = table.get_balls()

        
