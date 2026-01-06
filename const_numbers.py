"""
Global Configuration and Application State Management.

This module serves a dual purpose:
1.  It defines file system paths and geometric constants for the pool table
    analysis.
2.  It manages the mutable state of the application using global variables and
    accessor functions.

Warning:
    The use of global variables for application state is a significant
    architectural flaw. It makes the system difficult to test, debug, and scale,
    and it is not safe for concurrent requests. A future refactor should
    encapsulate this state within a dedicated context or state management class
    that is passed explicitly through the application.

"""

import math
from pathlib import Path
from typing import List, Optional

from analyzer_table.launcher_helper.json_models import Rectangle, BallType
from game_class.C_pocket import Pocket

# --- Mutable Global State ---
# These variables store the application's state and are modified during runtime.
# This is a temporary and unsafe design.

_TABLE_LENGTH_CM: float = 290.0
_TABLE_WIDTH_CM: float = 145.0
_PLAYER_BALL_TYPE: str = BallType.SOLID
_CROPPED_IMAGE_PATH: str = ""
_CROPPED_RECTANGLE: Optional[Rectangle] = None
_USE_PREDICTED_POCKETS: bool = False
_DETECTED_POCKETS: Optional[List[Pocket]] = None
_IMAGE_WIDTH_PX: float = 0.0
_IMAGE_HEIGHT_PX: float = 0.0

# --- File System Paths and Constants ---

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

RECTANGLE_JSON_PATH = "rectangles_cache.json"
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "img.png"
OUTPUT_JSON_PATH = OUTPUT_DIR / "analysis.json"
OUTPUT_CONTACT_VIEW_PATH = BASE_DIR / "img_contact.png"
FELT_MASK_PATH = "output/debug/black_white_detect/01_felt_mask.jpg"

MERGE_MAX_Y_DIFF: int = 65  # Allowed pixel distance on Y-axis for merging
NOT_FREE_SHOT: str = "dont have a free shot"
FORSE_WALL_SHOT: bool = False  # Debug flag to force wall shot calculations

# --- State Accessor Functions ---


def get_table_length() -> float:
    """Returns the current table length in centimeters."""
    global _TABLE_LENGTH_CM
    return _TABLE_LENGTH_CM


def set_table_length(value: float) -> None:
    """
    Updates the table length.

    Args:
        value: The new table length in centimeters.

    Raises:
        ValueError: If the value is not a positive number.
    """
    global _TABLE_LENGTH_CM
    if value <= 0:
        raise ValueError("Table length must be a positive number.")
    _TABLE_LENGTH_CM = value
    print(f"✅ Updated TABLE_LENGTH = {_TABLE_LENGTH_CM}")


def get_table_width() -> float:
    """Returns the current table width in centimeters."""
    global _TABLE_WIDTH_CM
    return _TABLE_WIDTH_CM


def set_table_width(value: float) -> None:
    """
    Updates the table width.

    Args:
        value: The new table width in centimeters.

    Raises:
        ValueError: If the value is not a positive number.
    """
    global _TABLE_WIDTH_CM
    if value <= 0:
        raise ValueError("Table width must be a positive number.")
    _TABLE_WIDTH_CM = value
    print(f"✅ Updated TABLE_WIDTH = {_TABLE_WIDTH_CM}")


def get_ball_type() -> str:
    """Returns the player's currently selected ball type ('SOLID' or 'STRIPED')."""
    return _PLAYER_BALL_TYPE


def set_ball_type(value: str) -> None:
    """
    Sets the player's ball type.

    Args:
        value: The ball type string, accepts 'solids' or 'stripes' as aliases.

    Raises:
        ValueError: If the provided ball type is invalid.
    """
    global _PLAYER_BALL_TYPE
    if value == "solids":
        value = BallType.SOLID
    elif value == "stripes":
        value = BallType.STRIPED

    if value not in {BallType.SOLID, BallType.STRIPED}:
        raise ValueError(f"Invalid ball type: '{value}'")
    _PLAYER_BALL_TYPE = value


def get_pocket_path() -> str:
    """Returns the file path to the cropped image used for pocket analysis."""
    return _CROPPED_IMAGE_PATH


def set_pocket_path(value: str) -> None:
    """Sets the file path for the cropped image used for pocket analysis."""
    global _CROPPED_IMAGE_PATH
    print("✅ Setting pocket path to:", value)
    _CROPPED_IMAGE_PATH = value


def get_rectangle_croped() -> Optional[Rectangle]:
    """Returns the Rectangle object used for cropping."""
    return _CROPPED_RECTANGLE


def set_rectangle_croped(value: Rectangle) -> None:
    """Sets the Rectangle object used for cropping."""
    global _CROPPED_RECTANGLE
    _CROPPED_RECTANGLE = value


def get_use_predicted_pockets() -> bool:
    """Returns True if user-predicted pockets should be used."""
    return _USE_PREDICTED_POCKETS


def set_use_predicted_pockets(value: bool) -> None:
    """Sets the flag to use user-predicted pockets."""
    global _USE_PREDICTED_POCKETS
    _USE_PREDICTED_POCKETS = value


def get_detected_pockets() -> Optional[List[Pocket]]:
    """Returns the list of detected Pocket objects."""
    return _DETECTED_POCKETS


def set_detected_pockets(value: List[Pocket]) -> None:
    """Sets the list of detected Pocket objects."""
    global _DETECTED_POCKETS
    print("✅ Setting detected pockets:", value)
    _DETECTED_POCKETS = value


def get_width_px() -> float:
    """Returns the width of the cropped image in pixels."""
    return _IMAGE_WIDTH_PX


def set_width_px(value: float) -> None:
    """Sets the width of the cropped image in pixels."""
    global _IMAGE_WIDTH_PX
    _IMAGE_WIDTH_PX = value


def get_height_px() -> float:
    """Returns the height of the cropped image in pixels."""
    return _IMAGE_HEIGHT_PX


def set_height_px(value: float) -> None:
    """Sets the height of the cropped image in pixels."""
    global _IMAGE_HEIGHT_PX
    _IMAGE_HEIGHT_PX = value


# --- Derived Geometric Calculations ---
# These functions calculate geometric properties based on the current table size.


def get_ball_radius() -> float:
    """Calculates the ball radius based on table length."""
    return get_table_length() / 100.0


def get_ball_radius_photo() -> float:
    """Calculates a larger ball radius for visualization purposes."""
    return get_table_length() / 60.0


def get_pocket_margin() -> float:
    """Calculates the margin around pockets for detection."""
    return get_table_length() / 45.0


def get_pocket_margin_merge() -> float:
    """Calculates a larger pocket margin used for merging pocket detections."""
    return get_table_length() / 30.0


def get_wall_margin() -> float:
    """Calculates the margin from the table walls."""
    return get_table_length() / 37.0


def get_ball_diameter() -> float:
    """Calculates the effective ball diameter, including a small buffer."""
    return get_ball_radius() * 2.2


def get_corner_pocket_radius() -> float:
    """Calculates the radius for corner pockets."""
    return get_table_length() / 72.5


def get_side_pocket_radius() -> float:
    """Calculates the radius for side pockets."""
    return get_table_length() / 64.4


def get_min_distance_from_pocket() -> float:
    """
    Calculates the minimum safe distance from a pocket's edge for a shot to
    be considered valid.
    """
    return get_ball_radius() * 1.2


def get_safe_distance() -> float:
    """Calculates the safe distance buffer between two balls."""
    return get_ball_radius() * 0.5


def get_pocket_radius() -> float:
    """Calculates a general pocket radius."""
    return get_table_length() / 50.0


def get_pocket_up_radius() -> float:
    """Calculates a larger pocket radius for upward angle shots."""
    return get_pocket_radius() * 1.5


def get_pocket_down_radius() -> float:
    """Calculates a smaller pocket radius for downward angle shots."""
    return get_pocket_radius() * 0.5


def get_pocket_radius_determinate() -> float:
    """Calculates a smaller, more precise radius for pocket determination."""
    return get_pocket_radius() * 0.5


def get_ball_radius_determinate() -> float:
    """Calculates a smaller, more precise radius for ball determination."""
    return get_ball_radius() * 0.3


def get_max_white_to_target_distance() -> float:
    """
    Returns the maximum allowed distance between the cue ball and a target ball.
    """
    return get_table_length() / 2.0


def get_get_max_white_to_target_distance() -> float:
    """
    Returns the maximum allowed distance between a target ball and a pocket.
    This is roughly half the diagonal of the table.
    """
    return math.hypot(get_table_length(), get_table_width()) / 2.0


def get_crop_half_size() -> int:
    """
    Calculates half the size of the cropped area for creating ball image samples.
    """
    return int(get_table_length() / 9.5)


def get_safe_from_wall() -> float:
    """Calculates the safe distance a ball must be from a wall."""
    return get_ball_radius() * 1.4


def get_merge_max_diff() -> int:
    """
    Returns the maximum allowed pixel distance on the X-axis for merging
    overlapping ball detections.
    """
    return int(get_table_length() / 20)


def get_merge_overlap_margin() -> float:
    """Returns the pixel margin for considering if two ball detections overlap."""
    return get_ball_radius() * 0.2


# --- Coordinate Conversion and Utility Functions ---


def clamp_to_table_pocket(coordinate: float, max_dimension: float) -> float:
    """
    Clamps a coordinate to the table boundaries, respecting the ball radius.

    Args:
        coordinate: The X or Y coordinate to clamp.
        max_dimension: The maximum dimension (length or width) of the table.

    Returns:
        The clamped coordinate.
    """
    ball_rad = get_ball_radius()
    return max(ball_rad, min(max_dimension - ball_rad, coordinate))


def convert_detected_pockets_to_game_coordinates() -> None:
    """
    Converts the coordinates of detected pockets from pixel space to game space.

    This function transforms pocket centers from image pixels (with Y=0 at the top)
    to game centimeters (with Y=0 at the bottom) and updates the global list of
    detected pockets. It relies on the global state for pixel dimensions and the
    list of pockets.
    """
    global _DETECTED_POCKETS
    if not _DETECTED_POCKETS:
        return

    pockets_after_conversion: List[Pocket] = []
    scale_x = get_table_length() / max(1.0, get_width_px())
    scale_y = get_table_width() / max(1.0, get_height_px())

    for pocket in _DETECTED_POCKETS:
        game_x = clamp_to_table_pocket(pocket.center[0] * scale_x, get_table_length())
        game_y = clamp_to_table_pocket(
            (get_height_px() - pocket.center[1]) * scale_y, get_table_width()
        )
        pockets_after_conversion.append(
            Pocket(
                id=pocket.id,
                center=(game_x, game_y),
                radius=pocket.radius,
                location=pocket.location or "",
                pocket_img_cordinates_on_table=pocket.pocket_img_cordinates_on_table,
                pocket_img_path=pocket.pocket_img_path or "",
            )
        )
    set_detected_pockets(pockets_after_conversion)
