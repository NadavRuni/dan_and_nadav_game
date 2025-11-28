import math
from pathlib import Path
from analyzer_table.launcher_helper.json_models import Rectangle, Ball_Color
from game_class.C_pocket import Pocket
from typing import List

# table_config.py

TABLE_LENGTH = 290
TABLE_WIDTH = 145
BALL_TYPE = "solid"
POCKET_PATH = ""
RECTANGLE_CROPED = None
USE_PREDICTED_POCKETS = False
DETECTED_POCKETS: List[Pocket] | None = None
WIDTH_PX = 0
HEIGHT_PX = 0



# -------------------------------
# 📏 TABLE LENGTH
# -------------------------------
def get_table_length() -> int | float:
    """מחזיר את אורך השולחן הנוכחי."""
    global TABLE_LENGTH
    return TABLE_LENGTH


def set_table_length(value: int | float) -> None:
    """מעדכן את אורך השולחן, בתנאי שהערך חיובי."""
    global TABLE_LENGTH
    if value <= 0:
        raise ValueError("TABLE_LENGTH must be a positive number.")
    TABLE_LENGTH = value
    print(f"✅ Updated TABLE_LENGTH = {TABLE_LENGTH}")


# -------------------------------
# 📐 TABLE WIDTH
# -------------------------------
def get_table_width() -> int | float:
    """מחזיר את רוחב השולחן הנוכחי."""
    global TABLE_WIDTH
    return TABLE_WIDTH


def set_table_width(value: int | float) -> None:
    """מעדכן את רוחב השולחן, בתנאי שהערך חיובי."""
    global TABLE_WIDTH
    if value <= 0:
        raise ValueError("TABLE_WIDTH must be a positive number.")
    TABLE_WIDTH = value
    print(f"✅ Updated TABLE_WIDTH = {TABLE_WIDTH}")


def get_ball_radius() -> int | float:
    """מחזיר את רדיוס הכדור הנוכחי."""
    return get_table_length() / 100


def get_ball_radius_photo() -> int | float:
    """מחזיר את רדיוס הכדור הנוכחי לתמונות."""
    return get_table_length() / 60


def get_pocket_margin() -> int | float:
    """מחזיר את מרווח הכיסים הנוכחי."""
    return get_table_length() / 30


def get_wall_margin() -> int | float:
    """מחזיר את מרווח הקירות הנוכחי."""
    return get_table_length() / 37


def get_ball_diameter() -> int | float:
    """מחזיר את קוטר הכדור הנוכחי."""
    return get_ball_radius() * 2.2


def get_corner_pocket_radius() -> int | float:
    """מחזיר את רדיוס הכיסים הפינתיים הנוכחי."""
    return get_table_length() / 72.5


def get_side_pocket_radius() -> int | float:
    """מחזיר את רדיוס הכיסים הצדדיים הנוכחי."""
    return get_table_length() / 64.4


def get_min_distance_from_pocket() -> int | float:
    """מחזיר את המרחק המינימלי מהכיס לפגיעה בטוחה."""
    return get_ball_radius() * 1.2


def get_safe_distance() -> int | float:
    """מחזיר את המרחק הבטוח בין כדורים."""
    return get_ball_radius() * 0.5


def get_pocket_radius() -> int | float:
    """מחזיר את רדיוס הכיס הנוכחי."""
    return get_table_length() / 50


def get_pocket_up_radius() -> int | float:
    return get_pocket_radius() * 1.5


def get_pocket_down_radius() -> int | float:
    return get_pocket_radius() * 0.5


def get_pocket_radius_determinate() -> int | float:
    return get_pocket_radius() * 0.5


def get_ball_radius_determinate() -> int | float:
    return get_ball_radius() * 0.3


def get_max_white_to_target_distance() -> int | float:
    """מחזיר את המרחק המקסימלי בין הכדור הלבן לכדור היעד."""
    return get_table_length() / 2


def get_get_max_white_to_target_distance() -> int | float:
    """מחזיר את המרחק המקסימלי בין כדור היעד לכיס."""
    return math.hypot(get_table_length(), get_table_width()) / 2


def get_crop_half_size() -> int:
    """מחזיר את חצי הגודל של החיתוך לתמונות."""
    return get_table_length() / 9.5


def get_safe_from_wall() -> int | float:
    """מחזיר את המרחק הבטוח מהקירות."""
    return get_ball_radius() * 0.5


def get_merge_max_diff() -> int:
    return int(get_table_length() / 20)  # מרחק מותר בציר X


def get_ball_type() -> str:
    return BALL_TYPE


def set_ball_type(value: str) -> None:
    if value == "solids":
        value = Ball_Color.SOLID
    elif value == "stripes":
        value = Ball_Color.STRIPED
    if value not in {
        Ball_Color.SOLID,
        Ball_Color.STRIPED,
    }:
        raise ValueError("Invalid ball type.")
    global BALL_TYPE
    BALL_TYPE = value


def get_pocket_path() -> str:
    return POCKET_PATH


def set_pocket_path(value: str) -> None:
    global POCKET_PATH
    POCKET_PATH = value


def get_rectangle_croped() -> Rectangle | None:
    return RECTANGLE_CROPED


def set_rectangle_croped(value: Rectangle) -> None:
    global RECTANGLE_CROPED
    RECTANGLE_CROPED = value


def get_use_predicted_pockets() -> bool:
    return USE_PREDICTED_POCKETS


def set_use_predicted_pockets(value: bool) -> None:
    global USE_PREDICTED_POCKETS
    USE_PREDICTED_POCKETS = value


def get_detected_pockets() -> List[Pocket] | None:
    return DETECTED_POCKETS


def set_detected_pockets(value: List[Pocket]) -> None:
    print ("✅ Setting detected pockets:")
    print (value)
    global DETECTED_POCKETS
    DETECTED_POCKETS = value

def clamp_to_table_pocket(x: float, length: float) -> float:
    """גזירה לגבולות השולחן תוך שמירה על רדיוס הכדור."""
    return max(get_ball_radius(), min(length - get_ball_radius(), x))

def set_set_detected_pockets_to_upside_downside() -> None:
    global DETECTED_POCKETS
    pockets_after_conversion: List[Pocket] = []
    sx = get_table_length() / max(1.0, get_width_px())
    sy = get_table_width() / max(1.0, get_height_px())

    for pocket in DETECTED_POCKETS:
        game_x_pocket = clamp_to_table_pocket(pocket.center[0] * sx, get_table_length())
        game_y_pocket = clamp_to_table_pocket((get_height_px() - pocket.center[1]) * sy, get_table_width())
        pockets_after_conversion.append(
            Pocket(
                id=pocket.id,
                center=(game_x_pocket, game_y_pocket),
                radius=pocket.radius,
                location=pocket.location if pocket.location is not None else "",
                pocket_img_cordinates_on_table=pocket.pocket_img_cordinates_on_table,
                pocket_img_path=pocket.pocket_img_path if pocket.pocket_img_path is not None else ""
            )
        )
    DETECTED_POCKETS = pockets_after_conversion


def get_width_px() -> float:
    return WIDTH_PX

def set_width_px(value: float) -> None:
    global WIDTH_PX
    WIDTH_PX = value

def get_height_px() -> float:
    return HEIGHT_PX

def set_height_px(value: float) -> None:
    global HEIGHT_PX
    HEIGHT_PX = value


MERGE_MAX_Y_DIFF = 65  # מרחק מותר בציר Y
NOT_FREE_SHOT = "dont have a free shot"

OUTPUT_JSON_PATH = "photos/output/img_JSON.json"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"

FRONTEND_DIR = BASE_DIR / "frontend"


OUTPUT_IMAGE_PATH = OUTPUT_DIR / "img.png"
OUTPUT_CONTACT_VIEW_PATH = BASE_DIR / "img_contact.png"
FORSE_WALL_SHOT = False


RECTANGLE_JSON_PATH = "rectangles_cache.json"
FELT_MASK_PATH = "output/debug/black_white_detect/01_felt_mask.jpg"
