import math
from pathlib import Path
from analyzer_table.launcher_helper.json_models import Ball_Color

# table_config.py

TABLE_LENGTH = 290
TABLE_WIDTH = 145
BALL_TYPE = "solid"


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
    return get_table_length() / 5


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
    if value not in {Ball_Color.SOLID,Ball_Color.STRIPED,}:
        raise ValueError("Invalid ball type.")
    global BALL_TYPE
    BALL_TYPE = value
    


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
