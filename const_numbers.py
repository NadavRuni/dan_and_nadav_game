import math
from pathlib import Path
# table_config.py

TABLE_LENGTH = 290
TABLE_WIDTH = 145


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
    return get_table_length()/100

def get_ball_radius_photo() -> int | float:
    """מחזיר את רדיוס הכדור הנוכחי לתמונות."""
    return get_table_length()/14.5

def get_pocket_margin() -> int | float:
    """מחזיר את מרווח הכיסים הנוכחי."""
    return get_table_length()/4.8
def get_wall_margin() -> int | float:
    """מחזיר את מרווח הקירות הנוכחי."""
    return get_table_length()/2.9

def get_ball_diameter() -> int | float:
    """מחזיר את קוטר הכדור הנוכחי."""
    return get_ball_radius() * 2.2




CORNER_POCKET_RADIUS = 4
SIDE_POCKET_RADIUS = 4.5
ADD_TO_POCKET = 3.5

SAFE_DISTANCE = get_ball_radius() * 0.5

WEIGHT_WHITE_TO_TARGET = 1 / (TABLE_LENGTH / 2)
WEIGHT_TARGET_TO_POCKET = 1 / (TABLE_LENGTH / 4)

MAX_WHITE_TO_TARGET = TABLE_LENGTH / 2
MAX_TARGET_TO_POCKET = math.hypot(TABLE_LENGTH, TABLE_WIDTH) / 2

NOT_FREE_SHOT = "dont have a free shot"

OUTPUT_JSON_PATH = "photos/output/img_JSON.json"
OUTPUT_IMAGE_PATH = Path("photos/output/img.png")
OUTPUT_CONTACT_VIEW_PATH = Path("photos/output/img_contact.png")

FORSE_WALL_SHOT = False



MIN_CONFIDENCE = 40 

CROP_HALF_SIZE = 30

RECTANGLE_JSON_PATH = "rectangles_cache.json"
BASE_DIR = Path(__file__).resolve().parent

