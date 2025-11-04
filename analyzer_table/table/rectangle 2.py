from typing import Tuple, List, Dict
from analyzer_table.launcher_helper.json_models import Rectangle
def parse_rectangle_from_data(data: Dict) -> Rectangle:
    """
    מקבל dict כמו מה-frontend ומחזיר מופע Rectangle לפי מיקומי הפינות.
    מערכת הצירים: (0,0) בפינה השמאלית העליונה.
    """
    points = data.get("points", [])
    if len(points) != 4:
        raise ValueError(f"Expected 4 points, got {len(points)}")

    # המרה לרשימת tuples נוחה
    pts = [(p["x"], p["y"]) for p in points]

    # חלוקה לפי ערך y (קטן = גבוה יותר)
    pts_sorted_by_y = sorted(pts, key=lambda p: p[1])

    # שתי הנקודות העליונות (y קטנים יותר)
    top_points = sorted(pts_sorted_by_y[:2], key=lambda p: p[0])  # לפי x
    bottom_points = sorted(pts_sorted_by_y[2:], key=lambda p: p[0])  # לפי x

    rect = Rectangle(
        top_left=(int(top_points[0][0]), int(top_points[0][1])),
        top_right=(int(top_points[1][0]), int(top_points[1][1])),
        bottom_left=(int(bottom_points[0][0]), int(bottom_points[0][1])),
        bottom_right=(int(bottom_points[1][0]), int(bottom_points[1][1])),
    )

    print(f"[DEBUG] Rectangle parsed successfully:\n{rect}")
    return rect