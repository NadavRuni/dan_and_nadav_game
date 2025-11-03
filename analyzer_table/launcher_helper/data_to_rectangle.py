from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from analyzer_table.launcher_helper.json_models import Rectangle

def create_rectangle_from_data(data: Dict[str, Any]) -> Rectangle:
    """
    מקבלת מילון בפורמט {'points': [{'x':..,'y':..}, ...]}
    ומחזירה Rectangle עם פינות מסודרות גיאומטרית:
    top_left, top_right, bottom_left, bottom_right
    """
    # שליפת הנקודות
    points: List[Tuple[int, int]] = [(int(p["x"]), int(p["y"])) for p in data["points"]]

    # מיון לפי y (גובה) ואח"כ לפי x (רוחב)
    points_sorted = sorted(points, key=lambda p: (p[1], p[0]))

    # שתי העליונות
    top_points = sorted(points_sorted[:2], key=lambda p: p[0])
    # שתי התחתונות
    bottom_points = sorted(points_sorted[2:], key=lambda p: p[0])

    # בניית מלבן
    rect = Rectangle(
        top_left=top_points[0],
        top_right=top_points[1],
        bottom_left=bottom_points[0],
        bottom_right=bottom_points[1]
    )
    return rect
