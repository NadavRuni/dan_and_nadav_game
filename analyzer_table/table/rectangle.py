from typing import Dict, Tuple
from analyzer_table.launcher_helper.json_models import Rectangle

def parse_rectangle_from_data(data: Dict) -> Rectangle:
    """
    מחזירה מופע Rectangle מתוך dict.
    תומכת בשני מבנים:
    1. {"points": [{"x":..,"y":..}, ...]}  → מה-frontend
    2. {"top_left": [...], "top_right": [...], ...}  → מקובץ rectangle.json
    """
    # 🟢 אם נשלח כ-4 נקודות מה-frontend
    if "points" in data:
        points = data.get("points", [])
        if len(points) != 4:
            raise ValueError(f"Expected 4 points, got {len(points)}")

        pts = [(p["x"], p["y"]) for p in points]
        pts_sorted_by_y = sorted(pts, key=lambda p: p[1])

        top_points = sorted(pts_sorted_by_y[:2], key=lambda p: p[0])
        bottom_points = sorted(pts_sorted_by_y[2:], key=lambda p: p[0])

        rect = Rectangle(
            top_left=(int(top_points[0][0]), int(top_points[0][1])),
            top_right=(int(top_points[1][0]), int(top_points[1][1])),
            bottom_left=(int(bottom_points[0][0]), int(bottom_points[0][1])),
            bottom_right=(int(bottom_points[1][0]), int(bottom_points[1][1])),
        )
        print(f"[DEBUG] Rectangle parsed successfully from 'points': {rect}")
        return rect

    # 🟣 אם זה קובץ rectangle.json (אובייקט מלבן ישיר)
    elif all(k in data for k in ["top_left", "top_right", "bottom_left", "bottom_right"]):
        rect = Rectangle(
            top_left=tuple(map(int, data["top_left"])),
            top_right=tuple(map(int, data["top_right"])),
            bottom_left=tuple(map(int, data["bottom_left"])),
            bottom_right=tuple(map(int, data["bottom_right"])),
        )
        print(f"[DEBUG] Rectangle parsed successfully from dict: {rect}")
        return rect

    else:
        raise ValueError("Invalid rectangle data format — expected 'points' or 'top_left' keys.")
