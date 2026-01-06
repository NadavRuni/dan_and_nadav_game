"""
A utility for parsing a Rectangle object from different dictionary formats.
"""

from typing import Dict, Any, Tuple

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Rectangle


def parse_rectangle_from_data(data: Dict[str, Any]) -> Rectangle:
    """
    Parses a dictionary to create a Rectangle object.

    This function supports two common dictionary structures:
    1. A dictionary with a 'points' key containing a list of four
       coordinate dictionaries (e.g., from a frontend UI).
    2. A dictionary with keys like 'top_left', 'top_right', etc., directly
       representing a Rectangle object (e.g., from a JSON file).

    Args:
        data: The dictionary containing the rectangle data.

    Returns:
        A populated Rectangle object.

    Raises:
        ValueError: If the data format is invalid or does not contain the
                    expected keys or number of points.
    """
    # Case 1: Data comes from a frontend-style 'points' list
    if "points" in data:
        points = data.get("points", [])
        if len(points) != 4:
            raise ValueError(f"Expected 4 points, but got {len(points)}")

        # Sort points by y-coordinate to separate top and bottom pairs
        pts_sorted_by_y = sorted(points, key=lambda p: p["y"])

        # Sort the top and bottom pairs by x-coordinate to find corners
        top_points = sorted(pts_sorted_by_y[:2], key=lambda p: p["x"])
        bottom_points = sorted(pts_sorted_by_y[2:], key=lambda p: p["x"])

        rectangle = Rectangle(
            top_left=(int(top_points[0]["x"]), int(top_points[0]["y"])),
            top_right=(int(top_points[1]["x"]), int(top_points[1]["y"])),
            bottom_left=(int(bottom_points[0]["x"]), int(bottom_points[0]["y"])),
            bottom_right=(int(bottom_points[1]["x"]), int(bottom_points[1]["y"])),
        )
        Debugger.log(f"Rectangle parsed successfully from 'points': {rectangle}")
        return rectangle

    # Case 2: Data comes from a direct dictionary representation of a Rectangle
    elif all(
        k in data for k in ["top_left", "top_right", "bottom_left", "bottom_right"]
    ):
        rectangle = Rectangle(
            top_left=tuple(map(int, data["top_left"])),
            top_right=tuple(map(int, data["top_right"])),
            bottom_left=tuple(map(int, data["bottom_left"])),
            bottom_right=tuple(map(int, data["bottom_right"])),
        )
        Debugger.log(f"Rectangle parsed successfully from dictionary: {rectangle}")
        return rectangle

    else:
        raise ValueError(
            "Invalid rectangle data format. Expected a 'points' key or "
            "keys for all four corners ('top_left', etc.)."
        )
