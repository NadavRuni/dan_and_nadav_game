"""
A utility for creating a Rectangle object from raw point data.

This module provides a function to convert a dictionary of coordinates into a
geometrically sorted Rectangle object.
"""

from typing import Dict, Any, List, Tuple

from analyzer_table.launcher_herlper.json_models import Rectangle


def create_rectangle_from_data(point_data: Dict[str, Any]) -> Rectangle:
    """
    Creates a Rectangle from a dictionary of points.

    The function expects a dictionary with a "points" key, which holds a list
    of four dictionaries, each with "x" and "y" keys. It sorts these points
    geometrically to identify the top-left, top-right, bottom-left, and
    bottom-right corners.

    The sorting logic assumes that the four points roughly form a rectangle
    and are not in a degenerate configuration.

    Args:
        point_data: A dictionary containing the list of points, for example:
                    {'points': [{'x': 1, 'y': 1}, ...]}

    Returns:
        A Rectangle object with its corner attributes correctly assigned.
    """
    # Extract points from the dictionary
    points: List[Tuple[int, int]] = [
        (int(p["x"]), int(p["y"])) for p in point_data["points"]
    ]

    # Sort points first by y-coordinate, then by x-coordinate.
    # This groups the points into top and bottom pairs.
    points_sorted_by_y = sorted(points, key=lambda p: (p[1], p[0]))

    # The first two points are the top pair; sort them by x to find left/right.
    top_points = sorted(points_sorted_by_y[:2], key=lambda p: p[0])

    # The last two points are the bottom pair; sort them by x to find left/right.
    bottom_points = sorted(points_sorted_by_y[2:], key=lambda p: p[0])

    # Construct the Rectangle object with correctly ordered corners.
    rectangle = Rectangle(
        top_left=top_points[0],
        top_right=top_points[1],
        bottom_left=bottom_points[0],
        bottom_right=bottom_points[1],
    )
    return rectangle
