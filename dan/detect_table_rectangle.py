"""
Handles the detection, user confirmation, and dimension updates for the table
rectangle.
"""

import math
from typing import Optional

from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.launcher_helper.json_models import Rectangle
from analyzer_table.table.table import confirm_or_correct_rectangle
from const_numbers import set_table_length, set_table_width


def update_table_size_from_rectangle(rectangle: Rectangle) -> None:
    """
    Updates the global table length and width based on a detected rectangle.

    This function calculates the horizontal and vertical distances between the
    rectangle's corners and assumes the larger dimension is the length.

    Args:
        rectangle: The Rectangle object detected in the image.
    """
    # Calculate the distances in pixels
    width_px = math.dist(rectangle.top_left, rectangle.top_right)
    height_px = math.dist(rectangle.top_left, rectangle.bottom_left)

    print(
        f"[DEBUG] Raw rectangle dimensions: width={width_px:.2f}, "
        f"height={height_px:.2f}"
    )

    # Determine which dimension is length vs. width
    table_length = max(width_px, height_px)
    table_width = min(width_px, height_px)

    # Update the global state with the new dimensions
    set_table_length(table_length)
    set_table_width(table_width)

    print(
        f"[INFO] ✅ Updated table size: LENGTH={table_length:.2f}, "
        f"WIDTH={table_width:.2f}"
    )


def detect_and_confirm_table_rectangle(image_path: str) -> Optional[Rectangle]:
    """
    Detects the table rectangle and initiates the user confirmation workflow.

    This function first attempts to automatically detect the table rectangle.
    It then calls a blocking function that waits for a user to confirm or
    correct the rectangle via a web interface.

    Args:
        image_path: The path to the image being analyzed.

    Returns:
        The user-confirmed Rectangle object, or None if the user does not
        confirm within the timeout period.
    """
    print(f"[DEBUG] 🖼  Detecting table rectangle from: {image_path}")
    try:
        initial_rectangle = find_table_rectangle(image_path)
    except Exception as e:
        print(f"[ERROR] An exception occurred during initial rectangle detection: {e}")
        initial_rectangle = None

    if initial_rectangle is None:
        print("[WARN] ⚠️ No table detected automatically.")
    else:
        print("[DEBUG] ✅ Rectangle detected. Asking user for confirmation...")

    # This call is blocking and waits for user input via a file-based system.
    confirmed_rectangle = confirm_or_correct_rectangle(image_path, initial_rectangle)

    if confirmed_rectangle:
        print("[DEBUG] ✅ User confirmed rectangle.")
        update_table_size_from_rectangle(confirmed_rectangle)
    else:
        print("[WARN] ⚠️ No rectangle was confirmed by the user.")

    return confirmed_rectangle
