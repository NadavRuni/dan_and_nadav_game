"""
Utilities for updating pocket locations based on user input.

This module contains functions to process pocket locations provided by a user
from a frontend interface, scale them to the original image coordinates, and
update the application's list of detected pockets.
"""

import math
from typing import Dict, List, Optional

from const_numbers import get_detected_pockets, set_detected_pockets
from game_class.C_pocket import Pocket


def find_nearest_pocket(
    pockets: List[Pocket], center: Dict[str, float]
) -> Optional[Pocket]:
    """
    Finds the nearest pocket object to a given center point.

    Args:
        pockets: A list of Pocket objects to search through.
        center: A dictionary with 'x' and 'y' keys for the point.

    Returns:
        The Pocket object closest to the center point, or None if the list is empty.
    """
    if not pockets:
        return None

    min_distance = float("inf")
    nearest_pocket = None
    target_x = center["x"]
    target_y = center["y"]

    for pocket in pockets:
        pocket_x, pocket_y = pocket.center
        distance = math.sqrt((target_x - pocket_x) ** 2 + (target_y - pocket_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_pocket = pocket

    return nearest_pocket


def fetch_pockets_from_data(data: Dict) -> None:
    """
    Updates global pocket locations based on new data from the frontend.

    This function scales the new pocket points from display coordinates to
    original image coordinates and matches them to the nearest existing
    pockets, updating their centers. Unmatched existing pockets are preserved.

    Note:
        This function relies on global state via `get_detected_pockets` and
        `set_detected_pockets`, which is not a robust design.

    Args:
        data: A dictionary containing the new pocket points and coordinate
              system dimensions.
    """
    required_keys = [
        "pocket_points",
        "display_width",
        "display_height",
        "original_width",
        "original_height",
    ]
    if not all(key in data for key in required_keys):
        print("User did not provide new pocket detection data.")
        return

    # Calculate scaling factors
    scale_x = data["original_width"] / data["display_width"]
    scale_y = data["original_height"] / data["display_height"]

    # Scale the new pocket centers
    new_centers = [
        {"x": c["x"] * scale_x, "y": c["y"] * scale_y}
        for c in data.get("pocket_points", [])
    ]

    old_pockets = get_detected_pockets() or []
    unmatched_pockets = old_pockets.copy()
    updated_pockets = []

    for new_center in new_centers:
        nearest_pocket = find_nearest_pocket(unmatched_pockets, new_center)
        if nearest_pocket:
            # Create a new pocket with the updated center but same ID/radius
            updated_pocket = Pocket(
                id=nearest_pocket.id,
                center=(new_center["x"], new_center["y"]),
                radius=nearest_pocket.radius,
                location=nearest_pocket.location,
                pocket_img_cordinates_on_table=(new_center["x"], new_center["y"]),
            )
            updated_pockets.append(updated_pocket)
            unmatched_pockets.remove(nearest_pocket)

    # Combine the newly updated pockets with the old ones that weren't matched
    final_pockets = updated_pockets + unmatched_pockets
    set_detected_pockets(final_pockets)

    print(f"[fetch_pockets] Updated {len(updated_pockets)} pocket(s).")
    print(f"[fetch_pockets] Total pockets now: {len(final_pockets)}")
