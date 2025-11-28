import math
from typing import List, Tuple
from const_numbers import *
from game_class.C_pocket import Pocket

def find_nearest_pocket(pockets: List[Pocket], center: dict) -> Pocket | None:
    """Finds the nearest pocket object to a given center point."""
    if not pockets:
        return None

    min_distance = float('inf')
    nearest_pocket = None
    target_x = center['x']
    target_y = center['y']

    for pocket in pockets:
        pocket_x, pocket_y = pocket.center
        distance = math.sqrt((target_x - pocket_x)**2 + (target_y - pocket_y)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_pocket = pocket
            
    return nearest_pocket

def fetch_pockets_from_data(data: dict):
    """
    Updates pocket locations based on new data, preserving unmatched pockets.
    """
    if not all(k in data for k in ["pocket_points", "display_width", "display_height", "original_width", "original_height"]):
        print("user did not change pocket detection.")
        return None

    display_w = float(data["display_width"])
    display_h = float(data["display_height"])
    original_w = float(data["original_width"])
    original_h = float(data["original_height"])
    
    scale_x = original_w / display_w
    scale_y = original_h / display_h

    new_centers = data.get("pocket_points", [])
    centers_after_scaling = [{"x": c["x"] * scale_x, "y": c["y"] * scale_y} for c in new_centers]

    old_pockets = get_detected_pockets()
    if old_pockets is None:
        old_pockets = []

    unmatched_pockets = old_pockets.copy()
    updated_pockets = []

    for new_center in centers_after_scaling:
        nearest_pocket = find_nearest_pocket(unmatched_pockets, new_center)
        if nearest_pocket:
            # Create a new pocket with the updated center
            updated_pocket = Pocket(
                id=nearest_pocket.id,
                center=(new_center['x'], new_center['y']),
                radius=nearest_pocket.radius,
                location=nearest_pocket.location,
                pocket_img_cordinates_on_table=(new_center['x'], new_center['y']),
            )
            updated_pockets.append(updated_pocket)
            # Remove from unmatched to prevent matching it again
            unmatched_pockets.remove(nearest_pocket)

    # Combine the newly updated pockets with the remaining old ones
    final_pockets = updated_pockets + unmatched_pockets
    
    set_detected_pockets(final_pockets)

    print(f"[fetch_pockets_from_data] Scaling factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    print(f"[fetch_pockets_from_data] Updated {len(updated_pockets)} pocket(s).")
    print(f"[fetch_pockets_from_data] Kept {len(unmatched_pockets)} old pocket(s).")
    print(f"[fetch_pockets_from_data] Total pockets now: {len(final_pockets)}")
    print ("[fetch_pockets_from_data] Final pockets:")
    print(final_pockets)

def find_nearest_pocket_location(old_pockets: List[Pocket], center: dict) -> str | None:
    """
    Finds the location of the nearest pocket to a given center point.

    Args:
        old_pockets: A list of Pocket objects.
        center: A dictionary with 'x' and 'y' keys representing the center point.

    Returns:
        The location (string) of the nearest pocket, or None if old_pockets is empty.
    """
    nearest = find_nearest_pocket(old_pockets, center)
    return nearest.location if nearest else None



