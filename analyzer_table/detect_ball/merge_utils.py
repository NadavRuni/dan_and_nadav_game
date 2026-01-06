"""
Utilities for merging and filtering ball detections from multiple sources.

This module provides functions to combine lists of detected balls, remove
duplicates based on proximity, and filter out balls that are not within the
playable area of the table.
"""

import math
from typing import List

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import (
    Ball,
    PhotoData,
    Origin,
    Rectangle,
)
from const_numbers import (
    get_merge_overlap_margin,
    get_use_predicted_pockets,
    get_detected_pockets,
    get_safe_from_wall,
    get_pocket_margin_merge,
)


def _ball_exists(merged_balls: List[Ball], new_ball: Ball) -> bool:
    """
    Checks if a new ball is a duplicate of one already in the merged list.

    A ball is considered a duplicate if its center is so close to an existing
    ball that their areas overlap, considering a small margin.

    Args:
        merged_balls: The list of unique balls found so far.
        new_ball: The new ball candidate to check.

    Returns:
        True if the ball is a duplicate, False otherwise.
    """
    for existing_ball in merged_balls:
        delta_x = existing_ball.center[0] - new_ball.center[0]
        delta_y = existing_ball.center[1] - new_ball.center[1]
        distance = (delta_x**2 + delta_y**2) ** 0.5

        # If centers overlap enough, they are considered the same ball.
        overlap_threshold = (
            existing_ball.radius + new_ball.radius + get_merge_overlap_margin()
        )
        if distance < overlap_threshold:
            Debugger.log(
                f"❌ Duplicate ball (overlapping): new={new_ball.center}, "
                f"existing={existing_ball.center}, dist={distance:.2f} < "
                f"threshold={overlap_threshold:.2f}"
            )
            return True
    return False


def _is_ball_a_pocket(ball: Ball) -> bool:
    """
    Checks if a ball-like detection is actually a pocket.

    This function compares the ball's position and radius to the list of
    globally stored pockets. It relies on global state.

    Args:
        ball: The ball to check.

    Returns:
        True if the ball is determined to be a pocket, False otherwise.
    """
    pockets = get_detected_pockets()
    if not pockets:
        return False

    ball_x, ball_y = ball.center
    for pocket in pockets:
        pocket_x, pocket_y = pocket.center
        distance = math.sqrt((ball_x - pocket_x) ** 2 + (ball_y - pocket_y) ** 2)

        # If the distance is less than half the sum of radii, they are
        # likely the same object.
        if distance < (ball.radius + pocket.radius) * 0.5:
            Debugger.log(
                f"⚠️ Ball at ({ball.center[0]:.1f}, {ball.center[1]:.1f}) is "
                f"identified as a pocket ({pocket.location}) and will be excluded."
            )
            return True
    return False


def is_inside_table(ball: Ball, table_rect: Rectangle) -> bool:
    """
    Checks if a ball is within the playable area of the table.

    This involves two checks:
    1. A point-in-polygon test to see if the ball's boundary is inside the
       main table rectangle.
    2. A check to ensure the ball is not overlapping with a pocket area.

    Args:
        ball: The ball to check.
        table_rect: The Rectangle object defining the table boundaries.

    Returns:
        True if the ball is inside the playable area, False otherwise.
    """
    ball_x, ball_y = ball.center
    radius = ball.radius + get_safe_from_wall()
    polygon_points = [
        table_rect.top_left,
        table_rect.top_right,
        table_rect.bottom_right,
        table_rect.bottom_left,
    ]

    # Use a point-in-polygon test for the ball's center and boundary points
    def point_in_polygon(px, py, poly_verts):
        is_inside = False
        num_verts = len(poly_verts)
        j = num_verts - 1
        for i in range(num_verts):
            ix, iy = poly_verts[i]
            jx, jy = poly_verts[j]
            # Check if the point is on a horizontal line of the polygon
            if ((iy > py) != (jy > py)) and (
                px < (jx - ix) * (py - iy) / (jy - iy + 1e-9) + ix
            ):
                is_inside = not is_inside
            j = i
        return is_inside

    # Check center and 8 cardinal points around the ball's circumference
    edge_points = [
        (ball_x + radius, ball_y),
        (ball_x - radius, ball_y),
        (ball_x, ball_y + radius),
        (ball_x, ball_y - radius),
        (ball_x + radius / 1.414, ball_y + radius / 1.414),
        (ball_x - radius / 1.414, ball_y - radius / 1.414),
        (ball_x + radius / 1.414, ball_y - radius / 1.414),
        (ball_x - radius / 1.414, ball_y + radius / 1.414),
    ]

    if not all(point_in_polygon(px, py, polygon_points) for (px, py) in edge_points):
        return False

    # After confirming it's inside the main rectangle, check if it's a pocket
    if get_use_predicted_pockets():
        return not _is_ball_a_pocket(ball)

    return True


def mergeData(
    main_photo: PhotoData,
    sub_photos: List[PhotoData],
    black_and_white_list: List[Ball],
    table_rectangle: Rectangle,
) -> PhotoData:
    """
    Merges ball detections from the main image, sub-images, and a dedicated
    black/white detection pass into a single, de-duplicated list.

    Args:
        main_photo: PhotoData from the full image analysis.
        sub_photos: A list of PhotoData from the cropped sub-images.
        black_and_white_list: A list of balls from a separate b/w detection.
        table_rectangle: The rectangle defining the table boundaries for filtering.

    Returns:
        A new PhotoData object containing the final merged and filtered list
        of balls.
    """
    Debugger.log("🔄 Starting mergeData process with table filtering")
    merged_balls: List[Ball] = []
    skipped_outside = 0
    duplicates_found = 0

    all_ball_lists = [
        black_and_white_list,
        main_photo.balls,
        *[p.balls for p in sub_photos],
    ]

    for ball_list in all_ball_lists:
        for ball in ball_list:
            if not is_inside_table(ball, table_rectangle):
                skipped_outside += 1
                continue
            if _ball_exists(merged_balls, ball):
                duplicates_found += 1
                continue
            merged_balls.append(Ball(center=ball.center, radius=ball.radius))

    # Create a final PhotoData object for the merged results
    merged_photo = PhotoData(
        cut_name="merged_all.png",
        origin=Origin(0, 0),
        rectangle=main_photo.rectangle,
        balls=merged_balls,
    )

    # Print a summary of the merge process
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    print(f"\n{BOLD}{CYAN}========== ⚪ MERGE SUMMARY =========={RESET}")
    print(f"Balls skipped (outside table):      {skipped_outside}")
    print(f"Duplicate balls ignored:            {duplicates_found}")
    print(
        f"{GREEN}{BOLD}Final unique balls:                 "
        f"{len(merged_balls)}{RESET}\n"
    )
    Debugger.log(f"✅ Merge complete — {len(merged_balls)} unique balls retained")

    return merged_photo


# def _ball_exists(merged_balls: list[Ball], new_ball: Ball) -> bool:
#     for existing in merged_balls:
#         dx = abs(existing.center[0] - new_ball.center[0])
#         dy = abs(existing.center[1] - new_ball.center[1])
#         if dx <= get_merge_max_diff() and dy <= get_merge_max_diff():
#             print (f"❌ Duplicate ball found at {new_ball.center} (existing at {existing.center})")
#             return True
#     return False
def _ball_exists(merged_balls: list[Ball], new_ball: Ball) -> bool:
    for existing in merged_balls:
        dx = existing.center[0] - new_ball.center[0]
        dy = existing.center[1] - new_ball.center[1]
        distance = (dx * dx + dy * dy) ** 0.5

        # If centers overlap enough → same ball
        if distance < (existing.radius + new_ball.radius + get_merge_overlap_margin()):
            print(
                f"❌ Duplicate ball (overlapping): new={new_ball.center}, existing={existing.center}, dist={distance:.2f} < threshold={(existing.radius + new_ball.radius+ get_merge_overlap_margin()):.2f}"
            )
            return True

    return False


def is_inside_inner_rectangle(ball: Ball, rect: Rectangle, margin: float) -> bool:
    if get_use_predicted_pockets():
        pockets = get_detected_pockets()
        if pockets:
            ball_x, ball_y = ball.center
            for pocket in pockets:
                pocket_x, pocket_y = pocket.center
                pocket_radius = pocket.radius

                distance = math.sqrt(
                    (ball_x - pocket_x) ** 2 + (ball_y - pocket_y) ** 2
                )
                # If the distance is less than sum of radii, they are likely the same object
                if distance < (ball.radius + pocket_radius) * 0.8:
                    Debugger.log(
                        f"⚠️ Ball at ({ball.center[0]:.1f}, {ball.center[1]:.1f}) is identified as a pocket ({pocket.location}) and will be excluded."
                    )
                    return False  # It's a pocket, so not "inside" the playable area
                
            # If the ball is not identified as any of the pockets
            return True  # It is not a pocket, so it's a valid ball in this context.

    # Original logic for when get_use_predicted_pockets() is False or no pockets are found
    tlx, tly = rect.top_left
    trx, try_ = rect.top_right
    blx, bly = rect.bottom_left
    brx, bry = rect.bottom_right
    inner_rect = Rectangle(
        top_left=(tlx + margin, tly + margin),
        top_right=(trx - margin, try_ + margin),
        bottom_left=(blx + margin, bly - margin),
        bottom_right=(brx - margin, bry - margin),
    )
    x, y = ball.center[0], ball.center[1]
    min_x = min(inner_rect.top_left[0], inner_rect.bottom_left[0])
    max_x = max(inner_rect.top_right[0], inner_rect.bottom_right[0])
    min_y = min(inner_rect.top_left[1], inner_rect.top_right[1])
    max_y = max(inner_rect.bottom_left[1], inner_rect.bottom_right[1])
    inside = (min_x + ball.radius <= x <= max_x - ball.radius) and (
        min_y + ball.radius <= y <= max_y - ball.radius
    )
    return inside


def is_inside_table(ball: Ball, rect: Rectangle) -> bool:
    x, y = ball.center
    r = ball.radius + get_safe_from_wall()
    polygon = [rect.top_left, rect.top_right, rect.bottom_right, rect.bottom_left]

    def point_in_polygon(px, py, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > py) != (y2 > py)) and (
                px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
            ):
                inside = not inside
        return inside

    edge_points = [
        (x + r, y),
        (x - r, y),
        (x, y + r),
        (x, y - r),
        (x + r / 1.414, y + r / 1.414),
        (x - r / 1.414, y - r / 1.414),
        (x + r / 1.414, y - r / 1.414),
        (x - r / 1.414, y + r / 1.414),
    ]
    if all(point_in_polygon(px, py, polygon) for (px, py) in edge_points):
        return is_inside_inner_rectangle(ball, rect, get_pocket_margin_merge())
    else:
        return False
