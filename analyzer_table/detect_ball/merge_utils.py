from analyzer_table.launcher_helper.json_models import (
    PhotoData,
    Ball,
    Origin,
    Rectangle,
)
from analyzer_table.detect_ball.Debugger import Debugger
import math
import cv2
import numpy as np
from const_numbers import *


def mergeData(
    main_photo: PhotoData,
    sub_photos: list[PhotoData],
    black_and_white_list: list[Ball],
    table_rectangle: Rectangle,
) -> PhotoData:
    Debugger.log("🔄 Starting mergeData process with table filtering")
    main_balls = [Ball(center=b.center, radius=b.radius) for b in main_photo.balls]
    merged_balls = []
    for b in black_and_white_list:
        if not is_inside_table(b, table_rectangle):
            continue
        if _ball_exists(merged_balls, b):

            continue
        Debugger.log(
            f"Adding ball at {b.center} with radius {b.radius} from main image"
        )
        merged_balls.append(Ball(center=b.center, radius=b.radius))
    Debugger.log(f"Initialized with {len(merged_balls)} balls from main image")
    added, skipped, duplicates = 0, 0, 0
    for photo in sub_photos:
        for b in photo.balls:
            if not is_inside_table(b, table_rectangle):
                skipped += 1
                continue
            if _ball_exists(merged_balls, b):
                duplicates += 1
                continue
            Debugger.log(
                f"Adding ball at {b.center} with radius {b.radius} from sub-image"
            )
            merged_balls.append(Ball(center=b.center, radius=b.radius))
            added += 1
    for b in main_balls:
        if not is_inside_table(b, table_rectangle):
            skipped += 1
            continue
        if _ball_exists(merged_balls, b):
            duplicates += 1
            continue
        Debugger.log(f"Adding black_white ball at {b.center} with radius {b.radius}")
        merged_balls.append(Ball(center=b.center, radius=b.radius))
        added += 1
    finall_balls = []
    for ball in merged_balls:
        if not is_inside_table(ball, table_rectangle):
            skipped += 1
            continue
        finall_balls.append(ball)
    merged_photo = PhotoData(
        cut_name="merged_all.png",
        origin=Origin(0, 0),
        rectangle=main_photo.rectangle,
        balls=finall_balls,
    )
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    print(f"\n{BOLD}{CYAN}========== ⚪ MERGE SUMMARY =========={RESET}")
    print(f"Total balls from main image:        {len(main_photo.balls)}")
    print(f"Balls added from sub-images:        {added}")
    print(f"Balls skipped (outside table):      {skipped}")
    print(f"Duplicate balls ignored:            {duplicates}")
    print(
        f"{GREEN}{BOLD}Final unique balls:                 {len(merged_balls)}{RESET}\n"
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
