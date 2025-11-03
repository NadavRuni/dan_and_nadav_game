from analyzer_table.launcher_helper.json_models import PhotoData, Ball, Origin, Rectangle
from analyzer_table.detect_ball.Debugger import Debugger
import math
import cv2
import numpy as np
from const_numbers import *

# קבועים לקביעת גבולות מיזוג
def mergeData(main_photo: PhotoData, sub_photos: list[PhotoData], black_and_white_list: list[Ball],  table_rectangle: Rectangle) -> PhotoData:
    """
    מאחד את כל הכדורים מהתמונה הראשית וכל שאר התמונות.
    כדורים קרובים (מתחת לסף מרחק) נחשבים כאותו כדור.
    כדורים שאינם לפחות 80% בתוך גבולות השולחן — לא ייכללו.
    """
    Debugger.log("🔄 Starting mergeData process with table filtering")

    # === שלב 1: אתחול רשימת הכדורים מהתמונה הראשית ===

    main_balls = [Ball(center=b.center, radius=b.radius) for b in main_photo.balls]
    merged_balls = []
    for b in main_balls:
        if not is_inside_table(b , table_rectangle):
            # Debugger.log(f"Skipping ball at {b.center} from main image (outside table)")
            continue

        if _ball_exists(merged_balls, b):
            # Debugger.log(f"Skipping duplicate ball at {b.center} from main image")
            continue
        Debugger.log(f"Adding ball at {b.center} with radius {b.radius} from main image")
        merged_balls.append(Ball(center=b.center, radius=b.radius))



    Debugger.log(f"Initialized with {len(merged_balls)} balls from main image")

    
    # === שלב 3: איחוד כל הכדורים משאר התמונות ===
    added, skipped, duplicates = 0, 0, 0

    for photo in sub_photos:
        for b in photo.balls:
            if not is_inside_table(b , table_rectangle):
                skipped += 1
                continue

            if _ball_exists(merged_balls, b):
                duplicates += 1
                continue
            Debugger.log(f"Adding ball at {b.center} with radius {b.radius} from sub-image")
            merged_balls.append(Ball(center=b.center, radius=b.radius))
            added += 1

    # merge black and white balls        
    for b in black_and_white_list:
        if not is_inside_table(b , table_rectangle):
            skipped += 1
            continue

        if _ball_exists(merged_balls, b):
            duplicates += 1
            continue

        Debugger.log(f"Adding black_white ball at {b.center} with radius {b.radius}")
        merged_balls.append(Ball(center=b.center, radius=b.radius))
        added += 1

    # === שלב 4: יצירת אובייקט מאוחד ===
    finall_balls = [] 
    for ball in merged_balls:
        if not is_inside_table(ball , table_rectangle):
            skipped += 1
            continue
        finall_balls.append(ball)

        

         
    merged_photo = PhotoData(
        cut_name="merged_all.png",
        origin=Origin(0, 0),
        rectangle=main_photo.rectangle,
        balls=finall_balls
    )

    # === שלב 5: סיכום תוצאות יפה ===
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"\n{BOLD}{CYAN}========== ⚪ MERGE SUMMARY =========={RESET}")
    print(f"Total balls from main image:        {len(main_photo.balls)}")
    print(f"Balls added from sub-images:        {added}")
    print(f"Balls skipped (outside table):      {skipped}")
    print(f"Duplicate balls ignored:            {duplicates}")
    print(f"{GREEN}{BOLD}Final unique balls:                 {len(merged_balls)}{RESET}\n")

    Debugger.log(f"✅ Merge complete — {len(merged_balls)} unique balls retained")

    return merged_photo


def _ball_exists(merged_balls: list[Ball], new_ball: Ball) -> bool:
    """בודק אם כדור כבר קיים (לפי קרבה גאומטרית ברדיוס ובמיקום)."""

    for existing in merged_balls:

        dx = abs(existing.center[0] - new_ball.center[0])
        dy = abs(existing.center[1] - new_ball.center[1])

        # אם הם קרובים מאוד — נחשב אותו כדור
        if dx <= get_merge_max_diff() and dy <= get_merge_max_diff() :
            return True
        
    return False

def is_inside_table(ball: Ball, rect: Rectangle) -> bool:
    """
    בודקת אם כל הכדור (כולל רדיוס + SAFE_PATH) נמצא בתוך המלבן (נטוי).
    לא נחשב כ'בפנים' אם הכדור נוגע או חורג מגבולות השולחן.
    """
    x, y = ball.center
    r = ball.radius + get_safe_from_wall()  # כולל מרחק ביטחון

    polygon = [
        rect.top_left,
        rect.top_right,
        rect.bottom_right,
        rect.bottom_left
    ]

    # פונקציה פנימית לבדיקת נקודה בתוך פוליגון (ray casting)
    def point_in_polygon(px, py, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1):
                inside = not inside
        return inside

    edge_points = [
        (x + r, y), (x - r, y), (x, y + r), (x, y - r),
        (x + r / 1.414, y + r / 1.414), (x - r / 1.414, y - r / 1.414),
        (x + r / 1.414, y - r / 1.414), (x - r / 1.414, y + r / 1.414)
    ]

    # אם כל הנקודות האלו בתוך הפוליגון — הכדור כולו בפנים
    if all(point_in_polygon(px, py, polygon) for (px, py) in edge_points):
        Debugger.log(f"🟢 INSIDE ball center=({x},{y}) r={ball.radius}+safe={get_safe_from_wall()} within polygon {polygon}")
        return True
    else:
        Debugger.log(f"🔴 TOO CLOSE TO EDGE ball center=({x},{y}) r={ball.radius}+safe={get_safe_from_wall()}")
        return False
