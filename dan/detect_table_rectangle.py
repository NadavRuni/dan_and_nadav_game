from analyzer_table.table.table import confirm_or_correct_rectangle
from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.launcher_helper.json_models import Rectangle
from const_numbers import set_table_length, set_table_width


def update_table_size_from_rectangle(rect: Rectangle) -> None:
    """
    מעדכן את אורכי הטבלה (length & width) לפי מלבן שזוהה בתמונה.
    - מחשב את המרחק האופקי והאנכי בין הקודקודים.
    - מוודא שהאורך גדול מהרוחב, ואם לא — מחליף ביניהם.
    """
    import math

    # מחשבים את המרחקים בפיקסלים
    width_px = math.dist(rect.top_left, rect.top_right)
    height_px = math.dist(rect.top_left, rect.bottom_left)

    print(
        f"[DEBUG] Raw rectangle dimensions: width={width_px:.2f}, height={height_px:.2f}"
    )

    # קובעים מה האורך ומה הרוחב
    table_length = max(width_px, height_px)
    table_width = min(width_px, height_px)

    # נורמליזציה או סקלת יחידות אם צריך (כאן נשאיר פיקסלים)
    set_table_length(table_length)
    set_table_width(table_width)

    print(
        f"[INFO] ✅ Updated table size: LENGTH={table_length:.2f}, WIDTH={table_width:.2f}"
    )


def detect_table_rectangle(image_path: str) -> Rectangle | None:
    """
    מנסה לזהות את מלבן השולחן מתוך התמונה,
    ואז מעביר לאישור המשתמש דרך confirm_or_correct_rectangle.
    """
    print(f"[DEBUG] 🖼 Detecting table rectangle from: {image_path}")
    try:
        rect = find_table_rectangle(image_path)  # ← מתוך table.py שלך
    except Exception as e:
        rect = None

    if rect is None:
        print("[WARN] ⚠️ No table detected automatically.")
    else:
        print("[DEBUG] ✅ Rectangle detected. Asking user for confirmation...")
    confirmed_rect = confirm_or_correct_rectangle(image_path, rect)
    print(
        "[DEBUG] ✅ User confirmed rectangle."
        if confirmed_rect
        else "[WARN] ⚠️ No rectangle confirmed by user."
    )
    update_table_size_from_rectangle(confirmed_rect) if confirmed_rect else None

    return confirmed_rect
