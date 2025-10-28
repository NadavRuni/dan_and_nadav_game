from analyzer_table.table.table import confirm_or_correct_rectangle
from analyzer_table.detect_ball.detect_table import find_table_rectangle
from analyzer_table.launcher_helper.json_models import Rectangle

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
    else :
        print("[DEBUG] ✅ Rectangle detected. Asking user for confirmation...")
    confirmed_rect = confirm_or_correct_rectangle(image_path, rect)
    return confirmed_rect
