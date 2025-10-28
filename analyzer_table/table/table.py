import cv2
from dataclasses import dataclass
from typing import Tuple
from analyzer_table.launcher_helper.json_models import Rectangle


def show_rectangle_for_confirmation(image, rect: Rectangle | None):
    """מציג את המלבן שזוהה ומאפשר למשתמש לאשר או לתקן."""
    preview = image.copy()
    if rect:
        pts = [rect.top_left, rect.top_right, rect.bottom_right, rect.bottom_left]
        for i in range(4):
            cv2.line(preview, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
    else:
        cv2.putText(preview, "No rectangle detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detected Table Rectangle", preview)
    print("👉 Click the OpenCV window and press 'y' to confirm, 'n' to adjust manually, or 'q' to cancel.")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def manual_select_rectangle(image):
    """מאפשר למשתמש לבחור ידנית 4 פינות של השולחן בעזרת קליקים."""
    points = []
    clone = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Manual Selection", clone)

    print("🖱️ Click 4 corners of the table (clockwise from top-left). Press ESC to cancel.")
    cv2.imshow("Manual Selection", clone)
    cv2.setMouseCallback("Manual Selection", click_event)

    while True:
        if len(points) >= 4:
            break
        if cv2.waitKey(1) == 27:  # ESC to cancel
            points.clear()
            break

    cv2.destroyAllWindows()
    return points


def confirm_or_correct_rectangle(image_path: str, rectangle: Rectangle | None) -> Rectangle | None:
    """
    מאפשר למשתמש לאשר או לתקן מלבן שנמצא בתמונה.
    אם המשתמש לוחץ 'n' – הוא נכנס לתיקון ידני, ואז רואה שוב את התוצאה ויכול להחליט שוב.
    """
    img = cv2.imread(image_path)

    while True:
        key = show_rectangle_for_confirmation(img, rectangle)

        if key == ord('y'):
            print("✅ User confirmed detected rectangle.")
            return rectangle

        elif key == ord('n'):
            print("🖱️ User requested manual correction.")
            pts = manual_select_rectangle(img)
            if len(pts) == 4:
                rectangle = Rectangle(
                    top_left=pts[0],
                    top_right=pts[1],
                    bottom_right=pts[2],
                    bottom_left=pts[3]
                )
                print("✅ User provided new rectangle. Showing confirmation...")
                # נחזור ללולאה – כדי שיוכל לאשר או לתקן שוב
                continue
            else:
                print("❌ Not enough points selected. Returning to confirmation screen...")
                continue

        elif key == ord('q') or key == 27:  # 'q' או ESC
            print("❌ User canceled operation.")
            return None

        else:
            print("⚠️ Invalid key pressed. Please use 'y', 'n', or 'q'.")
