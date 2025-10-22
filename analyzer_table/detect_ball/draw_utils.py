import cv2
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import PhotoData, Rectangle
import numpy as np
import os


def draw_balls_on_image(photo_data: PhotoData, image_path: str, output_path: str, rectangle: Rectangle = None):
    """
    מצייר את הכדורים בתמונה על פי הנתונים מתוך photo_data.
    אם rectangle קיים — מצייר גם את המלבן שלו.
    שומר את התמונה החדשה ל-output_path.
    """
    Debugger.log(f"🖼️ Drawing {len(photo_data.balls)} balls on image: {image_path}")

    # קריאת התמונה
    img = cv2.imread(image_path)
    if img is None:
        Debugger.error(f"❌ Failed to load image: {image_path}")
        return

    # ציור כל הכדורים
    for ball in photo_data.balls:
        cx, cy = int(ball.center[0]), int(ball.center[1])
        r = int(ball.radius)

        # מעגל אדום מסביב לכדור
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 3)

        # נקודת מרכז
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

        # כיתוב של רדיוס
        label = f"{cx, cy}"
        cv2.putText(
            img, label,
            (cx - 15, cy - r - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ציור מלבן אם קיים
    if rectangle is not None:
        Debugger.log("🟦 Drawing table rectangle on image")
        pts = [
            rectangle.top_left,
            rectangle.top_right,
            rectangle.bottom_right,
            rectangle.bottom_left
        ]
        cv2.polylines(img, [np.array(pts, np.int32)], isClosed=True, color=(255, 255, 0), thickness=3)

        # כיתוב נקודות הפינות
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "TL", rectangle.top_left, font, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "TR", rectangle.top_right, font, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "BL", rectangle.bottom_left, font, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "BR", rectangle.bottom_right, font, 0.6, (0, 255, 255), 2)

    # שמירה
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    Debugger.log(f"✅ Saved image with {len(photo_data.balls)} drawn balls{' and rectangle' if rectangle else ''} to {output_path}")
