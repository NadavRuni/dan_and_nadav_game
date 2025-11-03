import cv2
import os
import numpy as np
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import PhotoData, Rectangle, table_pockets


def draw_balls_on_image(
    photo_data: PhotoData,
    image_path: str,
    output_path: str,
    rectangle: Rectangle = None,
    all_pockets: table_pockets = None
):
    """
    🎱 מצייר על תמונה:
      1. את כל הכדורים (balls)
      2. את מלבן השולחן (אם קיים)
      3. את כל הכיסים (pockets) עם נקודת מרכז ורדיוס.

    Args:
        photo_data: נתוני הכדורים והמלבן.
        image_path: נתיב לקובץ המקור.
        output_path: נתיב לשמירה לאחר הציור.
        rectangle: מלבן השולחן (לא חובה).
        all_pockets: אובייקט table_pockets לציור החורים (לא חובה).
    """
    Debugger.log(f"🖼️ Drawing balls{' and pockets' if all_pockets else ''} on image: {image_path}")

    # טעינת תמונה
    img = cv2.imread(image_path)
    if img is None:
        Debugger.error(f"❌ Failed to load image: {image_path}")
        return

    # 🎯 ציור כדורים
    for ball in photo_data.balls:
        cx, cy = int(ball.center[0]), int(ball.center[1])
        r = int(ball.radius)

        cv2.circle(img, (cx, cy), r, (0, 0, 255), 3)   # היקף אדום
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)  # מרכז ירוק
        label = f"{cx}, {cy}"
        cv2.putText(
            img, label,
            (cx - 20, cy - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

    # 🟦 ציור מלבן השולחן
    if rectangle is not None:
        Debugger.log("🟦 Drawing table rectangle on image")
        pts = [
            rectangle.top_left,
            rectangle.top_right,
            rectangle.bottom_right,
            rectangle.bottom_left
        ]
        cv2.polylines(img, [np.array(pts, np.int32)], isClosed=True, color=(255, 255, 0), thickness=3)

        # ציור נקודות פינות
        for point_name, point in zip(
            ["TL", "TR", "BR", "BL"],
            [rectangle.top_left, rectangle.top_right, rectangle.bottom_right, rectangle.bottom_left]
        ):
            cv2.circle(img, point, radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, point_name, (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 🎱 ציור חורים (Pockets)
    if all_pockets is not None and all_pockets.pocket_list:
        Debugger.log(f"🎯 Drawing {len(all_pockets.pocket_list)} pockets on table")
        for pocket in all_pockets.pocket_list:
            cx, cy = pocket.pocket_img_cordinates_on_table
            cx, cy = int(cx), int(cy)
            r = int(pocket.pocker_radius)

            # עיגול כחול סביב החור
            cv2.circle(img, (cx, cy), r, (255, 0, 0), 2)
            # נקודת מרכז אדומה
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            # כיתוב מזהה
            label = f"ID {pocket.pocket_id}"
            cv2.putText(img, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            Debugger.log(f"🕳️ Pocket {pocket.pocket_id}: Center={cx, cy}, Radius={r}")

    # 💾 שמירה
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    Debugger.log(f"✅ Saved image with {len(photo_data.balls)} balls{' and rectangle' if rectangle else ''}{' and pockets' if all_pockets else ''} → {output_path}")
