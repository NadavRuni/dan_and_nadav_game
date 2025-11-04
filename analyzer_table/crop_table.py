from analyzer_table.launcher_helper.json_models import Rectangle
import numpy as np
import cv2, requests, shutil
from pathlib import Path

def crop_image_by_rectangle(rectangle: Rectangle, image_path: str, output_dir: Path, data: dict):
    """
    Crops an image based on a Rectangle object and scaling info from 'data',
    returns both cropped image path and a new scaled Rectangle that matches
    the size of the new image.

    Args:
        rectangle (Rectangle): original rectangle (top_left, top_right, bottom_left, bottom_right)
        image_path (str): path or URL to image
        output_dir (Path): folder to save cropped image
        data (dict): includes display_width, display_height, original_width, original_height

    Returns:
        tuple[Path, Rectangle]: (cropped_path, scaled_rectangle)
    """
    print("[DEBUG] Starting crop_image_by_rectangle")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ הורדת תמונה מרוחקת אם צריך
    local_path = Path(image_path)
    if str(image_path).startswith("http"):
        filename = Path(image_path).name
        local_path = output_dir / filename
        print(f"[DEBUG] Downloading image from URL: {image_path}")
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"[DEBUG] Saved image to: {local_path}")

    # 2️⃣ טעינה
    img = cv2.imread(str(local_path))
    if img is None:
        raise FileNotFoundError(f"❌ Could not load image: {local_path}")

    # 3️⃣ חישוב סקייל בין תצוגה למקור
    display_w = float(data.get("display_width", img.shape[1]))
    display_h = float(data.get("display_height", img.shape[0]))
    original_w = float(data.get("original_width", img.shape[1]))
    original_h = float(data.get("original_height", img.shape[0]))

    scale_x = original_w / display_w
    scale_y = original_h / display_h
    print(f"[DEBUG] Scaling factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # 4️⃣ מיפוי נקודות
    pts = np.float32([
        [rectangle.top_left[0] * scale_x, rectangle.top_left[1] * scale_y],
        [rectangle.top_right[0] * scale_x, rectangle.top_right[1] * scale_y],
        [rectangle.bottom_right[0] * scale_x, rectangle.bottom_right[1] * scale_y],
        [rectangle.bottom_left[0] * scale_x, rectangle.bottom_left[1] * scale_y]
    ])

    print(f"[DEBUG] Scaled points: {pts}")

    # 5️⃣ חישוב גודל יעד
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])

    max_width = int(max(widthA, widthB))
    max_height = int(max(heightA, heightB))

    print(f"[DEBUG] Target crop size: {max_width}x{max_height}")

    # 6️⃣ יצירת מטריצת perspective transform
    dst = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))

    # 7️⃣ שמירה
    cropped_path = output_dir / f"cropped_{Path(image_path).stem}.jpeg"
    cv2.imwrite(str(cropped_path), warped)
    print(f"[DEBUG] Cropped image saved to: {cropped_path}")

    # 8️⃣ יצירת מלבן חדש ביחס לגודל התמונה החדשה
    new_rect = Rectangle(
        top_left=(0, 0),
        top_right=(max_width - 1, 0),
        bottom_left=(0, max_height - 1),
        bottom_right=(max_width - 1, max_height - 1),
    )

    print(f"[DEBUG] ✅ Created new scaled Rectangle matching cropped image: "
          f"{new_rect.top_left}, {new_rect.top_right}, {new_rect.bottom_right}, {new_rect.bottom_left}")

    # 9️⃣ החזרת גם התמונה וגם המלבן החדש
    return cropped_path, new_rect
