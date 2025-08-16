# detect_billiard_balls_fixed_path.py
# Requirements (first time):
#   pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO
import os

# ==== CONFIGURATION ====
IMAGE_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-9.jpg"   # Change this to your image path
OUTPUT_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/table-9.jpg"
CONFIDENCE_THRESHOLD = 0.1
MODEL_PATH = "yolov8n.pt"  # you can change to yolov8m.pt or yolov8l.pt
# =======================

# COCO class index for 'sports ball' is 32
SPORTS_BALL_CLASS_ID = 32

def draw_detections(image, boxes, scores, label="ball"):
    out = image.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(3, int(0.25 * min(x2 - x1, y2 - y1)))
        cv2.circle(out, (cx, cy), radius, (0, 255, 255), 2)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out

def main():
    if not os.path.isfile(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to read image.")
        return

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print(f"Running detection on {IMAGE_PATH}...")
    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        classes=[SPORTS_BALL_CLASS_ID],
        verbose=False,
        imgsz=1024
    )

    boxes_px = []
    scores = []
    if len(results) > 0 and results[0].boxes is not None:
        r = results[0]
        for b in r.boxes:
            xyxy = b.xyxy.squeeze().tolist()
            conf = float(b.conf.item())
            boxes_px.append(tuple(xyxy))
            scores.append(conf)

    print(f"Detected {len(boxes_px)} ball(s).")
    annotated = draw_detections(img, boxes_px, scores, label="ball")
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Annotated image saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
