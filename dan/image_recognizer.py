import cv2
from ultralytics import YOLO
import numpy as np
import os
import math

# ==== CONFIGURATION ====
IMAGE_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-11.jpg"
OUTPUT_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/table-11.jpg"
MODEL_PATH = "yolov8n.pt"
SPORTS_BALL_CLASS_ID = 32
YOLO_CONF = 0.004
YOLO_IOU  = 0.40
YOLO_IMGZ = 1536
YOLO_MAXD = 300
USE_TTA   = True

# New constraints
MAX_BALLS = 16
MIN_RADIUS_REL = 0.009   # minimal radius as fraction of min(image_height, image_width)
MIN_RADIUS_PX_OVERRIDE = None  # set to an int (e.g., 10) to force absolute min radius in pixels
# =======================

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def est_radius_from_box(box):
    x1, y1, x2, y2 = box
    return 0.5 * min(max(0, x2 - x1), max(0, y2 - y1))

def draw_detections(image, boxes, scores, label="ball", color=(0,255,0)):
    out = image.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(3, int(0.25 * min(x2 - x1, y2 - y1)))
        cv2.circle(out, (cx, cy), radius, (0, 255, 255), 2)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out

def yolo_detect(img):
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Running YOLO...")
    results = model.predict(
        source=img,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        classes=[SPORTS_BALL_CLASS_ID],
        imgsz=YOLO_IMGZ,
        augment=USE_TTA,
        max_det=YOLO_MAXD,
        verbose=False
    )

    boxes_px, scores = [], []
    if len(results) > 0 and results[0].boxes is not None:
        for b in results[0].boxes:
            xyxy = b.xyxy.squeeze().tolist()
            conf = float(b.conf.item())
            boxes_px.append(tuple(xyxy))
            scores.append(conf)

    print(f"YOLO detected {len(boxes_px)} ball(s).")
    return boxes_px, scores

def hough_fallback(img, existing_boxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape[:2]
    r_min = max(6, int(0.009 * min(h, w)))
    r_max = max(r_min + 6, int(0.015 * min(h, w)))

    print(f"Running HoughCircles (r_min={r_min}, r_max={r_max})...")
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(20, r_min*2),
        param1=120, param2=18, minRadius=r_min, maxRadius=r_max
    )

    boxes_px, scores, radii = [], [], []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for x, y, r in circles:
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            # Skip if it highly overlaps an existing YOLO box
            too_close = False
            for bx in existing_boxes:
                if iou_xyxy((x1, y1, x2, y2), bx) > 0.2:
                    too_close = True
                    break
            if not too_close:
                boxes_px.append((float(x1), float(y1), float(x2), float(y2)))
                scores.append(0.50)  # pseudo-score
                radii.append(float(r))
    print(f"Hough fallback added {len(boxes_px)} ball(s).")
    return boxes_px, scores, radii

def filter_and_limit(img, boxes, scores, radii=None):
    """Remove too-small balls and keep at most MAX_BALLS by score."""
    h, w = img.shape[:2]
    min_r_px = MIN_RADIUS_PX_OVERRIDE if MIN_RADIUS_PX_OVERRIDE is not None else int(MIN_RADIUS_REL * min(h, w))

    kept = []
    for i, (bx, sc) in enumerate(zip(boxes, scores)):
        r = radii[i] if (radii is not None and i < len(radii)) else est_radius_from_box(bx)
        if r >= min_r_px:
            kept.append((bx, sc, r))

    # Sort by score descending and take top MAX_BALLS
    kept.sort(key=lambda t: t[1], reverse=True)
    kept = kept[:MAX_BALLS]

    if radii is None:
        return [k[0] for k in kept], [k[1] for k in kept]
    else:
        return [k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept]

def main():
    if not os.path.isfile(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        return
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to read image.")
        return

    # 1) YOLO
    yolo_boxes, yolo_scores = yolo_detect(img)

    # 2) Hough fallback
    h_boxes, h_scores, h_radii = hough_fallback(img, yolo_boxes)

    # 3) Merge
    all_boxes = yolo_boxes + h_boxes
    all_scores = yolo_scores + h_scores
    all_radii  = [est_radius_from_box(b) for b in yolo_boxes] + h_radii

    print(f"Total before filtering: {len(all_boxes)}")

    # 4) Filter small balls and limit to MAX_BALLS
    f_boxes, f_scores, f_radii = filter_and_limit(img, all_boxes, all_scores, all_radii)
    print(f"After min-radius filter and limit({MAX_BALLS}): {len(f_boxes)}")

    # 5) Draw and save (color by source just for visibility)
    # Split again to tag source in drawing (optional)
    # Here we simply draw all in one color
    annotated = draw_detections(img, f_boxes, f_scores, label="ball", color=(0,255,0))
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Annotated image saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
