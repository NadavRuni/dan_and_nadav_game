from output_utils import get_output_path
import cv2
from ultralytics import YOLO
import numpy as np
import os
import json

MODEL_PATH = "yolov8n.pt"
SPORTS_BALL_CLASS_ID = 32
YOLO_CONF = 0.1
YOLO_IOU = 0.40
YOLO_IMGZ = 1280
YOLO_MAXD = 30
MAX_BALLS = 16
MIN_RADIUS_REL = 0.01
POCKET_INCLUSION_FACTOR = 1.05


def est_radius_from_box(box):
    x1, y1, x2, y2 = box
    return 0.5 * min(max(0, x2 - x1), max(0, y2 - y1))


def boxes_to_centers(boxes):
    return [(0.5 * (x1 + x2), 0.5 * (y1 + y2)) for (x1, y1, x2, y2) in boxes]


def detect_balls_in_image(img):
    """Runs YOLO detection on a single image and returns filtered results."""
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=img,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        classes=[SPORTS_BALL_CLASS_ID],
        imgsz=YOLO_IMGZ,
        max_det=YOLO_MAXD,
        verbose=False,
    )
    boxes, centers, radii = [], [], []
    if len(results) > 0 and results[0].boxes is not None:
        min_r_px = int(MIN_RADIUS_REL * min(img.shape[0], img.shape[1]))
        for b in results[0].boxes:
            box = tuple(b.xyxy.squeeze().tolist())
            radius = est_radius_from_box(box)
            if radius >= min_r_px:
                boxes.append(box)
                radii.append(radius)
                centers.append((0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])))
    return boxes, centers, radii


def detect_pocket_candidates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    h, w = gray.shape[:2]
    rmin = max(10, int(0.018 * min(h, w)))
    rmax = max(rmin + 10, int(0.040 * min(h, w)))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(0.25 * min(h, w)),
        param1=100,
        param2=20,
        minRadius=rmin,
        maxRadius=rmax,
    )
    if circles is None:
        return []
    cand = []
    for x, y, r in np.uint16(np.around(circles[0, :])):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, r - 2), 255, -1)
        if cv2.mean(gray, mask=mask)[0] < 110:
            cand.append((float(x), float(y), float(r)))
    return cand


def pick_nearest(candidates, target_xy):
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (c[0] - target_xy[0]) ** 2 + (c[1] - target_xy[1]) ** 2,
    )


def remove_balls_in_pockets(
    centers, radii, boxes, pockets_list, factor=POCKET_INCLUSION_FACTOR
):
    kept_idx = []
    for i, (cx, cy) in enumerate(centers):
        if not any(
            np.hypot(cx - px, cy - py) <= pr * factor for (px, py, pr) in pockets_list
        ):
            kept_idx.append(i)
    return (
        [boxes[i] for i in kept_idx],
        [centers[i] for i in kept_idx],
        [radii[i] for i in kept_idx],
    )


def classify_white_black_lab(img_bgr, centers, radii):
    """
    Finds white and black balls using the L-channel (Lightness) from the LAB color space.
    This is much more robust to shadows and lighting changes.
    """
    if not centers:
        return None, None
    lab_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab_image[:, :, 0]
    white_idx, max_l = None, -1
    black_idx, min_l = None, 256
    for i, (center, r) in enumerate(zip(centers, radii)):
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros_like(l_channel)
        cv2.circle(mask, (cx, cy), int(r * 0.8), 255, -1)
        mean_l_value = cv2.mean(l_channel, mask=mask)[0]
        if mean_l_value > max_l:
            max_l = mean_l_value
            white_idx = i
        if mean_l_value < min_l:
            min_l = mean_l_value
            black_idx = i
    if white_idx == black_idx:
        return white_idx, None
    return white_idx, black_idx


def save_annotated(img, boxes, types, origin_pocket, out_path):
    ann = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        t = types[i]
        color_map = {
            "white": (255, 255, 0),
            "black": (255, 0, 255),
            "other": (0, 255, 0),
        }
        color = color_map.get(t, (0, 255, 0))
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            ann, t, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    if origin_pocket is not None:
        px, py, pr = origin_pocket
        cv2.circle(ann, (int(px), int(py)), int(pr), (0, 165, 255), 2)
        cv2.putText(
            ann,
            "Origin",
            (int(px) + 4, int(py) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
    out_path = get_output_path(out_path)
    cv2.imwrite(out_path, ann)
