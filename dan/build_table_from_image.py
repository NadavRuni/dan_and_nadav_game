import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# ===== Make local imports work no matter where you run from =====
import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ===============================================================

# ==== import your game classes & constants ====
from const_numbers import TABLE_LENGTH, TABLE_WIDTH, BALL_RADIUS
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_draw import draw_table
# ==============================================

# --------------- CONFIG ----------------
IMAGE_PATH  = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/table-10-annotated.jpg"
OUTPUT_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/table-10-annotated.jpg"  # annotated image
MODEL_PATH = "yolov8n.pt"
SPORTS_BALL_CLASS_ID = 32
YOLO_CONF = 0.004
YOLO_IOU  = 0.40
YOLO_IMGZ = 1536
YOLO_MAXD = 300
USE_TTA   = True

MAX_BALLS = 16
MIN_RADIUS_REL = 0.009      # min ball radius as fraction of min(image h,w)
MIN_RADIUS_PX_OVERRIDE = None  # e.g. 12 to enforce absolute min radius
# --------------------------------------

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

def centers_from_boxes(boxes):
    return [((x1+x2)/2.0, (y1+y2)/2.0) for (x1,y1,x2,y2) in boxes]

def yolo_detect_balls(img):
    model = YOLO(MODEL_PATH)
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
    boxes, scores = [], []
    if len(results) and results[0].boxes is not None:
        for b in results[0].boxes:
            boxes.append(tuple(b.xyxy.squeeze().tolist()))
            scores.append(float(b.conf.item()))
    return boxes, scores

def hough_balls_fallback(img, existing_boxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    r_min = max(6, int(MIN_RADIUS_REL * min(h, w)))
    r_max = max(r_min + 6, int(0.015 * min(h, w)))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(20, r_min*2),
        param1=120, param2=18, minRadius=r_min, maxRadius=r_max
    )
    boxes, scores, radii = [], [], []
    if circles is not None:
        for x, y, r in np.uint16(np.around(circles[0, :])):
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            if not any(iou_xyxy((x1, y1, x2, y2), bx) > 0.2 for bx in existing_boxes):
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
                scores.append(0.50)
                radii.append(float(r))
    return boxes, scores, radii

def filter_and_cap_balls(img, boxes, scores, radii=None):
    h, w = img.shape[:2]
    min_r_px = MIN_RADIUS_PX_OVERRIDE if MIN_RADIUS_PX_OVERRIDE else int(MIN_RADIUS_REL * min(h, w))
    kept = []
    for i, (bx, sc) in enumerate(zip(boxes, scores)):
        r = radii[i] if (radii is not None and i < len(radii)) else est_radius_from_box(bx)
        if r >= min_r_px:
            kept.append((bx, sc, r))
    kept.sort(key=lambda t: t[1], reverse=True)
    kept = kept[:MAX_BALLS]
    out_boxes = [k[0] for k in kept]
    out_scores = [k[1] for k in kept]
    out_radii = [k[2] for k in kept]
    return out_boxes, out_scores, out_radii

# ---------- Pocket detection & mapping ----------
def detect_pocket_candidates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    h, w = gray.shape[:2]
    rmin = max(10, int(0.018 * min(h, w)))
    rmax = max(rmin + 10, int(0.040 * min(h, w)))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(0.25*min(h,w)),
        param1=100, param2=20, minRadius=rmin, maxRadius=rmax
    )
    if circles is None:
        return []
    cand = []
    for x, y, r in np.uint16(np.around(circles[0, :])):
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(max(1, r-2)), 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        if mean_intensity < 110:
            cand.append((float(x), float(y), float(r)))
    return cand

def assign_pockets(img, candidates):
    if len(candidates) < 4:
        raise ValueError("Not enough pocket candidates detected.")
    h, w = img.shape[:2]
    def score_bottom_left(c):  return ( c[0]/w) - ( c[1]/h)
    def score_bottom_right(c): return (-c[0]/w) - ( c[1]/h)
    def score_top_left(c):     return ( c[0]/w) + ( c[1]/h)
    def score_top_right(c):    return (-c[0]/w) + ( c[1]/h)
    def is_bottom(c):          return c[1] > 0.7*h
    def is_top(c):             return c[1] < 0.3*h

    bl = min(candidates, key=score_bottom_left)
    br = min(candidates, key=score_bottom_right)
    tl = min(candidates, key=score_top_left)
    tr = min(candidates, key=score_top_right)

    mid_bottom_cands = [c for c in candidates if is_bottom(c)]
    mid_top_cands    = [c for c in candidates if is_top(c)]
    if not mid_bottom_cands: mid_bottom_cands = candidates
    if not mid_top_cands:    mid_top_cands    = candidates
    mid_bottom = min(mid_bottom_cands, key=lambda c: abs(c[0] - w/2))
    mid_top    = min(mid_top_cands,    key=lambda c: abs(c[0] - w/2))

    # deduplicate close ones
    chosen = [bl, br, tr, tl, mid_bottom, mid_top]
    uniq = []
    for c in chosen:
        if all(np.hypot(c[0]-u[0], c[1]-u[1]) > 0.08*min(w,h) for u in uniq):
            uniq.append(c)
    if len(uniq) < 4:
        raise ValueError("Pocket assignment ambiguous; not enough unique pockets.")

    return {0: bl, 1: br, 2: tr, 3: tl, 4: mid_bottom, 5: mid_top}

# ---------- Brightness-based white/black classification ----------
def ball_color_features(img_bgr, centers, radii_est, inner_scale=0.6):
    """
    Extract per-ball features on a small inner disk to avoid edges/highlights:
    - mean V and S (HSV)
    - mean L,a,b and chroma C = sqrt(a^2 + b^2) (Lab)
    Returns list of dicts: [{"V":..,"S":..,"L":..,"C":..}, ...]
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    feats = []
    for (cx, cy), r_est in zip(centers, radii_est):
        r = max(3, int(inner_scale * r_est))
        x, y = int(round(cx)), int(round(cy))
        # keep mask inside image
        r = int(min(r, x, y, w-1-x, h-1-y)) if (0 <= x < w and 0 <= y < h) else 0
        if r < 3:
            feats.append({"V":0.0,"S":255.0,"L":0.0,"C":128.0})
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        # Optional: erode a bit to stay away from edges
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)

        mean_V = cv2.mean(V, mask=mask)[0]
        mean_S = cv2.mean(S, mask=mask)[0]
        mean_L = cv2.mean(L, mask=mask)[0]
        mean_A = cv2.mean(A, mask=mask)[0] - 128.0
        mean_B = cv2.mean(B, mask=mask)[0] - 128.0
        chroma = float(np.hypot(mean_A, mean_B))  # ~0=achromatic (white/gray/black)

        feats.append({"V": mean_V, "S": mean_S, "L": mean_L, "C": chroma})
    return feats

def assign_white_black_robust(features):
    """
    White: high L / V, low S and low chroma C (achromatic & bright).
    Black: lowest V (or lowest L).
    Returns indices (white_idx, black_idx).
    """
    if not features:
        return None, None

    V = np.array([f["V"] for f in features])   # 0..255
    S = np.array([f["S"] for f in features])   # 0..255
    L = np.array([f["L"] for f in features])   # 0..255
    C = np.array([f["C"] for f in features])   # ~0..~180

    # Normalize to 0..1
    Vn = V / 255.0
    Sn = S / 255.0
    Ln = L / 255.0
    Cn = np.clip(C / 128.0, 0.0, 2.0)

    # Whiteness score: bright (L,V) and low saturation/chroma
    whiteness = 0.55*Ln + 0.35*Vn - 0.5*Sn - 0.45*Cn

    white_idx = int(np.argmax(whiteness))

    # Heuristic guardrails to avoid picking bright yellow as white:
    # if candidate has high S or not bright enough, try next best
    order = list(np.argsort(-whiteness))  # descending
    for idx in order:
        if V[idx] >= 170 and S[idx] <= 80 and C[idx] <= 45:
            white_idx = int(idx)
            break

    # Black = darkest by V
    black_idx = int(np.argmin(Vn))

    # Avoid collision: if same index (pathological), pick next for white
    if white_idx == black_idx and len(order) > 1:
        white_idx = int(order[0] if order[1] == black_idx else order[1])

    return white_idx, black_idx

# ---------- Table builder ----------
def build_table_from_image(image_path: str, save_annotated_to: str = None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # 1) Balls
    y_boxes, y_scores = yolo_detect_balls(img)
    h_boxes, h_scores, h_radii = hough_balls_fallback(img, y_boxes)
    all_boxes = y_boxes + h_boxes
    all_scores = y_scores + h_scores
    all_radii  = [est_radius_from_box(b) for b in y_boxes] + h_radii
    b_boxes, b_scores, b_radii = filter_and_cap_balls(img, all_boxes, all_scores, all_radii)
    centers_px = centers_from_boxes(b_boxes)

    # 2) Pockets & scaling
    pockets_px = assign_pockets(img, detect_pocket_candidates(img))
    blx, bly, _ = pockets_px[0]
    brx, bry, _ = pockets_px[1]
    tlx, tly, _ = pockets_px[3]
    width_px  = abs(blx - brx)
    height_px = abs(bly - tly)
    if width_px < 5 or height_px < 5:
        raise ValueError("Pocket distances too small; scaling failed.")
    sx = TABLE_LENGTH / width_px
    sy = TABLE_WIDTH  / height_px

    # 3) Brightness → white/black classification
    bright = ball_color_features(img, centers_px, b_radii, inner_scale=0.6)
    w_idx, bk_idx = assign_white_black_robust(bright)

    # 4) Build Ball objects with proper ids/types
    balls = []
    used_ids = set()
    for i, ((cx, cy), r_est) in enumerate(zip(centers_px, b_radii)):
        rel_x_px = cx - blx
        rel_y_px = bly - cy
        x_game = max(BALL_RADIUS, min(TABLE_LENGTH - BALL_RADIUS, rel_x_px * sx))
        y_game = max(BALL_RADIUS, min(TABLE_WIDTH  - BALL_RADIUS, rel_y_px * sy))

        if i == w_idx:
            btype, bid = "white", 0
        elif i == bk_idx:
            btype, bid = "black", 8
        else:
            btype, bid = "solid", None

        # assign remaining ids 1..7,9..15 in order
        if bid is None:
            for candidate in list(range(1, 8)) + list(range(9, 16)):
                if candidate not in used_ids:
                    bid = candidate
                    break

        used_ids.add(bid)
        balls.append(Ball(ball_id=bid, x_cord=x_game, y_cord=y_game, ball_type=btype, radius=BALL_RADIUS))

    # 5) Build table
    table = Table(TABLE_LENGTH, TABLE_WIDTH, balls)

    # 6) Optional: save annotated image with colored boxes per type
    if save_annotated_to is not None:
        ann = img.copy()
        # colors: white = cyan, black = magenta, others = green
        for i, (box, (cx, cy)) in enumerate(zip(b_boxes, centers_px)):
            x1, y1, x2, y2 = map(int, box)
            if i == w_idx:
                color = (255, 255, 0)   # cyan (BGR)
                label = "white"
            elif i == bk_idx:
                color = (255,   0, 255) # magenta (BGR)
                label = "black"
            else:
                color = (0, 255, 0)     # green
                label = "ball"
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            cv2.putText(ann, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(save_annotated_to, ann)
        print(f"Annotated image saved to: {save_annotated_to}")

    # 7) Print summary JSON
    out = {
        "origin_px": {"x": blx, "y": bly},
        "scale": {"sx": sx, "sy": sy},
        "classified": {"white_index": int(w_idx) if w_idx is not None else None,
                       "black_index": int(bk_idx) if bk_idx is not None else None},
        "balls": [{"id": b.id, "type": b.type, "x": b.x_cord, "y": b.y_cord} for b in sorted(table.balls, key=lambda b: b.id)]
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return table

if __name__ == "__main__":
    tbl = build_table_from_image(IMAGE_PATH, save_annotated_to=OUTPUT_PATH)
    # Draw using your drawer (colors are derived from Ball.type)
    draw_table(tbl)
