import cv2
from ultralytics import YOLO
import numpy as np
import os
import json

# ====== PATHS (ללא CLI): ======
IMAGE_PATH       = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-11.jpg"
OUTPUT_ANN_PATH  = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/table-11-annotated.jpg"
OUTPUT_JSON_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/analysis-table-11.json"

# ====== YOLO / Hough ======
MODEL_PATH           = "yolov8n.pt"
SPORTS_BALL_CLASS_ID = 32
YOLO_CONF            = 0.004
YOLO_IOU             = 0.40
YOLO_IMGZ            = 1536
YOLO_MAXD            = 300
USE_TTA              = True

# ====== constraints ======
MAX_BALLS              = 16
MIN_RADIUS_REL         = 0.010
MIN_RADIUS_PX_OVERRIDE = None

# “כדור בכיס” — ניפוי דיטקציות שמרכזן בתוך רדיוס כיס *פקטור*
POCKET_INCLUSION_FACTOR = 1.05

# ----- Utils -----
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

def boxes_to_centers(boxes):
    return [(0.5 * (x1 + x2), 0.5 * (y1 + y2)) for (x1, y1, x2, y2) in boxes]

# ----- Detection -----
def yolo_detect(img):
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=img, conf=YOLO_CONF, iou=YOLO_IOU,
        classes=[SPORTS_BALL_CLASS_ID], imgsz=YOLO_IMGZ,
        augment=USE_TTA, max_det=YOLO_MAXD, verbose=False
    )
    boxes_px, scores = [], []
    if len(results) > 0 and results[0].boxes is not None:
        for b in results[0].boxes:
            xyxy = b.xyxy.squeeze().tolist()
            conf = float(b.conf.item())
            boxes_px.append(tuple(xyxy))
            scores.append(conf)
    return boxes_px, scores

def hough_fallback(img, existing_boxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    r_min = max(6, int(0.009 * min(h, w)))
    r_max = max(r_min + 6, int(0.015 * min(h, w)))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(20, r_min*2),
        param1=120, param2=18, minRadius=r_min, maxRadius=r_max
    )
    boxes_px, scores, radii = [], [], []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for x, y, r in circles:
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            if not any(iou_xyxy((x1, y1, x2, y2), bx) > 0.2 for bx in existing_boxes):
                boxes_px.append((float(x1), float(y1), float(x2), float(y2)))
                scores.append(0.50)
                radii.append(float(r))
    return boxes_px, scores, radii

def filter_and_limit(img, boxes, scores, radii=None):
    h, w = img.shape[:2]
    min_r_px = MIN_RADIUS_PX_OVERRIDE if MIN_RADIUS_PX_OVERRIDE else int(MIN_RADIUS_REL * min(h, w))
    kept = []
    for i, (bx, sc) in enumerate(zip(boxes, scores)):
        r = radii[i] if (radii and i < len(radii)) else est_radius_from_box(bx)
        if r >= min_r_px:
            kept.append((bx, sc, r))
    kept.sort(key=lambda t: t[1], reverse=True)
    kept = kept[:MAX_BALLS]
    if radii is None:
        return [k[0] for k in kept], [k[1] for k in kept], [est_radius_from_box(k[0]) for k in kept]
    else:
        return [k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept]

# ----- Pockets: מועמדים + BL בלבד -----
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
        cv2.circle(mask, (x, y), max(1, r-2), 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        if mean_intensity < 110:  # כיסים כהים
            cand.append((float(x), float(y), float(r)))
    return cand

def pick_nearest(candidates, target_xy):
    if not candidates:
        return None
    tx, ty = target_xy
    best, best_d = None, 1e18
    for (x, y, r) in candidates:
        d = (x - tx)**2 + (y - ty)**2
        if d < best_d:
            best, best_d = (x, y, r), d
    return best

# ===================== Classification helpers =====================

def _disk_mask(h, w, cx, cy, r):
    m = np.zeros((h, w), dtype=np.uint8)
    if r >= 1:
        cv2.circle(m, (cx, cy), int(r), 255, -1)
    return m

def _annulus_mask(h, w, cx, cy, r_in, r_out):
    outer = _disk_mask(h, w, cx, cy, r_out)
    inner = _disk_mask(h, w, cx, cy, max(0, r_in))
    return cv2.subtract(outer, inner)

def patch_stats(img_bgr, centers, radii, inner_ratio=0.55, outer_ratio=0.85):
    """סטטיסטיקות טבעת פנימית/חיצונית לכל כדור."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H,S,V = cv2.split(hsv)
    L,A,B = cv2.split(lab)

    h, w = img_bgr.shape[:2]
    out = []
    for (cx_f, cy_f), r_est in zip(centers, radii):
        cx, cy = int(round(cx_f)), int(round(cy_f))
        r_in  = max(2, int(inner_ratio * r_est))
        r_out = max(r_in+1, int(outer_ratio * r_est))
        lim = int(min(cx, cy, w-1-cx, h-1-cy))
        r_out = min(r_out, lim)
        if r_out <= r_in:
            r_in = max(1, r_out-1)

        ann = _annulus_mask(h, w, cx, cy, r_in, r_out)

        # מסכות צבע
        mask_white  = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
        mask_black  = cv2.inRange(hsv, (0, 0, 0),   (180, 80, 70))
        mask_orange = cv2.inRange(hsv, (10, 80, 80), (30, 255, 255))  # H≈10-30°

        tot = cv2.countNonZero(ann) + 1e-6
        frac_white  = cv2.countNonZero(cv2.bitwise_and(mask_white,  mask_white,  mask=ann)) / tot
        frac_black  = cv2.countNonZero(cv2.bitwise_and(mask_black,  mask_black,  mask=ann)) / tot
        frac_orange = cv2.countNonZero(cv2.bitwise_and(mask_orange, mask_orange, mask=ann)) / tot

        # שונות גוון (להעניש מפוספס)
        Hf = H.astype(np.float32)
        ang = Hf * (2*np.pi/180.0)
        hue_var = 1.0 - np.hypot(cv2.mean(np.sin(ang), mask=ann)[0],
                                 cv2.mean(np.cos(ang), mask=ann)[0])

        # בהירות ממוצעת
        mean_V = cv2.mean(V, mask=ann)[0]
        mean_L = cv2.mean(L, mask=ann)[0]

        out.append({
            "frac_white":  float(frac_white),
            "frac_black":  float(frac_black),
            "frac_orange": float(frac_orange),
            "hue_var":     float(hue_var),
            "mean_V":      float(mean_V),
            "mean_L":      float(mean_L)
        })
    return out

# ======= שחור עמיד להיילייטים =======
def find_black_index_robust(img_bgr, centers, radii,
                            inner_ratio=0.55, outer_ratio=0.85,
                            very_dark_V=60,      # פיקסלים כהים ממש (0..255)
                            min_dark_frac=0.08,  # לפחות 8% מהטבעת מאוד כהים
                            w_V10=0.55, w_L10=0.25, w_meanV=0.20, w_chroma=0.35):
    """
    בוחר את הכדור השחור על בסיס:
    - V10 (אחוזון 10 של V) + L10  → עמיד להיילייטים
    - meanV  → תוספת יציבות
    - chroma (Lab) → מענישים צבעים רוויים
    - דרישת סף: אחוז פיקסלים מאוד כהים ≥ min_dark_frac
    מחזיר: (black_idx, scores, details)
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV); H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB); L, A, B = cv2.split(lab)

    scores = []
    details = []
    for (cx_f, cy_f), r_est in zip(centers, radii):
        cx, cy = int(round(cx_f)), int(round(cy_f))
        r_in  = max(2, int(inner_ratio * r_est))
        r_out = max(r_in+1, int(outer_ratio * r_est))
        lim   = int(min(cx, cy, w-1-cx, h-1-cy))
        r_out = min(r_out, lim)
        if r_out <= r_in or r_out < 3:
            scores.append(+1e9); details.append({}); continue

        ann = _annulus_mask(h, w, cx, cy, r_in, r_out)
        v_vals = V[ann.astype(bool)]
        l_vals = L[ann.astype(bool)]
        a_vals = (A[ann.astype(bool)].astype(np.float32) - 128.0)
        b_vals = (B[ann.astype(bool)].astype(np.float32) - 128.0)
        chroma = float(np.hypot(a_vals.mean() if a_vals.size else 0.0,
                                b_vals.mean() if b_vals.size else 0.0))

        if v_vals.size == 0:
            scores.append(+1e9); details.append({}); continue

        V10   = float(np.percentile(v_vals, 10))
        L10   = float(np.percentile(l_vals, 10)) if l_vals.size else 255.0
        meanV = float(v_vals.mean())

        dark_frac = float((v_vals < very_dark_V).sum()) / float(len(v_vals))

        score = (w_V10 * (V10/255.0)) + (w_L10 * (L10/255.0)) \
                + (w_meanV * (meanV/255.0)) + (w_chroma * min(chroma/80.0, 1.5))

        if dark_frac < min_dark_frac:
            score += 0.6  # ענישה חזקה

        scores.append(score)
        details.append({"V10": V10, "L10": L10, "meanV": meanV,
                        "chroma": chroma, "dark_frac": dark_frac,
                        "r_est": r_est})

    if not scores:
        return None, [], []

    black_idx = int(np.argmin(scores))
    return black_idx, scores, details

def classify_white_black_orange(img_bgr, centers, radii):
    stats = patch_stats(img_bgr, centers, radii)

    white_scores = []
    orange_scores = []
    for st in stats:
        ws = st["frac_white"]  - 0.35*st["hue_var"]   # לבן: הרבה לבן ומעט שונות גוון
        os = st["frac_orange"]                        # כתום: יחס כתום בטבעת
        white_scores.append(ws)
        orange_scores.append(os)

    white_idx  = int(np.argmax(white_scores)) if white_scores else None
    orange_idx = int(np.argmax(orange_scores)) if orange_scores else None

    # --- שחור: מדד כהות עמיד ---
    black_idx, dark_scores, _details = find_black_index_robust(img_bgr, centers, radii)

    # אם לבן ושחור אותו אינדקס, קח את הלבן הבא בתור
    order_w = list(np.argsort(-np.array(white_scores))) if white_scores else []
    if white_idx is not None and black_idx is not None and white_idx == black_idx and len(order_w) > 1:
        white_idx = int(order_w[1])

    return white_idx, black_idx, orange_idx, stats, (white_scores, dark_scores, orange_scores)

# ----- הוספת דיטקציות כתומות אם חסר -----
def add_missing_orange_candidates(img, existing_boxes, min_radius_px, max_radius_px):
    """מסנן כתום בכל התמונה ומוצא עיגולים חדשים שאינם חופפים לדיטקציות קיימות."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (10, 90, 90), (30, 255, 255))  # כתום חזק יותר
    mask1 = cv2.medianBlur(mask1, 5)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)

    circles = cv2.HoughCircles(mask1, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(min_radius_px*2.0),
                               param1=100, param2=15, minRadius=int(min_radius_px), maxRadius=int(max_radius_px))
    added_boxes, added_scores, added_r = [], [], []
    if circles is not None:
        for x, y, r in np.uint16(np.around(circles[0, :])):
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            if not any(iou_xyxy((x1, y1, x2, y2), bx) > 0.25 for bx in existing_boxes):
                added_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                added_scores.append(0.45)   # ציון בינוני
                added_r.append(float(r))
    return added_boxes, added_scores, added_r

# ----- ניפוי כדורים בתוך כיסים -----
def remove_balls_in_pockets(centers, radii, pockets_list, factor=POCKET_INCLUSION_FACTOR):
    kept_idx = []
    for i, (cx, cy) in enumerate(centers):
        inside = False
        for (px, py, pr) in pockets_list:
            if np.hypot(cx - px, cy - py) <= pr * factor:
                inside = True
                break
        if not inside:
            kept_idx.append(i)
    return kept_idx

# ----- Annotated -----
def save_annotated(img, boxes, types, origin_pocket, out_path):
    ann = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        t = types[i]
        if t == "white":      color = (255,255,0)
        elif t == "black":    color = (255,0,255)
        elif t == "orange":   color = (0,140,255)  # BGR כתום
        else:                 color = (0,255,0)
        cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
        cv2.putText(ann, t, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if origin_pocket is not None:
        px, py, pr = origin_pocket
        cv2.circle(ann, (int(px), int(py)), int(pr), (0,165,255), 2)
        cv2.putText(ann, "Origin", (int(px)+4, int(py)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    cv2.imwrite(out_path, ann)

# ===================== MAIN =====================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    H, W = img.shape[:2]

    # 1) detect balls
    yolo_boxes, yolo_scores = yolo_detect(img)
    h_boxes, h_scores, h_radii = hough_fallback(img, yolo_boxes)
    all_boxes  = yolo_boxes + h_boxes
    all_scores = yolo_scores + h_scores
    all_radii  = [est_radius_from_box(b) for b in yolo_boxes] + h_radii

    # 1.1 הוספת מועמדים כתומים אם חסר
    min_r = int(max(6, 0.009 * min(H, W)))
    max_r = int(max(min_r + 6, 0.018 * min(H, W)))
    add_boxes, add_scores, add_r = add_missing_orange_candidates(img, all_boxes, min_r, max_r)
    all_boxes  += add_boxes
    all_scores += add_scores
    all_radii  += add_r

    f_boxes, f_scores, f_radii = filter_and_limit(img, all_boxes, all_scores, all_radii)
    centers = boxes_to_centers(f_boxes)

    # 2) כיסים מועמדים + בחירת BL כ-Origin (fallback לפינת התמונה)
    pockets_cand = detect_pocket_candidates(img)
    bl_guess = pick_nearest(pockets_cand, (0.0 * W, 1.0 * H))
    if bl_guess is None:
        blx, bly, blr = 0.0, float(H - 1), 0.03 * min(W, H)
    else:
        blx, bly, blr = map(float, bl_guess)

    # 3) הסרת כדורים שנמצאים בתוך כיסים
    if pockets_cand:
        keep_idx = remove_balls_in_pockets(centers, f_radii, pockets_cand, factor=POCKET_INCLUSION_FACTOR)
        f_boxes  = [f_boxes[i]  for i in keep_idx]
        f_scores = [f_scores[i] for i in keep_idx]
        f_radii  = [f_radii[i]  for i in keep_idx]
        centers  = [centers[i]  for i in keep_idx]

    # 4) סיווג: לבן/שחור/כתום/אחר
    w_idx, b_idx, o_idx, stats, _ = classify_white_black_orange(img, centers, f_radii)

    types = []
    for i in range(len(centers)):
        if i == w_idx:
            types.append("white")
        elif i == b_idx:
            types.append("black")
        elif i == o_idx and stats[i]["frac_orange"] > 0.25:  # מסנן מינימלי לכתום
            types.append("orange")
        else:
            types.append("other")

    # 5) JSON (קואורדינטות יחסית ל-BL, y למעלה)
    balls_json = []
    for i, ((cx, cy), t) in enumerate(zip(centers, types)):
        balls_json.append({
            "index": i,
            "type": t,
            "x_px": float(cx - blx),
            "y_px": float(bly - cy)
        })

    result = {
        "image_path": IMAGE_PATH,
        "origin_px": {"x": float(blx), "y": float(bly)},
        "pockets_px": {"0": {"x": float(blx), "y": float(bly), "r": float(blr)}},  # BL בלבד
        "table_size_px": {"width_px": float(W), "height_px": float(H)},
        "balls": balls_json
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    save_annotated(img, f_boxes, types, (blx, bly, blr), OUTPUT_ANN_PATH)
    print(f"[OK] Analysis JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"[OK] Annotated image saved to: {OUTPUT_ANN_PATH}")

if __name__ == "__main__":
    main()
