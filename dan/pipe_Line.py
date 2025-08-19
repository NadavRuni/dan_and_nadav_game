import cv2
from ultralytics import YOLO
import numpy as np
import os, json

# ====== PATHS ======
IMAGE_PATH       = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-13.jpg"
OUTPUT_ANN_PATH  = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/table-13-annotated.jpg"
OUTPUT_JSON_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/analysis-table-13.json"

# Debug stage outputs (לא חובה, אבל נוח לראות)
OUT_DIR          = os.path.dirname(OUTPUT_ANN_PATH) or "."
OUTPUT_WHITE_ANN = os.path.join(OUT_DIR, "stage_white.jpg")
OUTPUT_BLACK_ANN = os.path.join(OUT_DIR, "stage_black.jpg")

# ====== YOLO / Hough ======
MODEL_PATH           = "yolov8n.pt"
SPORTS_BALL_CLASS_ID = 32
YOLO_CONF            = 0.01
YOLO_IOU             = 0.40
YOLO_IMGZ            = 1536
YOLO_MAXD            = 300
USE_TTA              = True

# ====== constraints ======
MAX_BALLS              = 16
MIN_RADIUS_REL         = 0.010
MIN_RADIUS_PX_OVERRIDE = None

POCKET_INCLUSION_FACTOR = 1.05  # “כדור בכיס” — ניפוי דיטקציות שמרכזן בתוך רדיוס כיס *פקטור*


# ---------------- Utils ----------------
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

def make_felt_mask(img_bgr):
    """מסכת משטח הלבד (כחול/ירוק) ב-HSV, עם ניקוי מורפולוגי."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # טווחים סלחניים לכחול/ירוק של שולחן ביליארד
    mask_green = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255))
    mask_blue  = cv2.inRange(hsv, (85, 30, 30), (135, 255, 255))
    mask = cv2.bitwise_or(mask_green, mask_blue)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
    return mask




def est_radius_from_box(box):
    x1, y1, x2, y2 = box
    return 0.5 * min(max(0, x2 - x1), max(0, y2 - y1))

def boxes_to_centers(boxes):
    return [(0.5 * (x1 + x2), 0.5 * (y1 + y2)) for (x1, y1, x2, y2) in boxes]

def draw_rect_label(img, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ---------------- Detectors ----------------
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
    r_max = max(r_min + 6, int(0.022 * min(h, w)))  # קצת יותר רחב
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


# ---------------- Pockets ----------------
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


    


# =========================================================
#                    PIPELINE STAGES
# =========================================================
# Stage object structure:
# {
#   "img": <BGR image>,
#   "white": {"center":(x,y),"radius":r,"box":(x1,y1,x2,y2)} | None,
#   "black": {"center":(x,y),"radius":r,"box":(x1,y1,x2,y2)} | None,
# }

def white_recognizer(stage):
    img = stage["img"]
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # זיהוי עיגולים בטווח רדיוסים סביר
    r_min = max(6, int(0.009 * min(h, w)))
    r_max = max(r_min + 6, int(0.022 * min(h, w)))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(24, r_min*2),
        param1=100, param2=18, minRadius=r_min, maxRadius=r_max
    )

    best = None
    best_score = -1e9
    border_margin = 22

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        H,S,V = cv2.split(hsv)
        for x, y, r in circles:
            if x < border_margin or x > w - border_margin or y < border_margin or y > h - border_margin:
                continue
            # מדד "לובן": V גבוה, S נמוך
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.circle(mask, (x,y), r, 255, -1)
            mean_S = cv2.mean(S, mask=mask)[0]
            mean_V = cv2.mean(V, mask=mask)[0]
            frac_bright = cv2.countNonZero(cv2.inRange(hsv, (0,0,200), (180,60,255)) & mask) / (np.pi*r*r + 1e-6)
            score = (mean_V/255.0) - 0.6*(mean_S/255.0) + 0.4*frac_bright
            if score > best_score:
                best_score = score
                box = (float(x-r), float(y-r), float(x+r), float(y+r))
                best = {"center":(float(x),float(y)), "radius":float(r), "box":box}

    # כתוב קובץ debug
    dbg = img.copy()
    if best:
        draw_rect_label(dbg, best["box"], "white", (255,255,0))
    cv2.imwrite(OUTPUT_WHITE_ANN, dbg)

    stage["white"] = best
    return stage


def black_recognizer(stage):
    img = stage["img"]
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=50,
        param1=50, param2=30,
        minRadius=10, maxRadius=max(50, int(0.03*min(h,w)))
    )

    best = None
    best_intensity = float('inf')
    border_margin = 30

    if circles is not None:
        arr = np.round(circles[0, :]).astype("int")
        radii = arr[:,2]
        median_r = int(np.median(radii)) if len(radii) else 0

        for (x, y, r) in arr:
            if median_r and abs(r - median_r) > 5:
                continue
            if x < border_margin or x > w - border_margin or y < border_margin or y > h - border_margin:
                continue

            mask = np.zeros_like(gray)
            cv2.circle(mask, (x,y), r, 255, -1)
            mean_bgr = cv2.mean(img, mask=mask)[:3]
            intensity = float(sum(mean_bgr))
            if intensity < 20:  # מאוד כהה -> כנראה כיס
                continue

            if intensity < best_intensity:
                best_intensity = intensity
                box = (float(x-r), float(y-r), float(x+r), float(y+r))
                best = {"center":(float(x),float(y)), "radius":float(r), "box":box}

    # Debug draw
    dbg = img.copy()
    if best:
        draw_rect_label(dbg, best["box"], "black", (255,0,255))
    cv2.imwrite(OUTPUT_BLACK_ANN, dbg)

    stage["black"] = best
    return stage


def image_recognizer(stage):
    """
    מזהה את כל הכדורים (YOLO + Hough fallback), מסיר כיסים,
    מסמן WHITE/BLACK לפי מה שהוגדר בשלבים הקודמים. אין טיפול מיוחד לכתום.
    """
    img = stage["img"]
    felt_mask = make_felt_mask(img)
    H, W = img.shape[:2]

    # 1) detect balls
    yolo_boxes, yolo_scores = yolo_detect(img)
    h_boxes, h_scores, h_radii = hough_fallback(img, yolo_boxes)
    all_boxes  = yolo_boxes + h_boxes
    all_scores = yolo_scores + h_scores
    all_radii  = [est_radius_from_box(b) for b in yolo_boxes] + h_radii

    # 2) filter
    f_boxes, f_scores, f_radii = filter_and_limit(img, all_boxes, all_scores, all_radii)
    centers = boxes_to_centers(f_boxes)

    # 3) pockets + BL origin (רק BL בתור origin)
    pockets_cand = detect_pocket_candidates(img)
    bl_guess = None
    if pockets_cand:
        bl_guess = min(pockets_cand, key=lambda p: (p[0]-0)**2 + (p[1]-H)**2)
    if bl_guess is None:
        blx, bly, blr = 0.0, float(H - 1), 0.03 * min(W, H)
    else:
        blx, bly, blr = map(float, bl_guess)

    # הסר כדורים שהמרכז שלהם בתוך כיס
    if pockets_cand:
        keep_idx = remove_balls_in_pockets(centers, f_radii, pockets_cand, factor=POCKET_INCLUSION_FACTOR)
        f_boxes  = [f_boxes[i]  for i in keep_idx]
        f_scores = [f_scores[i] for i in keep_idx]
        f_radii  = [f_radii[i]  for i in keep_idx]
        centers  = [centers[i]  for i in keep_idx]

    # 4) קבע אינדקסים ל־WHITE/BLACK לפי הקרוב ביותר למרכזים שזוהו בשלבים הקודמים
    forced_w_idx = forced_b_idx = None
    if stage.get("white") and centers:
        wx, wy = stage["white"]["center"]
        forced_w_idx = int(np.argmin([(cx-wx)**2 + (cy-wy)**2 for (cx,cy) in centers]))
    if stage.get("black") and centers:
        bx, by = stage["black"]["center"]
        forced_b_idx = int(np.argmin([(cx-bx)**2 + (cy-by)**2 for (cx,cy) in centers]))

    # אם התנגשו, נשמור שחור ונחליף לבן לאלטרנטיבי הקרוב הבא
    if forced_w_idx is not None and forced_b_idx is not None and forced_w_idx == forced_b_idx and len(centers) > 1:
        dists = [(i, (centers[i][0]-stage["white"]["center"][0])**2 + (centers[i][1]-stage["white"]["center"][1])**2) for i in range(len(centers))]
        dists.sort(key=lambda t: t[1])
        for i, _ in dists:
            if i != forced_b_idx:
                forced_w_idx = i
                break

    # 5) types: רק white/black/other (אין orange)
    types = []
    for i in range(len(centers)):
        if forced_w_idx is not None and i == forced_w_idx:
            types.append("white")
        elif forced_b_idx is not None and i == forced_b_idx:
            types.append("black")
        else:
            types.append("other")

    # 6) שמירה + JSON
    ann = img.copy()
    for i, b in enumerate(f_boxes):
        t = types[i]
        color = (0,255,0)
        if t == "white": color = (255,255,0)
        elif t == "black": color = (255,0,255)
        draw_rect_label(ann, b, t, color)
    cv2.circle(ann, (int(blx), int(bly)), int(blr), (0,165,255), 2)
    cv2.putText(ann, "Origin", (int(blx)+4, int(bly)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    cv2.imwrite(OUTPUT_ANN_PATH, ann)

    balls_json = []
    for i, (cx, cy) in enumerate(centers):
        balls_json.append({
            "index": i,
            "type": types[i],
            "x_px": float(cx - blx),
            "y_px": float(bly - cy)
        })

    result = {
        "image_path": IMAGE_PATH,
        "origin_px": {"x": float(blx), "y": float(bly)},
        "pockets_px": {"0": {"x": float(blx), "y": float(bly), "r": float(blr)}},
        "table_size_px": {"width_px": float(W), "height_px": float(H)},
        "balls": balls_json
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] White stage annotated: {OUTPUT_WHITE_ANN}")
    print(f"[OK] Black stage annotated: {OUTPUT_BLACK_ANN}")
    print(f"[OK] Final annotated:      {OUTPUT_ANN_PATH}")
    print(f"[OK] Analysis JSON saved:  {OUTPUT_JSON_PATH}")

    return stage


# =========================================================
#                         MAIN
# =========================================================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # input → white → black → image_recognizer → output
    stage0 = {"img": img, "white": None, "black": None}
    stage1 = white_recognizer(stage0)
    stage2 = black_recognizer(stage1)
    _      = image_recognizer(stage2)

if __name__ == "__main__":
    main()
