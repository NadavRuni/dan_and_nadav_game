import cv2
from ultralytics import YOLO
import numpy as np
import os, json

# ====== PATHS ======
IMAGE_PATH       = "photos/img_start7.jpeg"
OUTPUT_ANN_PATH  = "photos/output/img_output.jpg"
OUTPUT_JSON_PATH = "photos/output/img_JSON.json"

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

POCKET_INCLUSION_FACTOR = 1.05  


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

# ---------------- memory ----------------
def nms_with_tags(boxes, scores, radii, tags, iou_thr=0.45):
    """NMS שמשמר תגיות 'זיכרון' (mem_white/mem_black)."""
    if not boxes:
        return [], [], [], []
    order = np.argsort(scores)[::-1]
    used = np.zeros(len(order), dtype=bool)

    kept_boxes, kept_scores, kept_radii, kept_tags = [], [], [], []
    while True:
        valid = np.where(~used)[0]
        if valid.size == 0:
            break
        i = int(order[valid[0]])
        used[valid[0]] = True

        kb, ks = boxes[i], scores[i]
        kr = radii[i] if (radii and i < len(radii) and radii[i] is not None) else est_radius_from_box(kb)
        kt = set(tags[i]) if tags and i < len(tags) else set()

        # דחיית חופפים + איסוף תגיות מהם
        for idx in valid[1:]:
            j = int(order[idx])
            if iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                used[idx] = True
                if tags and j < len(tags):
                    kt |= set(tags[j]) if isinstance(tags[j], (set, list, tuple)) else ({tags[j]} if tags[j] else set())

        kept_boxes.append(kb)
        kept_scores.append(ks)
        kept_radii.append(kr)
        kept_tags.append(kt)

    return kept_boxes, kept_scores, kept_radii, kept_tags


def inject_memory_candidate(mem, all_boxes, all_scores, all_radii, all_tags, boost_score, tag_name):
    """אם יש זיכרון (box,radius), ננסה לתייג דיטקציה קיימת; אחרת נוסיף חדשה עם ציון גבוה."""
    if not mem:
        return
    box = mem["box"]; r = mem["radius"]
    # נסה לתייג קופסה קיימת
    for j, bx in enumerate(all_boxes):
        if iou_xyxy(box, bx) > 0.30:
            all_scores[j] = max(all_scores[j], boost_score)
            all_tags[j].add(tag_name)
            return
    # לא נמצאה חופפת — הוסף חדשה
    all_boxes.append(box)
    all_scores.append(boost_score)
    all_radii.append(r)
    all_tags.append({tag_name})






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


def assign_six_pockets(pockets_cand, W, H):
    """
    מכניס את המועמדים שזוהו לכיסים: TL, TM, TR, BL, BM, BR
    לפי הקרוב ביותר לעוגני הפינות/האמצעים. מחזיר dict עם נקודות (x,y) או None אם חסר.
    """
    if not pockets_cand:
        return None
    anchors = {
        "TL": (0.00*W, 0.00*H),
        "TM": (0.50*W, 0.00*H),
        "TR": (1.00*W, 0.00*H),
        "BL": (0.00*W, 1.00*H),
        "BM": (0.50*W, 1.00*H),
        "BR": (1.00*W, 1.00*H),
    }
    taken = set()
    out = {}
    for name, (ax, ay) in anchors.items():
        best, best_d = None, 1e18
        best_k = None
        for k, (x, y, r) in enumerate(pockets_cand):
            if k in taken:
                continue
            d = (x-ax)**2 + (y-ay)**2
            if d < best_d:
                best, best_d, best_k = (float(x), float(y)), d, k
        if best is None:
            return None
        out[name] = best
        taken.add(best_k)
    return out


def build_img2table_h(pmap, table_w=2.0, table_h=1.0):
    """
    בונה הומוגרפיה מהתמונה -> מערכת שולחן (2:1).
    pmap: dict עם TL,TM,TR,BL,BM,BR (כל אחת (x,y) בתמונה).
    מחזיר H_img2tab (3x3) והמידות (table_w, table_h).
    """
    # נקודות מקור (תמונה)
    src = np.array([
        pmap["TL"], pmap["TM"], pmap["TR"],
        pmap["BL"], pmap["BM"], pmap["BR"]
    ], dtype=np.float32)

    # נקודות יעד (שולחן ישר, יחס 2:1)
    dst = np.array([
        [0.0*table_w, 0.0*table_h],   # TL
        [0.5*table_w, 0.0*table_h],   # TM
        [1.0*table_w, 0.0*table_h],   # TR
        [0.0*table_w, 1.0*table_h],   # BL
        [0.5*table_w, 1.0*table_h],   # BM
        [1.0*table_w, 1.0*table_h],   # BR
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H, (table_w, table_h)

def warp_points_xy(points, H):
    """
    ממפה רשימת נקודות [(x,y),...] באמצעות הומוגרפיה H.
    מחזיר np.array Nx2 במערכת היעד.
    """
    if H is None or len(points) == 0:
        return np.zeros((0,2), dtype=np.float32)
    pts = np.array(points, dtype=np.float32).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    return warped







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



def keep_by_indices(seq, indices):
    return [seq[i] for i in indices]




def disk_overlap_frac_with_mask(H, W, cx, cy, r, mask):
    """
    מחזירה את אחוז שטח הדיסק (כדור) שנופל בתוך המסכה (mask>0).
    H,W – גודל התמונה; cx,cy,r – מרכז/רדיוס הכדור בפיקסלים; mask – מסכת uint8.
    """
    cx = int(round(cx)); cy = int(round(cy))
    r  = int(max(2, round(r)))

    # חלון חיתוך סביב הדיסק
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(W, cx + r + 1), min(H, cy + r + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    submask = mask[y1:y2, x1:x2]

    # מסכת דיסק לוגית בתוך חלון החיתוך
    yy, xx = np.ogrid[y1:y2, x1:x2]
    circle = ((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)) <= (r * r)

    area = int(circle.sum())
    if area == 0:
        return 0.0

    hit = (submask > 0) & circle
    return float(hit.sum()) / float(area)




def remove_in_pockets_protected(centers, radii, pockets, tags, factor):
    keep = []
    for i, (cx, cy) in enumerate(centers):
        protected = ("mem_white" in tags[i]) or ("mem_black" in tags[i])
        if protected:
            keep.append(i); continue
        inside = False
        for (px, py, pr) in pockets:
            if np.hypot(cx - px, cy - py) <= pr * factor:
                inside = True; break
        if not inside:
            keep.append(i)
    return keep

def felt_filter_protected(H, W, centers, radii, tags, felt_mask, min_frac=0.55, min_frac_in=0.35, erode_px=3, dilate_px=2):
    k1 = 2*erode_px + 1; k2 = 2*dilate_px + 1
    felt_eroded = cv2.erode(felt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)), 1)
    felt_dilated= cv2.dilate(felt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)), 1)

    def frac(mask, cx, cy, r):
        cx, cy, r = float(cx), float(cy), float(r)
        return disk_overlap_frac_with_mask(H, W, cx, cy, r, mask)

    keep = []
    dropped = []
    for i, (cx, cy) in enumerate(centers):
        if ("mem_white" in tags[i]) or ("mem_black" in tags[i]):
            keep.append(i); continue
        r = radii[i]
        f  = frac(felt_mask,  cx, cy, r)
        fi = frac(felt_eroded, cx, cy, r)
        if f >= min_frac or fi >= min_frac_in or (felt_dilated[int(round(cy)), int(round(cx))] > 0):
            keep.append(i)
        else:
            dropped.append(i)

    # Rescue רך
    for i in dropped:
        if len(keep) >= len(centers):
            break
        if disk_overlap_frac_with_mask(H, W, centers[i][0], centers[i][1], radii[i], felt_dilated) >= 0.25:
            keep.append(i)

    return sorted(set(keep))




def add_missing_orange_candidates(img, existing_boxes, min_radius_px, max_radius_px):
    """
    השלמת עיגולים חסרים באמצעות פילטר כתום + Hough.
    שימו לב: זה רק מוסיף מועמדים לרשימה (בסוף כולם יסווגו white/black/other).
    מחזיר: added_boxes, added_scores, added_r
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # כתום חזק (עובד טוב על לבד כחול/ירוק). אפשר להתאים לפי הצורך.
    mask = cv2.inRange(hsv, (10, 90, 90), (30, 255, 255))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(max(12, min_radius_px * 2.0)),
        param1=100,
        param2=15,
        minRadius=int(min_radius_px),
        maxRadius=int(max_radius_px)
    )

    added_boxes, added_scores, added_r = [], [], []
    if circles is not None:
        for x, y, r in np.uint16(np.around(circles[0, :])):
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            # אל תוסיף אם יש חפיפה משמעותית עם דיטקציה קיימת
            if any(iou_xyxy((x1, y1, x2, y2), bx) > 0.25 for bx in existing_boxes):
                continue
            added_boxes.append((float(x1), float(y1), float(x2), float(y2)))
            added_scores.append(0.45)   # ציון בינוני — רק כדי לעבור filter
            added_r.append(float(r))

    return added_boxes, added_scores, added_r






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
    מזהה כדורים (YOLO + Hough + השלמות), שומר את ה- white/black מהשלבים הקודמים
    בעזרת תגיות זיכרון (mem_white/mem_black), ומסנן כיסים/לבד בלי להפיל את הזיכרון.
    """
    img = stage["img"]
    H, W = img.shape[:2]

    # --- felt mask לשלב הסינון המאוחר ---
    felt_mask = make_felt_mask(img)

    # --- 1) YOLO + Hough ---
    yolo_boxes, yolo_scores = yolo_detect(img)
    h_boxes, h_scores, h_radii = hough_fallback(img, yolo_boxes)

    all_boxes  = yolo_boxes + h_boxes
    all_scores = yolo_scores + h_scores
    all_radii  = [est_radius_from_box(b) for b in yolo_boxes] + h_radii
    all_tags   = [set() for _ in all_boxes]

    # --- 1.1) הזרקת זיכרון: לבן/שחור שנמצאו בשלבים קודמים ---
    inject_memory_candidate(stage.get("white"), all_boxes, all_scores, all_radii, all_tags, boost_score=0.99, tag_name="mem_white")
    inject_memory_candidate(stage.get("black"), all_boxes, all_scores, all_radii, all_tags, boost_score=1.05, tag_name="mem_black")

    # --- 1.2) השלמת מועמדים (להחזיר כדורים חסרים) ---
    min_r = int(max(6, 0.009 * min(H, W)))
    max_r = int(max(min_r + 6, 0.018 * min(H, W)))
    add_boxes, add_scores, add_r = add_missing_orange_candidates(img, all_boxes, min_r, max_r)
    for b, s, r in zip(add_boxes, add_scores, add_r):
        all_boxes.append(b); all_scores.append(s); all_radii.append(r); all_tags.append(set())

    # --- 1.3) NMS עם תגיות ---
    all_boxes, all_scores, all_radii, all_tags = nms_with_tags(all_boxes, all_scores, all_radii, all_tags, iou_thr=0.40)

    # --- 2) סינון בסיסי + מגבלה (שומר תגיות) ---
    # גרסה קטנה של filter_and_limit ששומרת תגיות
    hmin, wmin = img.shape[:2]
    min_r_px = MIN_RADIUS_PX_OVERRIDE if MIN_RADIUS_PX_OVERRIDE else int(MIN_RADIUS_REL * min(hmin, wmin))
    triplets = []
    for bx, sc, rr, tg in zip(all_boxes, all_scores, all_radii, all_tags):
        r = rr if rr is not None else est_radius_from_box(bx)
        if r >= min_r_px:
            triplets.append((bx, sc, r, tg))
    triplets.sort(key=lambda t: t[1], reverse=True)
    triplets = triplets[:MAX_BALLS]
    f_boxes  = [t[0] for t in triplets]
    f_scores = [t[1] for t in triplets]
    f_radii  = [t[2] for t in triplets]
    f_tags   = [t[3] for t in triplets]
    centers  = boxes_to_centers(f_boxes)

    # --- 3) כיסים + Origin ---
    pockets_cand = detect_pocket_candidates(img)
    if pockets_cand:
        bl_guess = min(pockets_cand, key=lambda p: (p[0]-0)**2 + (p[1]-H)**2)
    else:
        bl_guess = None
    if bl_guess is None:
        blx, bly, blr = 0.0, float(H - 1), 0.03 * min(W, H)
    else:
        blx, bly, blr = map(float, bl_guess)

    # 3.1) הסרת כדורים בכיסים (מוגן לזיכרון)
    if pockets_cand:
        keep_idx = remove_in_pockets_protected(centers, f_radii, pockets_cand, f_tags, factor=POCKET_INCLUSION_FACTOR)
        f_boxes  = keep_by_indices(f_boxes,  keep_idx)
        f_scores = keep_by_indices(f_scores, keep_idx)
        f_radii  = keep_by_indices(f_radii,  keep_idx)
        f_tags   = keep_by_indices(f_tags,   keep_idx)
        centers  = keep_by_indices(centers,  keep_idx)

    # --- 4) סינון לפי הלבד (מוגן לזיכרון) ---
    median_r  = int(np.median(f_radii)) if f_radii else 8
    keep_soft = felt_filter_protected(H, W, centers, f_radii, f_tags, felt_mask,
                                      min_frac=0.55, min_frac_in=0.35,
                                      erode_px=max(2, int(0.40*median_r)),
                                      dilate_px=max(1, int(0.20*median_r)))
    f_boxes  = keep_by_indices(f_boxes,  keep_soft)
    f_scores = keep_by_indices(f_scores, keep_soft)
    f_radii  = keep_by_indices(f_radii,  keep_soft)
    f_tags   = keep_by_indices(f_tags,   keep_soft)
    centers  = keep_by_indices(centers,  keep_soft)


        # === הומוגרפיה: מהמועמדים לכיסים בנה מיפוי לתצוגת שולחן ישר ===
    six = assign_six_pockets(pockets_cand, W, H)
    H_img2tab, (TW, TH) = (None, (2.0, 1.0))
    if six is not None:
        H_img2tab, (TW, TH) = build_img2table_h(six, table_w=2.0, table_h=1.0)

    # העתק נקודות הכדורים למערכת השולחן (u,v)
    uv = warp_points_xy(centers, H_img2tab) if H_img2tab is not None else np.zeros((len(centers),2), dtype=np.float32)
    # נרמול ל-[0..1] (u_norm=x/TW, v_norm=y/TH)
    uv_norm = uv.copy()
    if uv_norm.size > 0:
        uv_norm[:,0] /= TW
        uv_norm[:,1] /= TH


    # --- 5) תיוג סופי לפי תגיות זיכרון ---
    types = []
    for tg in f_tags:
        if "mem_black" in tg and "mem_white" in tg:
            # אם בטעות שתיהן על אותה תיבה — עדיף black
            types.append("black")
        elif "mem_black" in tg:
            types.append("black")
        elif "mem_white" in tg:
            types.append("white")
        else:
            types.append("other")

    # --- 6) ציור + JSON ---
    ann = img.copy()
    for i, b in enumerate(f_boxes):
        t = types[i]
        color = (0,255,0)
        if t == "white": color = (255,255,0)
        elif t == "black": color = (255,0,255)
        x1,y1,x2,y2 = map(int,b)
        cv2.rectangle(ann,(x1,y1),(x2,y2),color,2)
        cv2.putText(ann,t,(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.circle(ann, (int(blx), int(bly)), int(blr), (0,165,255), 2)
    cv2.putText(ann, "Origin", (int(blx)+4, int(bly)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    cv2.imwrite(OUTPUT_ANN_PATH, ann)

    balls_json = []
    for i, ((cx, cy), t) in enumerate(zip(centers, types)):
        if t == "white":
            rec = {
                "index": 0,
                "type": t,
                "center_px": {"x": float(cx), "y": float(cy)},  # נשמר למעקב/דיבוג
                "rel_to_BL_px": {"x": float(cx - blx), "y": float(bly - cy)},  # תאימות לאחור
        }
        elif t == "black":
            rec = {
                "index": 1,
                "type": t,
                "center_px": {"x": float(cx), "y": float(cy)},  # נשמר למעקב/דיבוג
                "rel_to_BL_px": {"x": float(cx - blx), "y": float(bly - cy)},  # תאימות לאחור
            }
        else:
            rec = {
                "index": i,
                "type": t,
                "center_px": {"x": float(cx), "y": float(cy)},  # נשמר למעקב/דיבוג
                "rel_to_BL_px": {"x": float(cx - blx), "y": float(bly - cy)},  # תאימות לאחור
            }
        if uv_norm.size > 0:
            rec["table_uv"] = {"u": float(uv_norm[i,0]), "v": float(uv_norm[i,1])}   # 0..1, 2:1 כבר נלקח בחשבון
            rec["table_xy_units"] = {"x": float(uv[i,0]), "y": float(uv[i,1])}        # 0..2 ברוחב, 0..1 בגובה
        balls_json.append(rec)

    result = {
        "image_path": IMAGE_PATH,
        "origin_px": {"x": float(blx), "y": float(bly)},
        "pockets_px": {
            "TL": {"x": float(six["TL"][0]), "y": float(six["TL"][1])} if six else None,
            "TM": {"x": float(six["TM"][0]), "y": float(six["TM"][1])} if six else None,
            "TR": {"x": float(six["TR"][0]), "y": float(six["TR"][1])} if six else None,
            "BL": {"x": float(six["BL"][0]), "y": float(six["BL"][1])} if six else None,
            "BM": {"x": float(six["BM"][0]), "y": float(six["BM"][1])} if six else None,
            "BR": {"x": float(six["BR"][0]), "y": float(six["BR"][1])} if six else None,
        },
        "table_rect_units": {"width": 2.0, "height": 1.0},   # יחס 2:1
        "homography_img2table": H_img2tab.tolist() if H_img2tab is not None else None,
        "table_size_px": {"width_px": float(W), "height_px": float(H)},
        "balls": balls_json
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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
