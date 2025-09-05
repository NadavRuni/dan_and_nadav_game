import cv2
from ultralytics import YOLO
import numpy as np
import os, json

# ====== PATHS ======
IMAGE_PATH       = "photos/img_start.jpeg"
OUTPUT_ANN_PATH  = "photos/output/img_output.jpg"
OUTPUT_JSON_PATH = "photos/output/img_JSON.json"

# Debug stage outputs
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

POCKET_INCLUSION_FACTOR = 1.05  # factor for pocket inclusion

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

        for idx in valid[1:]:
            j = int(order[idx])
            if iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                used[idx] = True
                if tags and j < len(tags):
                    tg = tags[j]
                    kt |= set(tg) if isinstance(tg, (set, list, tuple)) else ({tg} if tg else set())

        kept_boxes.append(kb)
        kept_scores.append(ks)
        kept_radii.append(kr)
        kept_tags.append(kt)

    return kept_boxes, kept_scores, kept_radii, kept_tags


def inject_memory_candidate(mem, all_boxes, all_scores, all_radii, all_tags, boost_score, tag_name):
    if not mem:
        return
    box = mem["box"]; r = mem["radius"]
    for j, bx in enumerate(all_boxes):
        if iou_xyxy(box, bx) > 0.30:
            all_scores[j] = max(all_scores[j], boost_score)
            all_tags[j].add(tag_name)
            return
    all_boxes.append(box)
    all_scores.append(boost_score)
    all_radii.append(r)
    all_tags.append({tag_name})

# ---------------- Felt / Table geometry ----------------
def make_felt_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # טווחים סלחניים לכחול/ירוק
    mask_green = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255))
    mask_blue  = cv2.inRange(hsv, (85, 30, 30), (140, 255, 255))
    mask = cv2.bitwise_or(mask_green, mask_blue)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8), 2)
    return mask

def _order_corners(pts):
    # pts: 4x2 float32 (arbitrary order). Return dict TL,TR,BR,BL
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    TL = pts[np.argmin(s)]
    BR = pts[np.argmax(s)]
    TR = pts[np.argmin(d)]
    BL = pts[np.argmax(d)]
    return {"TL": (float(TL[0]), float(TL[1])),
            "TR": (float(TR[0]), float(TR[1])),
            "BR": (float(BR[0]), float(BR[1])),
            "BL": (float(BL[0]), float(BL[1]))}

def detect_table_corners(img_bgr):
    """
    מזהה את 4 פינות הלבד בצורה יציבה (לא תלוי בכיסים).
    """
    mask = make_felt_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("לא נמצאה מעטפת לבד.")
    cnt = max(contours, key=cv2.contourArea)
    # קופסת מינימום מסתובבת
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect)  # 4x2 float
    corners = _order_corners(box)
    return corners, mask

def anchors_from_corners(corners):
    TL = corners["TL"]; TR = corners["TR"]; BR = corners["BR"]; BL = corners["BL"]
    TM = ((TL[0]+TR[0])*0.5, (TL[1]+TR[1])*0.5)
    BM = ((BL[0]+BR[0])*0.5, (BL[1]+BR[1])*0.5)
    pockets = {"TL": TL, "TM": TM, "TR": TR, "BL": BL, "BM": BM, "BR": BR}
    origin = BL  # כאן קובעים חד-משמעית: מקור הצירים = הפינה התחתונה-שמאלית של הלבד
    return pockets, origin

# ---- Optional: refine each pocket near its anchor using local darkness (robust but not mandatory)
def refine_pockets_near_anchors(img_bgr, pockets, search_px=40):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    out = {}
    for name, (ax, ay) in pockets.items():
        x1 = int(max(0, ax - search_px)); x2 = int(min(w-1, ax + search_px))
        y1 = int(max(0, ay - search_px)); y2 = int(min(h-1, ay + search_px))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            out[name] = (float(ax), float(ay))
            continue
        # חפש את הנקודה הכי כהה ב־ROI (כיסים כהים). אם אין – השאר עוגן.
        minval, _, minloc, _ = cv2.minMaxLoc(roi)
        if minval < 140:  # כהה יחסית
            px = x1 + minloc[0]
            py = y1 + minloc[1]
            out[name] = (float(px), float(py))
        else:
            out[name] = (float(ax), float(ay))
    return out

# ---------------- Homography ----------------
def build_img2table_h(pmap, table_w=2.0, table_h=1.0):
    src = np.array([
        pmap["TL"], pmap["TM"], pmap["TR"],
        pmap["BL"], pmap["BM"], pmap["BR"]
    ], dtype=np.float32)
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
    if H is None or len(points) == 0:
        return np.zeros((0,2), dtype=np.float32)
    pts = np.array(points, dtype=np.float32).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    return warped

# ---------------- misc ----------------
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
    r_max = max(r_min + 6, int(0.022 * min(h, w)))
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

# ---------------- Pockets / felt filters used later ----------------
def keep_by_indices(seq, indices):
    return [seq[i] for i in indices]

def disk_overlap_frac_with_mask(Hh, Ww, cx, cy, r, mask):
    cx = int(round(cx)); cy = int(round(cy))
    r  = int(max(2, round(r)))
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(Ww, cx + r + 1), min(Hh, cy + r + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    submask = mask[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    circle = ((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)) <= (r * r)
    area = int(circle.sum())
    if area == 0:
        return 0.0
    hit = (submask > 0) & circle
    return float(hit.sum()) / float(area)

def felt_filter_protected(Hh, Ww, centers, radii, tags, felt_mask,
                          min_frac=0.55, min_frac_in=0.35,
                          erode_px=3, dilate_px=2):
    k1 = 2*erode_px + 1
    k2 = 2*dilate_px + 1
    felt_eroded = cv2.erode(felt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)), 1)
    felt_dilated= cv2.dilate(felt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)), 1)

    def frac(mask, cx, cy, r):
        cx, cy, r = float(cx), float(cy), float(r)
        return disk_overlap_frac_with_mask(Hh, Ww, cx, cy, r, mask)

    keep = []
    dropped = []
    for i, (cx, cy) in enumerate(centers):
        if ("mem_white" in tags[i]) or ("mem_black" in tags[i]):
            keep.append(i); continue
        r = radii[i]
        f  = frac(felt_mask,  cx, cy, r)
        fi = frac(felt_eroded, cx, cy, r)

        # הבטחת תחום חוקי
        ix = min(max(int(round(cx)), 0), Ww-1)
        iy = min(max(int(round(cy)), 0), Hh-1)

        if f >= min_frac or fi >= min_frac_in or (felt_dilated[iy, ix] > 0):
            keep.append(i)
        else:
            dropped.append(i)

    # Rescue רך
    for i in dropped:
        if len(keep) >= len(centers):
            break
        cx, cy, r = centers[i][0], centers[i][1], radii[i]
        if disk_overlap_frac_with_mask(Hh, Ww, cx, cy, r, felt_dilated) >= 0.25:
            keep.append(i)

    return sorted(set(keep))

# =========================================================
#                    PIPELINE STAGES
# =========================================================
def white_recognizer(stage):
    img = stage["img"]
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

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
        Hc,S,V = cv2.split(hsv)
        for x, y, r in circles:
            if x < border_margin or x > w - border_margin or y < border_margin or y > h - border_margin:
                continue
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
            if intensity < 20:
                continue
            if intensity < best_intensity:
                best_intensity = intensity
                box = (float(x-r), float(y-r), float(x+r), float(y+r))
                best = {"center":(float(x),float(y)), "radius":float(r), "box":box}

    dbg = img.copy()
    if best:
        draw_rect_label(dbg, best["box"], "black", (255,0,255))
    cv2.imwrite(OUTPUT_BLACK_ANN, dbg)

    stage["black"] = best
    return stage

def image_recognizer(stage):
    img = stage["img"]
    H, W = img.shape[:2]

    # --- 0) גיאומטריית השולחן (ודאית): פינות הלבד → כיסים → Origin=BL ---
    corners, felt_mask = detect_table_corners(img)
    six, origin_xy = anchors_from_corners(corners)
    # אפשר עדכון עדין סביב העוגנים (לא חובה, אבל יציב):
    six = refine_pockets_near_anchors(img, six, search_px=int(0.04*min(H,W)))

    blx, bly = origin_xy
    blr = 0.03 * min(W, H)  # רדיוס ציור להמחשה בלבד

    # --- 1) YOLO + Hough ---
    yolo_boxes, yolo_scores = yolo_detect(img)
    h_boxes, h_scores, h_radii = hough_fallback(img, yolo_boxes)

    all_boxes  = yolo_boxes + h_boxes
    all_scores = yolo_scores + h_scores
    all_radii  = [est_radius_from_box(b) for b in yolo_boxes] + h_radii
    all_tags   = [set() for _ in all_boxes]

    inject_memory_candidate(stage.get("white"), all_boxes, all_scores, all_radii, all_tags, boost_score=0.99, tag_name="mem_white")
    inject_memory_candidate(stage.get("black"), all_boxes, all_scores, all_radii, all_tags, boost_score=1.05, tag_name="mem_black")

    min_r = int(max(6, 0.009 * min(H, W)))
    max_r = int(max(min_r + 6, 0.018 * min(H, W)))
    # (אפשר להוסיף החלפת כתומים כאן אם תרצה)

    all_boxes, all_scores, all_radii, all_tags = nms_with_tags(all_boxes, all_scores, all_radii, all_tags, iou_thr=0.40)

    # --- 2) סינון בסיסי + מגבלה ---
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

    # --- 3) סינון לפי הלבד (מוגן לזיכרון) ---
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

    # === הומוגרפיה מהעוגנים היציבים ===
    H_img2tab, (TW, TH) = build_img2table_h(six, table_w=2.0, table_h=1.0)

    uv = warp_points_xy(centers, H_img2tab) if H_img2tab is not None else np.zeros((len(centers),2), dtype=np.float32)
    uv_norm = uv.copy()
    if uv_norm.size > 0:
        uv_norm[:,0] /= TW
        uv_norm[:,1] /= TH

    # --- תיוג לפי זיכרון ---
    types = []
    for tg in f_tags:
        if "mem_black" in tg and "mem_white" in tg:
            types.append("black")
        elif "mem_black" in tg:
            types.append("black")
        elif "mem_white" in tg:
            types.append("white")
        else:
            types.append("other")

    # --- ציור + JSON ---
    ann = img.copy()
    for i, b in enumerate(f_boxes):
        t = types[i]
        color = (0,255,0)
        if t == "white": color = (255,255,0)
        elif t == "black": color = (255,0,255)
        x1,y1,x2,y2 = map(int,b)
        cv2.rectangle(ann,(x1,y1),(x2,y2),color,2)
        cv2.putText(ann,t,(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # צייר עוגנים (בדגש על Origin=BL)
    cv2.circle(ann, (int(blx), int(bly)), int(blr), (0,165,255), 2)
    cv2.putText(ann, "Origin (BL)", (int(blx)+6, int(bly)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    for name, (px,py) in six.items():
        cv2.circle(ann, (int(px), int(py)), 8, (0,128,255), 2)
        cv2.putText(ann, name, (int(px)+6, int(py)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 1)

    cv2.imwrite(OUTPUT_ANN_PATH, ann)

    balls_json = []
    for i, ((cx, cy), t) in enumerate(zip(centers, types)):
        rec = {
            "index": i,
            "type": t,
            "center_px": {"x": float(cx), "y": float(cy)},
            "rel_to_BL_px": {"x": float(cx - blx), "y": float(bly - cy)},
        }
        if uv_norm.size > 0:
            rec["table_uv"] = {"u": float(uv_norm[i,0]), "v": float(uv_norm[i,1])}
            rec["table_xy_units"] = {"x": float(uv[i,0]), "y": float(uv[i,1])}
        balls_json.append(rec)

    result = {
        "image_path": IMAGE_PATH,
        "origin_px": {"x": float(blx), "y": float(bly)},
        "pockets_px": {
            "TL": {"x": float(six["TL"][0]), "y": float(six["TL"][1])},
            "TM": {"x": float(six["TM"][0]), "y": float(six["TM"][1])},
            "TR": {"x": float(six["TR"][0]), "y": float(six["TR"][1])},
            "BL": {"x": float(six["BL"][0]), "y": float(six["BL"][1])},
            "BM": {"x": float(six["BM"][0]), "y": float(six["BM"][1])},
            "BR": {"x": float(six["BR"][0]), "y": float(six["BR"][1])},
        },
        "table_rect_units": {"width": 2.0, "height": 1.0},
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

    stage0 = {"img": img, "white": None, "black": None}
    stage1 = white_recognizer(stage0)
    stage2 = black_recognizer(stage1)
    _      = image_recognizer(stage2)

if __name__ == "__main__":
    main()
