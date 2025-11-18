from typing import Tuple, Any
from PIL import Image, ImageDraw
import json, os
import math
from pathlib import Path
from const_numbers import *


class LineDrawer:
    def __init__(self, json_path: str, best_shot: Any, output_path: str = None):
        """
        json_path - קובץ JSON שמכיל image_path, origin_px, balls, pockets
        best_shot - אובייקט BestShot עם white.id, target.id, pocket.id
        """
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.best_shot = best_shot
        self.input_path = meta.get("image_path")
        print("self.input_path", self.input_path)
        if not self.input_path or not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ image not found: {self.input_path}")

        origin_data = meta.get("origin_px")
        if origin_data:
            self.origin_px = (float(origin_data["x"]), float(origin_data["y"]))
        else:
            # fallback — נשתמש בנקודת התחלה (0,0)
            self.origin_px = (0.0, 0.0)
            print("[LineDrawer] ⚠️ Missing origin_px in JSON, using (0,0) as fallback")
        self.table_rect_units = meta.get(
            "table_rect_units", {"width": 2.0, "height": 1.0}
        )
        self.balls = meta.get("balls", [])
        print("[DEBUG] Loaded balls:", [b["index"] for b in self.balls])
        self.pockets = meta.get("pockets", {})
        print("[DEBUG] Loaded pockets:", self.pockets.keys())


        self.img = Image.open(self.input_path).convert("RGB")
        base_dir = os.getcwd()
        self.output_path = os.path.join(
            base_dir, output_path or "output_with_lines.jpg"
        )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        print("self.output_path", self.output_path)

    def get_ball_px(self, ball_id: int) -> Tuple[float, float] | None:
        """מאחזר מיקום פיקסלים של כדור לפי index מה־JSON."""
        for b in self.balls:
            if b["index"] == ball_id:
                if "center_px" in b:  # עדיף להשתמש בנתוני center_px
                    return (b["center_px"]["x"], b["center_px"]["y"])
        return None

    def get_pocket_px(self, pocket_id: int) -> Tuple[float, float] | None:
        """מאחזר מיקום כיס לפי id (0..5)."""
        mapping = ["BL", "BR", "TR", "TL", "BM", "TM"]
        print("Getting pocket px for pocket_id:", pocket_id)
        if 0 <= pocket_id < len(mapping):
            name = mapping[pocket_id]
            print("Mapped pocket name:", name)
            print ("Available pockets:", self.pockets.keys())
            print ("Pocket data:", self.pockets[name])
            if name in self.pockets and self.pockets[name]:
                return (self.pockets[name]["x"], self.pockets[name]["y"])
        return None

    def draw_lines(self, color_target=(255, 0, 0), color_white=(0, 0, 255), width=3) -> str:
        """
        מצייר את המסלול הפיזיקלי הנכון:
        לבן → נקודת מגע על המטרה
        מטרה (היקף בצד של הכיס) → כיס
        עם קווים מקווקווים.
        """

        draw = ImageDraw.Draw(self.img)
        print("Drawing contact-based dashed lines...")
        print(f"  White ID: {self.best_shot.white.id}")
        print(f"  Target ID: {self.best_shot.target.id}")
        print(f"  Pocket ID: {self.best_shot.pocket.id}")

        white_px = self.get_ball_px(self.best_shot.white.id)
        target_px = self.get_ball_px(self.best_shot.target.id)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)

        if not (white_px and target_px and pocket_px):
            raise ValueError("❌ Missing ball or pocket coordinates")

        # --- חישוב נקודת המגע על המטרה לפי הכיס ---
        dx_p, dy_p = pocket_px[0] - target_px[0], pocket_px[1] - target_px[1]
        dist_p = math.hypot(dx_p, dy_p)
        ux_p, uy_p = dx_p / dist_p, dy_p / dist_p

        # נקודת מגע על ההיקף (בצד שפונה לכיס)
        contact_target = (
            target_px[0] - ux_p * get_ball_radius_photo(),
            target_px[1] - uy_p * get_ball_radius_photo(),
        )

        # --- קו לבן → נקודת מגע ---
        dx_w, dy_w = contact_target[0] - white_px[0], contact_target[1] - white_px[1]
        dist_w = math.hypot(dx_w, dy_w)
        ux_w, uy_w = dx_w / dist_w, dy_w / dist_w

        start_white = (
            white_px[0] + ux_w * get_ball_radius_photo(),
            white_px[1] + uy_w * get_ball_radius_photo(),
        )

        def draw_dashed_line(draw, start, end, fill, width=3, dash_length=15, gap_length=10):
            x1, y1 = start
            x2, y2 = end
            total_length = math.hypot(x2 - x1, y2 - y1)
            dx, dy = (x2 - x1) / total_length, (y2 - y1) / total_length

            pos = 0
            while pos < total_length:
                x_start = x1 + dx * pos
                y_start = y1 + dy * pos
                pos += dash_length
                if pos > total_length:
                    pos = total_length
                x_end = x1 + dx * pos
                y_end = y1 + dy * pos
                draw.line([(x_start, y_start), (x_end, y_end)], fill=fill, width=width)
                pos += gap_length

        # לבן → נקודת מגע
        draw_dashed_line(draw, start_white, contact_target, fill=color_white, width=width)

        # --- מטרה (צד שפונה לכיס) → כיס ---
        start_target = (
            target_px[0] + ux_p * get_ball_radius_photo(),
            target_px[1] + uy_p * get_ball_radius_photo(),
        )
        pocket_before = (
            pocket_px[0] - ux_p * get_pocket_margin(),
            pocket_px[1] - uy_p * get_pocket_margin(),
        )
        draw_dashed_line(draw, start_target, pocket_before, fill=color_target, width=width)

        self.img.save(self.output_path, quality=95)
        return self.output_path

    def show_contact_hit(
        self, ball_radius: int = get_ball_radius_photo()-3, color=(255, 0, 0), size: int = 8, crop_size: int = 120
    ) -> str:
        """
        מצייר נקודת מגע על הכדור המטרה בצד שפונה לכיס (ולא בצד שפונה ללב),
        חותך (zoom-in) סביב הכדור המטרה ושומר לנתיב OUTPUT_CONTACT_VIEW_PATH.
        """
        # נקודות פיקסלים
        print("Drawing contact-to-pocket point...")
        target_px = self.get_ball_px(self.best_shot.target.id)
        print("target_px", target_px)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)
        print("pocket_px", pocket_px)

        if not target_px or not pocket_px:
            raise ValueError("❌ Missing target or pocket positions for contact hit")

        dx = pocket_px[0] - target_px[0]
        dy = pocket_px[1] - target_px[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            raise ValueError("❌ Target and pocket overlap")

        ux, uy = dx / dist, dy / dist

        # נקודת מגע: בקצה הכדור המטרה בצד שפונה לכיס
        contact_x = target_px[0] - ux * ball_radius
        contact_y = target_px[1] - uy * ball_radius

        # חיתוך סביב הכדור (zoom-in)
        left   = int(target_px[0] - crop_size)
        top    = int(target_px[1] - crop_size)
        right  = int(target_px[0] + crop_size)
        bottom = int(target_px[1] + crop_size)

        cropped = self.img.crop((left, top, right, bottom)).copy()
        draw = ImageDraw.Draw(cropped)

        # ציור נקודת הפגיעה
        r = size
        contact_x_cropped = contact_x - left
        contact_y_cropped = contact_y - top
        draw.ellipse(
            [
                contact_x_cropped - r,
                contact_y_cropped - r,
                contact_x_cropped + r,
                contact_y_cropped + r,
            ],
            outline=color,
            width=3,
        )

        cropped.save(OUTPUT_CONTACT_VIEW_PATH, quality=95)
        print(
            f"[DEBUG] Contact-to-pocket point drawn at ({contact_x:.2f}, {contact_y:.2f}), zoom saved."
        )
        return str(OUTPUT_CONTACT_VIEW_PATH)
        
    def table_to_px(self, x: float, y: float, smart_wall_margin=False) -> tuple[float, float]:
        """
        ממיר נקודה מיחידות שולחן (x,y) לפיקסלים בתמונה.
        
        סדר פעולות (כאשר smart_wall_margin=True):
        1. המרה מלאה לפיקסלים.
        2. הזזת הנקודה בפיקסלים (Margin) פנימה, אם זוהתה כדופן.
        """
        
        # 1. קבלת מידות
        width_px, height_px = self.img.size
        table_length = get_table_length()
        table_width = get_table_width()

        if table_length == 0 or table_width == 0:
            print("❌ Error: Table dimensions are zero.")
            return (0.0, 0.0)

        # -------------------------------------------
        # 2. המרה בסיסית לפיקסלים (לפני Margin)
        # -------------------------------------------
        u = x / table_length
        v = y / table_width

        px = u * width_px
        py = height_px - (v * height_px) # היפוך ציר Y

        # -------------------------------------------
        # 3. לוגיקה חכמה: הוספת Margin בפיקסלים
        # -------------------------------------------
        if smart_wall_margin:
            # ההנחה היא ש-get_wall_margin מחזיר ערך שמתאים לפיקסלים (או שהמשתמש רוצה להחיל אותו במישור התמונה)
            margin = get_wall_margin()
            epsilon = 2.0  # זיהוי קיר לפי יחידות שולחן מקוריות

            # -- ציר X --
            
            # דופן שמאל (x=0) -> בתמונה זה 0 -> דחיפה ימינה (+)
            if abs(x - 0) < epsilon:
                print(f"🛡️ Wall (Left): Pushing px {px:.1f} -> {px + margin:.1f}")
                px += margin
            
            # דופן ימין (x=Length) -> בתמונה זה Width -> דחיפה שמאלה (-)
            elif abs(x - table_length) < epsilon:
                print(f"🛡️ Wall (Right): Pushing px {px:.1f} -> {px - margin:.1f}")
                px -= margin

            # -- ציר Y --
            
            # דופן תחתונה (y=0) -> בתמונה זה Height (למטה) -> דחיפה למעלה (-)
            # הערה: בתמונה Y עולה ככל שיושבים למטה, לכן כדי לעלות למעלה צריך להחסיר
            if abs(y - 0) < epsilon:
                print(f"🛡️ Wall (Bottom): Pushing py {py:.1f} -> {py - margin:.1f}")
                py -= margin

            # דופן עליונה (y=Width) -> בתמונה זה 0 (למעלה) -> דחיפה למטה (+)
            elif abs(y - table_width) < epsilon:
                print(f"🛡️ Wall (Top): Pushing py {py:.1f} -> {py + margin:.1f}")
                py += margin

        print(f"Converting: Table({x:.1f}, {y:.1f}) -> Px({px:.1f}, {py:.1f}) [SmartMargin={smart_wall_margin}]")
        return (px, py)
    def draw_lines_with_wall(
        self,
        wall_point: tuple[float, float],   # ביחידות שולחן (x,y)
        color_target=(255, 0, 0),
        color_white=(0, 0, 255),
        color_wall=(0, 255, 0),
        width=6,
    ) -> str:
        """
        מצייר 3 קווים מקווקווים על התמונה:
        1) לבן → מטרה
        2) מטרה → קיר
        3) קיר → חור

        wall_point מתקבל ביחידות שולחן (290x145) ולכן מומר לפיקסלים.
        """

        def v_sub(a, b): return (a[0]-b[0], a[1]-b[1])
        def v_add(a, b): return (a[0]+b[0], a[1]+b[1])
        def v_len(v): return math.hypot(v[0], v[1])
        def v_unit(v):
            L = v_len(v)
            return (0.0, 0.0) if L == 0 else (v[0]/L, v[1]/L)
        def v_scale(v, s): return (v[0]*s, v[1]*s)

        def draw_dashed_line(draw, start, end, fill, width=6, dash_length=15, gap_length=10):
            x1, y1 = start
            x2, y2 = end
            total_length = math.hypot(x2 - x1, y2 - y1)
            if total_length == 0:
                return
            dx, dy = (x2 - x1) / total_length, (y2 - y1) / total_length
            pos = 0.0
            while pos < total_length:
                x_start = x1 + dx * pos
                y_start = y1 + dy * pos
                pos += dash_length
                if pos > total_length:
                    pos = total_length
                x_end = x1 + dx * pos
                y_end = y1 + dy * pos
                draw.line([(x_start, y_start), (x_end, y_end)], fill=fill, width=width)
                pos += gap_length

        # --- נקודות הכדורים/כיס (כבר בפיקסלים) ---
        white_c   = self.get_ball_px(self.best_shot.white.id)
        target_c  = self.get_ball_px(self.best_shot.target.id)
        pocket_c  = self.get_pocket_px(self.best_shot.pocket.id)

        # --- נקודת הקיר (המרה) ---
        wall_px = self.table_to_px(wall_point[0], wall_point[1], True)
        print ("Converted wall point to px:", wall_px)
        tw_dir = v_unit(v_sub(wall_px, target_c))  
        
        # FIXED: added () to get_wall_margin
        # wall_px = v_add(wall_px, v_scale(tw_dir, -get_wall_margin()))

        if not (white_c and target_c and pocket_c and wall_px):
            raise ValueError("❌ Missing coordinates for white/target/pocket/wall")

        draw = ImageDraw.Draw(self.img)

        print("=== Drawing Debug Info ===")
        print(f"White center   = {white_c}")
        print(f"Target center  = {target_c}")
        print(f"Pocket center  = {pocket_c}")
        print(f"Wall point (table) = {wall_point}")
        print(f"Wall point (px)    = {wall_px}")

        # --- 1) לבן → מטרה ---
        wt_dir = v_unit(v_sub(target_c, white_c))
        # FIXED: added () to get_ball_radius_photo
        start_white   = v_add(white_c,  v_scale(wt_dir, get_ball_radius_photo()))
        # FIXED: added () to get_ball_radius_photo
        end_on_target = v_add(target_c, v_scale(wt_dir, -get_ball_radius_photo()))
        
        print(f"Line 1: White edge {start_white} → Target edge {end_on_target}")
        draw_dashed_line(draw, start_white, end_on_target, fill=color_white, width=width)

        # --- 2) מטרה → קיר ---
        tw_dir = v_unit(v_sub(wall_px, target_c))
        # FIXED: added () to get_ball_radius_photo
        start_target_wall = v_add(target_c, v_scale(tw_dir, get_ball_radius_photo()))
        
        print(f"Line 2: Target edge {start_target_wall} → Wall {wall_px}")
        draw_dashed_line(draw, start_target_wall, wall_px, fill=color_target, width=width)

        # --- 3) קיר → חור ---
        wp_dir = v_unit(v_sub(pocket_c, wall_px))
        # FIXED: added () to get_pocket_margin
        pocket_before = v_add(pocket_c, v_scale(wp_dir, -get_pocket_margin()))
        
        print(f"Line 3: Wall {wall_px} → Pocket-before {pocket_before}")
        draw_dashed_line(draw, wall_px, pocket_before, fill=color_wall, width=width)

        print("=== Finished Drawing ===")

        self.img.save(self.output_path, quality=95)
        return self.output_path