from typing import Tuple, Any
from PIL import Image, ImageDraw
import json, os
import math
from pathlib import Path
from const_numbers import OUTPUT_CONTACT_VIEW_PATH , BALL_RADIUS_PHOTO, POCKET_MARGIN


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

        self.origin_px = (float(meta["origin_px"]["x"]), float(meta["origin_px"]["y"]))
        self.table_rect_units = meta.get(
            "table_rect_units", {"width": 2.0, "height": 1.0}
        )
        self.balls = meta.get("balls", [])
        print("[DEBUG] Loaded balls:", [b["index"] for b in self.balls])
        self.pockets = meta.get("pockets_px", {})

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
        if 0 <= pocket_id < len(mapping):
            name = mapping[pocket_id]
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
            target_px[0] - ux_p * BALL_RADIUS_PHOTO,
            target_px[1] - uy_p * BALL_RADIUS_PHOTO,
        )

        # --- קו לבן → נקודת מגע ---
        dx_w, dy_w = contact_target[0] - white_px[0], contact_target[1] - white_px[1]
        dist_w = math.hypot(dx_w, dy_w)
        ux_w, uy_w = dx_w / dist_w, dy_w / dist_w

        start_white = (
            white_px[0] + ux_w * BALL_RADIUS_PHOTO,
            white_px[1] + uy_w * BALL_RADIUS_PHOTO,
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
            target_px[0] + ux_p * BALL_RADIUS_PHOTO,
            target_px[1] + uy_p * BALL_RADIUS_PHOTO,
        )
        pocket_before = (
            pocket_px[0] - ux_p * POCKET_MARGIN,
            pocket_px[1] - uy_p * POCKET_MARGIN,
        )
        draw_dashed_line(draw, start_target, pocket_before, fill=color_target, width=width)

        self.img.save(self.output_path, quality=95)
        return self.output_path




    def show_contact_hit(
        self, ball_radius: int = BALL_RADIUS_PHOTO-3, color=(255, 0, 0), size: int = 8, crop_size: int = 120
    ) -> str:
        """
        מצייר נקודת מגע על הכדור המטרה בצד שפונה לכיס (ולא בצד שפונה ללב),
        חותך (zoom-in) סביב הכדור המטרה ושומר לנתיב OUTPUT_CONTACT_VIEW_PATH.
        """
        # נקודות פיקסלים
        target_px = self.get_ball_px(self.best_shot.target.id)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)

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
