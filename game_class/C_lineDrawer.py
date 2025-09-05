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
        מצייר קו מההיקף של הכדור הלבן → היקף המטרה,
        וקו מההיקף של המטרה → קצת לפני הכיס.
        """
        draw = ImageDraw.Draw(self.img)
        print("Drawing lines...")
        print(f"  White ID: {self.best_shot.white.id}")
        print(f"  Target ID: {self.best_shot.target.id}")
        print(f"  Pocket ID: {self.best_shot.pocket.id}")

        white_px = self.get_ball_px(self.best_shot.white.id)
        target_px = self.get_ball_px(self.best_shot.target.id)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)

        if not (white_px and target_px and pocket_px):
            raise ValueError("❌ Missing ball or pocket coordinates")

        # --- לבן → מטרה ---
        dx, dy = target_px[0] - white_px[0], target_px[1] - white_px[1]
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist

        # התחלה: היקף הלבן
        start_white = (
            white_px[0] + ux * BALL_RADIUS_PHOTO,
            white_px[1] + uy * BALL_RADIUS_PHOTO,
        )

        # סיום: היקף המטרה בצד שפונה ללב
        contact_target = (
            target_px[0] - ux * BALL_RADIUS_PHOTO,
            target_px[1] - uy * BALL_RADIUS_PHOTO,
        )

        draw.line([start_white, contact_target], fill=color_white, width=width)

        # --- מטרה → כיס ---
        dx2, dy2 = pocket_px[0] - target_px[0], pocket_px[1] - target_px[1]
        dist2 = math.hypot(dx2, dy2)
        ux2, uy2 = dx2 / dist2, dy2 / dist2

        # התחלה: היקף המטרה בצד שפונה לכיס
        start_target = (
            target_px[0] + ux2 * BALL_RADIUS_PHOTO,
            target_px[1] + uy2 * BALL_RADIUS_PHOTO,
        )

        pocket_before = (
            pocket_px[0] - ux2 * POCKET_MARGIN,
            pocket_px[1] - uy2 * POCKET_MARGIN,
        )

        draw.line([start_target, pocket_before], fill=color_target, width=width)

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
