from typing import Tuple, Any
from PIL import Image, ImageDraw
import json, os

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
        if not self.input_path or not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ image not found: {self.input_path}")

        self.origin_px = (float(meta["origin_px"]["x"]), float(meta["origin_px"]["y"]))
        self.table_rect_units = meta.get("table_rect_units", {"width": 2.0, "height": 1.0})
        self.balls = meta.get("balls", [])
        print("[DEBUG] Loaded balls:", [b["index"] for b in self.balls])
        self.pockets = meta.get("pockets_px", {})

        self.img = Image.open(self.input_path).convert("RGB")
        self.output_path = output_path or "output_with_lines.jpg"

    def get_ball_px(self, ball_id: int) -> Tuple[float, float] | None:
        """מאחזר מיקום פיקסלים של כדור לפי index מה־JSON."""
        for b in self.balls:
            if b["index"] == ball_id:
                if "center_px" in b:  # עדיף להשתמש בנתוני center_px
                    return (b["center_px"]["x"], b["center_px"]["y"])
        return None

    def get_pocket_px(self, pocket_id: int) -> Tuple[float, float] | None:
        """מאחזר מיקום כיס לפי id (0..5)."""
        mapping =  ["BL","BR" ,"TR" ,"TL" , "BM" ,"TM"]
        if 0 <= pocket_id < len(mapping):
            name = mapping[pocket_id]
            if name in self.pockets and self.pockets[name]:
                return (self.pockets[name]["x"], self.pockets[name]["y"])
        return None

    def draw_lines(self, color_target=(255, 0, 0), color_white=(0, 0, 255), width=3) -> str:
        """
        מצייר קו מהכדור הלבן → מטרה, וקו מהכדור מטרה → כיס.
        """
        draw = ImageDraw.Draw(self.img)
        print("Drawing lines...")
        print(f"  White ID: {self.best_shot.white.id}")
        print(f"  Target ID: {self.best_shot.target.id}")
        print(f"  Pocket ID: {self.best_shot.pocket.id}")

        white_px  = self.get_ball_px(self.best_shot.white.id)
        target_px = self.get_ball_px(self.best_shot.target.id)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)

        if white_px and target_px:
            draw.line([white_px, target_px], fill=color_white, width=width)  # לבן → מטרה
        if target_px and pocket_px:
            draw.line([target_px, pocket_px], fill=color_target, width=width)  # מטרה → כיס

        self.img.save(self.output_path, quality=95)
        return self.output_path
