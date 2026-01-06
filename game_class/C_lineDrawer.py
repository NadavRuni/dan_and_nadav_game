"""
Provides the LineDrawer class for drawing shot trajectories on an image.

This module uses the Pillow (PIL) library to draw detailed visualizations of
calculated shots, including direct shots, combination shots, and wall shots,
onto the original table image.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from const_numbers import (
    OUTPUT_CONTACT_VIEW_PATH,
    get_ball_radius_photo,
    get_pocket_margin,
    get_wall_margin,
)


class LineDrawer:
    """
    Draws calculated shot trajectories onto a table image.
    """

    def __init__(
        self, json_path: str, best_shot: Any, output_path: Optional[str] = None
    ):
        """
        Initializes the LineDrawer with data from a JSON file and a shot object.

        Args:
            json_path: Path to the JSON file containing the game state
                       (image_path, balls, pockets).
            best_shot: The BestShot object representing the shot to be drawn.
            output_path: Optional path for the output image. If not provided,
                         a default is used.

        Raises:
            FileNotFoundError: If the image specified in the JSON is not found.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.best_shot = best_shot
        self.input_path = meta.get("image_path")
        if not self.input_path or not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ Image not found: {self.input_path}")

        self.balls = meta.get("balls", [])
        self.pockets = meta.get("pockets", {})
        self.img = Image.open(self.input_path).convert("RGB")

        base_dir = Path.cwd()
        self.output_path = base_dir / (output_path or "output_with_lines.jpg")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def get_ball_px(self, ball_id: int) -> Optional[Tuple[float, float]]:
        """Retrieves the pixel coordinates of a ball by its ID."""
        for ball in self.balls:
            if ball.get("index") == ball_id and "center_px" in ball:
                return (ball["center_px"]["x"], ball["center_px"]["y"])
        return None

    def get_pocket_px(self, pocket_id: int) -> Optional[Tuple[float, float]]:
        """Retrieves the pixel coordinates of a pocket by its ID."""
        # Note: This mapping is fragile and assumes a consistent order.
        mapping = ["BL", "BR", "TR", "TL", "BM", "TM"]
        if 0 <= pocket_id < len(mapping):
            name = mapping[pocket_id]
            if name in self.pockets and self.pockets[name]:
                return (self.pockets[name]["x"], self.pockets[name]["y"])
        return None

    def _draw_dashed_line(
        self,
        draw: ImageDraw,
        start: tuple,
        end: tuple,
        fill: str,
        width: int,
        dash_length: int = 15,
        gap_length: int = 10,
    ) -> None:
        """Helper to draw a dashed line on the image."""
        x1, y1 = start
        x2, y2 = end
        total_length = math.hypot(x2 - x1, y2 - y1)
        if total_length == 0:
            return

        dx, dy = (x2 - x1) / total_length, (y2 - y1) / total_length
        pos = 0.0
        while pos < total_length:
            x_start, y_start = x1 + dx * pos, y1 + dy * pos
            pos += dash_length
            x_end, y_end = x1 + dx * min(pos, total_length), y1 + dy * min(
                pos, total_length
            )
            draw.line([(x_start, y_start), (x_end, y_end)], fill=fill, width=width)
            pos += gap_length

    def draw_direct_shot_lines(
        self, color_target: str = "red", color_white: str = "blue", width: int = 3
    ) -> str:
        """Draws the trajectory for a direct shot."""
        draw = ImageDraw.Draw(self.img)
        white_px = self.get_ball_px(self.best_shot.white.id)
        target_px = self.get_ball_px(self.best_shot.target.id)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)

        if not (white_px and target_px and pocket_px):
            raise ValueError("❌ Missing ball or pocket coordinates for drawing.")

        # Vector from target to pocket
        dx_p, dy_p = pocket_px[0] - target_px[0], pocket_px[1] - target_px[1]
        dist_p = math.hypot(dx_p, dy_p)
        ux_p, uy_p = (dx_p / dist_p, dy_p / dist_p) if dist_p else (0, 0)

        # Contact point on target ball
        contact_point = (
            target_px[0] - ux_p * get_ball_radius_photo(),
            target_px[1] - uy_p * get_ball_radius_photo(),
        )

        # Draw line from white ball to contact point
        self._draw_dashed_line(
            draw, white_px, contact_point, fill=color_white, width=width
        )

        # Draw line from target ball to pocket
        start_target = (
            target_px[0] + ux_p * get_ball_radius_photo(),
            target_px[1] + uy_p * get_ball_radius_photo(),
        )
        self._draw_dashed_line(
            draw, start_target, pocket_px, fill=color_target, width=width
        )

        self.img.save(self.output_path, quality=95)
        return str(self.output_path)

    def draw_combo_shot_lines(
        self,
        color_mid: str = "cyan",
        color_target: str = "red",
        color_white: str = "blue",
        width: int = 6,
    ) -> str:
        """Draws the trajectory for a combination (3-ball) shot."""
        draw = ImageDraw.Draw(self.img)

        white_px = self.get_ball_px(self.best_shot.white.id)
        mid_px = self.get_ball_px(self.best_shot.target_helper.id)
        target_px = self.get_ball_px(self.best_shot.target.id)
        pocket_px = self.get_pocket_px(self.best_shot.pocket.id)

        if not all((white_px, mid_px, target_px, pocket_px)):
            raise ValueError("❌ Missing coordinates for combo shot.")

        radius = get_ball_radius_photo()

        # Step 3 (End): Target -> Pocket
        ux3, uy3 = self._get_unit_vector(target_px, pocket_px)
        start_3 = (target_px[0] + ux3 * radius, target_px[1] + uy3 * radius)
        self._draw_dashed_line(draw, start_3, pocket_px, fill=color_target, width=width)

        # Step 2 (Middle): Helper -> Target
        contact_on_target = (target_px[0] - ux3 * radius, target_px[1] - uy3 * radius)
        ux2, uy2 = self._get_unit_vector(mid_px, contact_on_target)
        start_2 = (mid_px[0] + ux2 * radius, mid_px[1] + uy2 * radius)
        self._draw_dashed_line(
            draw, start_2, contact_on_target, fill=color_mid, width=width
        )

        # Step 1 (Start): White -> Helper
        contact_on_mid = (mid_px[0] - ux2 * radius, mid_px[1] - uy2 * radius)
        ux1, uy1 = self._get_unit_vector(white_px, contact_on_mid)
        start_1 = (white_px[0] + ux1 * radius, white_px[1] + uy1 * radius)
        self._draw_dashed_line(
            draw, start_1, contact_on_mid, fill=color_white, width=width
        )

        self.img.save(self.output_path, quality=95)
        return str(self.output_path)

    def _get_unit_vector(self, p1: tuple, p2: tuple) -> tuple:
        """Calculates the unit vector from point 1 to point 2."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        return (dx / dist, dy / dist) if dist != 0 else (0, 0)
