"""
Utilities for drawing detected objects on images for visualization.

This module provides functions to overlay visual information, such as circles
for balls, polygons for the table rectangle, and markers for pockets, onto an
image. This is primarily used for debugging and creating visual output of the
analysis results.
"""

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import PhotoData, Rectangle
from game_class.C_pocket import Pocket
from output_utils import get_output_path

# --- Drawing Configuration ---
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_balls(image: np.ndarray, photo_data: PhotoData) -> None:
    """Draws circles and labels for all balls in PhotoData."""
    for ball in photo_data.balls:
        center_x, center_y = int(ball.center[0]), int(ball.center[1])
        radius = int(ball.radius)
        cv2.circle(image, (center_x, center_y), radius, COLOR_RED, 3)
        cv2.circle(image, (center_x, center_y), 3, COLOR_GREEN, -1)
        label = f"{center_x}, {center_y}"
        cv2.putText(
            image,
            label,
            (center_x - 20, center_y - radius - 8),
            FONT,
            0.55,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )


def _draw_rectangle(image: np.ndarray, rectangle: Rectangle) -> None:
    """Draws the table rectangle and its corner labels."""
    Debugger.log("🟦 Drawing table rectangle on image")
    points = [
        rectangle.top_left,
        rectangle.top_right,
        rectangle.bottom_right,
        rectangle.bottom_left,
    ]
    cv2.polylines(
        image,
        [np.array(points, np.int32)],
        isClosed=True,
        color=COLOR_CYAN,
        thickness=3,
    )
    for point_name, point in zip(["TL", "TR", "BR", "BL"], points):
        cv2.circle(image, point, radius=6, color=COLOR_RED, thickness=-1)
        cv2.putText(
            image,
            point_name,
            (point[0] + 8, point[1] - 8),
            FONT,
            0.6,
            COLOR_YELLOW,
            2,
        )


def _draw_pockets(image: np.ndarray, all_pockets: List[Pocket]) -> None:
    """Draws circles and labels for all detected pockets."""
    Debugger.log(f"🎯 Drawing {len(all_pockets)} pockets on table")
    for pocket in all_pockets:
        center_x, center_y = pocket.pocket_img_cordinates_on_table
        center_x, center_y = int(center_x), int(center_y)
        radius = int(pocket.radius)
        cv2.circle(image, (center_x, center_y), radius, COLOR_BLUE, 2)
        cv2.circle(image, (center_x, center_y), 4, COLOR_RED, -1)
        label = f"ID {pocket.id}"
        cv2.putText(
            image,
            label,
            (center_x + 8, center_y - 8),
            FONT,
            0.5,
            COLOR_WHITE,
            2,
        )
        Debugger.log(
            f"🕳️ Pocket {pocket.location}: Center=({center_x}, {center_y}), "
            f"Radius={radius}"
        )


def draw_balls_on_image(
    photo_data: PhotoData,
    image_path: str,
    output_path_str: str,
    rectangle: Optional[Rectangle] = None,
    all_pockets: Optional[List[Pocket]] = None,
) -> None:
    """
    Draws detected balls, table rectangle, and pockets on an image.

    This function loads a source image and overlays visualizations for the
    detected objects, saving the result to a new file.

    Args:
        photo_data: The PhotoData object containing the list of balls.
        image_path: Path to the source image to draw on.
        output_path_str: The path where the output image will be saved.
        rectangle: An optional Rectangle object to draw.
        all_pockets: An optional list of Pocket objects to draw.
    """
    Debugger.log(
        f"🖼️ Drawing balls{' and pockets' if all_pockets else ''} on image: {image_path}"
    )
    image = cv2.imread(image_path)
    if image is None:
        Debugger.error(f"❌ Failed to load image: {image_path}")
        return

    _draw_balls(image, photo_data)

    if rectangle:
        _draw_rectangle(image, rectangle)

    if all_pockets:
        _draw_pockets(image, all_pockets)

    # Ensure the output directory exists
    final_output_path = Path(get_output_path(output_path_str))
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(final_output_path), image)
    Debugger.log(f"✅ Saved annotated image to → {final_output_path}")
