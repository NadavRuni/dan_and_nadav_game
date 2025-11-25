from output_utils import get_output_path
import os
import cv2
from typing import List
from dataclasses import dataclass
import numpy as np
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Ball


def crop_and_save_balls(
    image_path: str, balls: List[Ball], output_dir: str = "out/balls"
) -> None:
    Debugger.log(f"🖼️ Cropping and saving {len(balls)} balls from {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        Debugger.error(f"❌ Failed to load image from {image_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    h, w = img.shape[:2]
    for i, ball in enumerate(balls, start=1):
        cx, cy = map(int, ball.center)
        r = int(ball.radius * 1.3)
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            Debugger.warn(f"⚠️ Empty ROI for ball at center {ball.center}")
            continue
        filename = f"ball_{i}_{cx}_{cy}.png"
        ball_path = get_output_path(filename, sub_dir="balls")
        cv2.imwrite(ball_path, roi)
        ball.single_ball_path = ball_path
        Debugger.log(f"💾 Saved ball #{i} → {ball_path}")
    Debugger.log("✅ Finished cropping all balls.")
