from output_utils import get_output_path
import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Ball


def analyze_ball_brightness(
    image_path: str, balls: List[Ball], output_dir: str = "out/balls"
) -> Tuple[Optional[Ball], Optional[Ball]]:
    Debugger.log(f"🧠 Analyzing {len(balls)} balls to find the whitest and darkest...")
    img = cv2.imread(image_path)
    if img is None:
        Debugger.error(f"❌ Failed to load image from {image_path}")
        return None, None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    os.makedirs(output_dir, exist_ok=True)
    whitest_ball = None
    darkest_ball = None
    max_brightness = -1
    min_brightness = 9999
    for i, ball in enumerate(balls, 1):
        cx, cy = map(int, ball.center)
        r = int(ball.radius * 1.3)
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            Debugger.warn(f"⚠️ Ball #{i} ROI empty, skipping.")
            continue
        v_channel = roi[:, :, 2]
        mean_brightness = np.mean(v_channel)
        Debugger.log(
            f"⚪ Ball #{i}: center=({cx},{cy}), r={r}, brightness={mean_brightness:.2f}"
        )
        ball_img = img[y1:y2, x1:x2]
        ball_path = get_output_path(
            f"ball_{i}_({cx}_{cy})_r{r}.png", sub_dir="ball_color"
        )
        cv2.imwrite(ball_path, ball_img)
        if mean_brightness > max_brightness:
            max_brightness = mean_brightness
            whitest_ball = ball
        if mean_brightness < min_brightness:
            min_brightness = mean_brightness
            darkest_ball = ball
    if whitest_ball:
        Debugger.log(
            f"🏆 Whitest ball → center={whitest_ball.center}, r={whitest_ball.radius}, brightness={max_brightness:.2f}"
        )
    if darkest_ball:
        Debugger.log(
            f"⚫ Darkest ball → center={darkest_ball.center}, r={darkest_ball.radius}, brightness={min_brightness:.2f}"
        )
    Debugger.log(f"🗂️ All ball images saved under: {output_dir}")
    return whitest_ball, darkest_ball
