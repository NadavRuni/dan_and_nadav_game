from output_utils import get_output_path
import cv2
import numpy as np
import json
from .Debugger import Debugger
import os
from analyzer_table.launcher_helper.json_models import (
    PhotoData,
    Origin,
    Rectangle,
    Ball,
)
from const_numbers import *
import shutil


def save_debug_copy(img, file_name, sub_dir=""):
    """
    Saves a debug copy of the image to the debug directory.
    """
    save_path = get_output_path(file_name, sub_dir=sub_dir)
    cv2.imwrite(save_path, img)
    print(f"🟢 Saved debug copy to: {save_path}")


def detect_balls_opencv(
    input_dir, output_dir, parts_info, main_width, main_height
) -> list[PhotoData]:
    Debugger.log("Starting OpenCV detection (with radius filtering)")
    os.makedirs(output_dir, exist_ok=True)
    total_balls = 0
    all_photos = []
    for part_info in parts_info:
        file = part_info["file_name"]
        if file == "cut_main.png":
            Debugger.log("Main image flag is set to True")
            param2_value = 25
        else:
            param2_value = 50
        path = os.path.join(input_dir, file)
        Debugger.log(f"[OpenCV] Processing {path}")
        img = cv2.imread(path)
        if img is None:
            print("❌ התמונה לא נטענה!")
        else:
            height, width = img.shape[:2]
            print(f"📏 Image size: width={width}, height={height}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        print(
            f"[OpenCV] Running HoughCircles with radius ~{get_ball_radius()} ± {get_ball_radius_determinate()}"
        )
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(get_ball_radius() * 2),
            param1=60,
            param2=param2_value,
            minRadius=int(get_ball_radius() - 2 * get_ball_radius_determinate()),
            maxRadius=int(get_ball_radius() + 2 * get_ball_radius_determinate()),
        )
        found_balls = 0
        balls = []
        os.makedirs("out_debug", exist_ok=True)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0, :]:
                if not (
                    get_ball_radius() - get_ball_radius_determinate()
                    <= r
                    <= get_ball_radius() + get_ball_radius_determinate()
                ):
                    continue
                cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 3)
                cv2.putText(
                    img,
                    f"{r}",
                    (x - 15, y - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                found_balls += 1
                global_x = part_info["origin_x"] + x
                global_y = part_info["origin_y"] + y
                balls.append(Ball(center=(int(global_x), int(global_y)), radius=int(r)))
                os.makedirs("out_debug", exist_ok=True)
                debug_img = img.copy()
                cv2.circle(debug_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(debug_img, (int(x), int(y)), 2, (0, 0, 255), 3)
                info_text = f"x={int(global_x)}, y={int(global_y)}, r={int(r)}"
                cv2.putText(
                    debug_img,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imwrite(get_output_path(f"circle_{found_balls}.png"), debug_img)
        output_img_path = get_output_path(f"detect_{file}")
        Debugger.log(
            f"[OpenCV] {found_balls} balls detected, saving image to {output_img_path}"
        )
        cv2.imwrite(output_img_path, img)
        save_debug_copy(img, file)
        total_balls += found_balls
        x0, y0 = part_info["origin_x"], part_info["origin_y"]
        x1 = x0 + part_info["width"]
        y1 = y0 + part_info["height"]
        photo = PhotoData(
            cut_name=file,
            origin=Origin(x=x0, y=y0),
            rectangle=Rectangle(
                top_left=(x0, y1),
                top_right=(x1, y1),
                bottom_left=(x0, y0),
                bottom_right=(x1, y0),
            ),
            balls=balls,
        )
        json_path = os.path.join(output_dir, f"{file.replace('.png', '.json')}")
        photo.save_json(json_path)
        Debugger.log(f"[OpenCV] Saved metadata: {json_path}")
        all_photos.append(photo)
    Debugger.warn(f"[OpenCV] Total detected balls across all parts: {total_balls}")
    return all_photos
