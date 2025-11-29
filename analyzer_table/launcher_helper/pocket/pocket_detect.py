from output_utils import get_output_path
import cv2
import os
from typing import Tuple, List
from analyzer_table.launcher_helper.json_models import (
    Rectangle,
)
from game_class.C_pocket import Pocket
from const_numbers import *


def analyze_table_pockets(
    img_path: str,
    rectangle: Rectangle,
    output_dir: str = "out/pockets",
    half_size: int = get_crop_half_size(),
) -> List[Pocket]:
    if get_use_predicted_pockets():
        print("✅ Using previously detected pockets.")
        return get_detected_pockets()
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Failed to load image from {img_path}")
    os.makedirs(output_dir, exist_ok=True)
    h, w = img.shape[:2]
    top_middle = (
        int((rectangle.top_left[0] + rectangle.top_right[0]) / 2),
        int((rectangle.top_left[1] + rectangle.top_right[1]) / 2),
    )
    bottom_middle = (
        int((rectangle.bottom_left[0] + rectangle.bottom_right[0]) / 2),
        int((rectangle.bottom_left[1] + rectangle.bottom_right[1]) / 2),
    )
    pocket_positions = {
        "TL": rectangle.top_left,
        "TM": top_middle,
        "TR": rectangle.top_right,
        "BL": rectangle.bottom_left,
        "BM": bottom_middle,
        "BR": rectangle.bottom_right,
    }
    pocket_list: List[Pocket] = []
    for i, (name, (cx, cy)) in enumerate(pocket_positions.items()):
        x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
        x2, y2 = min(w, cx + half_size), min(h, cy + half_size)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"⚠️ Skipping {name} – ROI is empty (close to edge)")
            continue
        filename = f"{name}.png"
        out_path = get_output_path(filename, sub_dir="pockets")
        cv2.imwrite(out_path, roi)
        pocket = Pocket(
            id=i,
            center=(cx, cy),
            radius=half_size,
            pocket_img_path=out_path,
            pocket_img_cordinates_on_table=(cx, cy),
            location=name,
        )
        pocket_list.append(pocket)

    print(f"✅ Created table_pockets with {len(pocket_list)} pockets.")
    return pocket_list
