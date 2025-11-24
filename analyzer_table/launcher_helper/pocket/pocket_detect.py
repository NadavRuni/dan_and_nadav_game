from output_utils import get_output_path
import cv2
import os
import numpy as np
from dataclasses import field
from typing import Tuple, List
from analyzer_table.launcher_helper.json_models import (
    Pockets_img_paths,
    Pocket_Location_On_Table,
    Pocket,
    Rectangle,
    table_pockets,
)
from const_numbers import *


def analyze_table_pockets(
    img_path: str,
    rectangle: Rectangle,
    output_dir: str = "out/pockets",
    half_size: int = get_crop_half_size(),
) -> table_pockets:
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
        "top_left": rectangle.top_left,
        "top_middle": top_middle,
        "top_right": rectangle.top_right,
        "bottom_left": rectangle.bottom_left,
        "buttom_middle": bottom_middle,
        "bottom_right": rectangle.bottom_right,
    }
    pockets_img_paths = Pockets_img_paths()
    pocket_list: List[Pocket] = []
    for i, (name, (cx, cy)) in enumerate(pocket_positions.items(), start=1):
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
        setattr(pockets_img_paths, f"{name}_path", out_path)
        pocket = Pocket(
            pocket_id=i,
            pocket_center=(cx, cy),
            pocker_radius=half_size,
            pocket_img_path=out_path,
            pocket_img_cordinates_on_table=(cx, cy),
            pocket_loacation_on_table=getattr(
                Pocket_Location_On_Table, name, Pocket_Location_On_Table.unknown
            ),
        )
        pocket_list.append(pocket)
    result = table_pockets(
        pockets_img_paths=pockets_img_paths,
        pocket_list=pocket_list,
    )
    print(f"✅ Created table_pockets with {len(pocket_list)} pockets.")
    return result
