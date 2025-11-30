"""
Orchestrates the ball detection process by splitting an image into parts.

This module contains the main logic for dividing a large image of a pool table
into smaller, overlapping parts. It then runs the OpenCV ball detection
pipeline on each part, as well as on the main image, and consolidates the
results.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from PIL import Image

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.detect_ball.ball_ditect import detect_balls_opencv
from analyzer_table.launcher_helper.json_models import PhotoData


def _prepare_directories(base_dir: Path) -> Tuple[Path, Path]:
    """
    Creates the necessary output directories for the analysis.

    Args:
        base_dir: The base directory for the detect_ball module.

    Returns:
        A tuple containing the paths to the main output directory and the
        OpenCV-specific detection directory.
    """
    output_dir = base_dir / "analyzer"
    detect_dir_opencv = base_dir / "detect_analyzer_opencv"
    output_dir.mkdir(exist_ok=True)
    detect_dir_opencv.mkdir(exist_ok=True)
    Debugger.log("Created output directories for analysis.")
    return output_dir, detect_dir_opencv


def _separate_main_and_sub_photos(
    photo_data_list: List[PhotoData],
) -> Tuple[Optional[PhotoData], List[PhotoData]]:
    """
    Separates the analysis result of the main image from the sub-images.

    Args:
        photo_data_list: A list of PhotoData objects from the analysis.

    Returns:
        A tuple containing the PhotoData for the main image and a list of
        PhotoData objects for the sub-images.
    """
    main_photo = next(
        (p for p in photo_data_list if p.cut_name == "cut_main.png"), None
    )
    sub_photos = [p for p in photo_data_list if p.cut_name != "cut_main.png"]
    return main_photo, sub_photos


def run_full_analysis(
    image_path: str,
) -> Tuple[Optional[List[PhotoData]], Optional[PhotoData]]:
    """
    Runs the full analysis pipeline on a single image.

    This involves splitting the image, running detection on each part, and
    returning the consolidated results.

    Args:
        image_path: The path to the image to be analyzed.

    Returns:
        A tuple containing a list of PhotoData from the sub-images and a
        single PhotoData from the main image. Returns (None, None) if the
        initial path is invalid.
    """
    Debugger.log("Starting main analysis runner.")
    if not image_path:
        Debugger.error("Image path is missing.")
        return None, None
    Debugger.log(f"Found image path: {image_path}")

    base_dir = Path(__file__).resolve().parent
    output_dir, detect_dir_opencv = _prepare_directories(base_dir)

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        Debugger.error(f"Image file not found at {image_path}")
        return None, None

    width, height = img.size

    # Save a copy of the main image for the detection pipeline
    main_cut_path = output_dir / "cut_main.png"
    img.save(main_cut_path)
    Debugger.log("Saved main image as cut_main.png for detection")

    # Split the image into overlapping parts and save them
    parts_info = split_image_into_parts(img)
    Debugger.log(f"Image split into {len(parts_info)} parts.")
    for i, part_info in enumerate(parts_info, start=1):
        filename = output_dir / f"cut_{i}.png"
        part_info["image"].save(filename)
        part_info["file_name"] = filename.name
        part_info.pop("image")  # Remove PIL image object before detection

    # Add the main image to the list of parts to be processed
    main_info = {
        "file_name": "cut_main.png",
        "origin_x": 0,
        "origin_y": 0,
        "width": width,
        "height": height,
    }
    parts_info.append(main_info)
    Debugger.log("Added main image to the detection list.")

    # Run OpenCV detection on all parts
    photo_data_list = detect_balls_opencv(
        str(output_dir), str(detect_dir_opencv), parts_info
    )
    Debugger.warn("✅ Finished analyzing main image and all parts with OpenCV")

    main_photo, sub_photos = _separate_main_and_sub_photos(photo_data_list)
    return sub_photos, main_photo


def split_image_into_parts(
    image: Image.Image, overlap_ratio: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Splits an image into a grid of smaller, overlapping parts.

    Args:
        image: The PIL Image object to be split.
        overlap_ratio: The ratio of overlap between adjacent parts.

    Returns:
        A list of dictionaries, where each dictionary contains the PIL image
        part and its position metadata.
    """
    Debugger.log(f"Splitting image with overlap ratio {overlap_ratio}")
    width, height = image.size
    rows, cols = 4, 6  # Grid size

    part_w = width // cols
    part_h = height // rows
    overlap_w = int(part_w * overlap_ratio)
    overlap_h = int(part_h * overlap_ratio)

    parts_info = []
    for row in range(rows):
        for col in range(cols):
            # Calculate coordinates for cropping, ensuring they are within bounds
            left = max(0, col * part_w - overlap_w)
            top = max(0, row * part_h - overlap_h)
            right = min(width, (col + 1) * part_w + overlap_w)
            bottom = min(height, (row + 1) * part_h + overlap_h)

            # PIL crop uses (left, top, right, bottom)
            part_image = image.crop((left, top, right, bottom))

            part_info = {
                "image": part_image,
                "origin_x": left,
                "origin_y": top,  # Y-origin is the top edge
                "width": right - left,
                "height": bottom - top,
            }
            parts_info.append(part_info)
            Debugger.log(
                f"Created part ({row},{col}) -> origin=({left},{top}) "
                f"size=({right - left},{bottom - top})"
            )
    return parts_info
