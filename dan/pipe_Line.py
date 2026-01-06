"""
Main image recognition pipeline for pool table analysis.

This module orchestrates a complex pipeline that takes an image of a pool table
and outputs a structured JSON file containing the positions of the balls,
pockets, and other game state information.

Warning:
    This file has a large number of diverse responsibilities, including object
    detection, geometric calculations, and data serialization. It should be
    refactored into smaller, more focused modules.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from analyzer_table.balls_from_image import full_analyzer_pipeline
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import AnalyzerResult, Ball
from const_numbers import OUTPUT_JSON_PATH
from game_class.C_pocket import Pocket

# --- Configuration Constants ---
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"

# YOLOv8 Configuration
MODEL_PATH = "yolov8n.pt"
SPORTS_BALL_CLASS_ID = 32
YOLO_CONFIDENCE_THRESHOLD = 0.01
YOLO_IOU_THRESHOLD = 0.40
YOLO_IMAGE_SIZE = 1536
YOLO_MAX_DETECTIONS = 300
USE_TEST_TIME_AUGMENTATION = True


def _create_output_json(
    image_path: str, analyzer_result: AnalyzerResult, image_size: tuple
) -> dict:
    """
    Formats the raw analysis result into the final JSON structure.

    Args:
        image_path: Path to the original image.
        analyzer_result: The result from the core analysis pipeline.
        image_size: A tuple (width, height) of the original image.

    Returns:
        A dictionary formatted for the final JSON output.
    """
    width, height = image_size
    balls_json = []
    for i, ball in enumerate(analyzer_result.balls):
        balls_json.append(
            {
                "index": i,
                "type": getattr(ball, "final_color", "unknown"),
                "center_px": {"x": float(ball.center[0]), "y": float(ball.center[1])},
                "radius_px": float(ball.radius),
            }
        )

    pockets_json = {}
    for pocket in analyzer_result.pockets:
        pockets_json[pocket.location] = {
            "x": float(pocket.pocket_img_cordinates_on_table[0]),
            "y": float(pocket.pocket_img_cordinates_on_table[1]),
            "radius": float(pocket.radius),
        }

    return {
        "image_path": image_path,
        "image_size_px": {"width": float(width), "height": float(height)},
        "balls": balls_json,
        "white_ball": asdict(analyzer_result.white) if analyzer_result.white else None,
        "black_ball": asdict(analyzer_result.black) if analyzer_result.black else None,
        "pockets": pockets_json,
    }


def run_image_recognition_pipeline(image_path: str) -> dict:
    """
    Executes the full analysis pipeline on an image and saves the result as a JSON file.

    Args:
        image_path: The path to the image to analyze.

    Returns:
        A dictionary containing the structured analysis results.

    Raises:
        FileNotFoundError: If the image cannot be read.
        TypeError: If the core pipeline returns an unexpected data type.
    """
    Debugger.log(f"🚀 Starting image recognition for: {image_path}")

    # 1. Run the core analysis pipeline from the 'analyzer_table' module.
    analyzer_result: AnalyzerResult = full_analyzer_pipeline(image_path)
    if not isinstance(analyzer_result, AnalyzerResult):
        raise TypeError(
            "❌ full_analyzer_pipeline must return an AnalyzerResult object"
        )

    # 2. Read the image to get its dimensions.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Cannot read image from path: {image_path}")
    height, width = image.shape[:2]

    # 3. Format the results into the final JSON structure.
    result_json = _create_output_json(image_path, analyzer_result, (width, height))

    # 4. Write the result to the output JSON file.
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    Debugger.log(f"✅ JSON result saved successfully → {OUTPUT_JSON_PATH}")
    return result_json


async def start_pipe_line(image_path: str) -> dict:
    """
    An async wrapper for the image recognition pipeline.

    Handles downloading the image if a URL is provided.

    Args:
        image_path: The path or URL to the image.

    Returns:
        A dictionary containing the structured analysis results.
    """
    print("start_pipe_line", image_path)

    # If the image_path is a URL, download it to a local temporary file.
    if image_path.startswith("http"):
        Debugger.log(f"[DEBUG] Downloading remote image: {image_path}")
        local_name = Path(image_path.split("/")[-1]).name
        local_path = UPLOAD_DIR / local_name
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        image_path = str(local_path)
        Debugger.log(f"[DEBUG] Saved remote image locally to: {image_path}")

    # Run the main pipeline.
    return run_image_recognition_pipeline(image_path)


if __name__ == "__main__":
    # Example usage
    DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / "photos" / "img_start.jpeg"
    if not DEFAULT_IMAGE.exists():
        raise FileNotFoundError(f"Default image not found: {DEFAULT_IMAGE}")
    run_image_recognition_pipeline(str(DEFAULT_IMAGE))
