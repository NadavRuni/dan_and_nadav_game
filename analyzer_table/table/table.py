"""
Manages the user-in-the-loop flow for confirming the detected table rectangle.

This module contains a function that triggers a frontend page and then waits
for the user to manually confirm or adjust the detected pool table boundaries.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.launcher_helper.json_models import Rectangle

# Warning: This file-based polling mechanism is fragile and inefficient.
# It is not suitable for a production environment, especially with a concurrent
# web server, as it involves blocking waits and is prone to race conditions.

# Define paths relative to the project's base directory
PROJECT_BASE_DIR = Path(__file__).resolve().parents[2]
NAV_FILE_PATH = PROJECT_BASE_DIR / "frontend_nav.json"
CACHE_FILE_PATH = PROJECT_BASE_DIR / "rectangles_cache.json"

WAIT_TIMEOUT_SECONDS = 300


def confirm_or_correct_rectangle(
    image_path: str, rectangle: Optional[Rectangle]
) -> Optional[Rectangle]:
    """
    Waits for a user to confirm or correct a rectangle via a frontend interface.

    This function works by:
    1. Deleting any old navigation or cache files.
    2. Creating a 'frontend_nav.json' file that tells the frontend to navigate
       to the confirmation page.
    3. Entering a blocking loop, polling every second for the existence of a
       'rectangles_cache.json' file, which the frontend is expected to create
       upon user submission.
    4. Parsing the new rectangle data from the cache file if it appears.

    Args:
        image_path: The path to the image being confirmed.
        rectangle: The initial detected rectangle (currently unused).

    Returns:
        A new Rectangle object confirmed by the user, or None if the process
        times out.
    """
    Debugger.log(f"NAV_FILE path: {NAV_FILE_PATH}")
    Debugger.log(f"CACHE_FILE path: {CACHE_FILE_PATH}")

    # Clean up old files before starting
    if NAV_FILE_PATH.exists():
        NAV_FILE_PATH.unlink()
    if CACHE_FILE_PATH.exists():
        CACHE_FILE_PATH.unlink()

    # Create a relative URL for the frontend to use
    relative_image_path = os.path.relpath(image_path, PROJECT_BASE_DIR)
    confirmation_url = f"/frontend/confirm_rectangle.html?image={relative_image_path}"
    Debugger.log(f"🌐 Navigating frontend to: {confirmation_url}")

    # Write the navigation file to trigger the frontend
    try:
        with open(NAV_FILE_PATH, "w") as f:
            json.dump({"navigate_to": confirmation_url}, f)
        Debugger.log(f"✅ Wrote navigation file: {NAV_FILE_PATH}")
    except Exception as e:
        Debugger.error(f"❌ Failed to write navigation file: {e}")
        return None

    Debugger.log("⏳ Waiting for user to confirm rectangle in browser...")

    for _ in range(WAIT_TIMEOUT_SECONDS):
        if CACHE_FILE_PATH.exists():
            try:
                with open(CACHE_FILE_PATH, "r") as f:
                    data = json.load(f)
                points = data.get("points", [])
                if len(points) == 4:
                    Debugger.log("✅ User confirmed rectangle via browser.")
                    # Note: The point order from the frontend might be unreliable.
                    # This assumes a specific order that may not be guaranteed.
                    return Rectangle(
                        top_left=(int(points[0]["x"]), int(points[0]["y"])),
                        top_right=(int(points[1]["x"]), int(points[1]["y"])),
                        bottom_right=(int(points[2]["x"]), int(points[2]["y"])),
                        bottom_left=(int(points[3]["x"]), int(points[3]["y"])),
                    )
            except (json.JSONDecodeError, KeyError) as e:
                Debugger.warn(f"⚠️ Error reading cache file, will retry: {e}")

        time.sleep(1)

    Debugger.error("❌ Timeout waiting for rectangle confirmation.")
    return None
