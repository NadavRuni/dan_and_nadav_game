from pathlib import Path
import json
import time
import os
from analyzer_table.launcher_helper.json_models import Rectangle


def confirm_or_correct_rectangle(
    image_path: str, rectangle: Rectangle | None
) -> Rectangle | None:
    # 🧭 נשתמש בנתיב של קובץ השרת הראשי (api_server.py)
    NAV_FILE = Path(__file__).resolve().parents[2] / "frontend_nav.json"
    CACHE_FILE = Path(__file__).resolve().parents[2] / "rectangles_cache.json"

    print(f"[DEBUG] NAV_FILE path: {NAV_FILE}")
    print(f"[DEBUG] CACHE_FILE path: {CACHE_FILE}")

    # ניקוי ישן
    if NAV_FILE.exists():
        NAV_FILE.unlink()
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()

    # ניצור URL יחסי תקין
    relative_image_path = os.path.relpath(
        image_path, Path(__file__).resolve().parents[2]
    )
    url = f"/frontend/confirm_rectangle.html?image={relative_image_path}"
    print(f"🌐 Navigate frontend to: {url}")

    # ✨ כותב את קובץ הניווט
    try:
        with open(NAV_FILE, "w") as f:
            json.dump({"navigate_to": url}, f)
        print(f"✅ Wrote navigation file: {NAV_FILE}")
    except Exception as e:
        print(f"❌ Failed to write NAV_FILE: {e}")

    print("⏳ Waiting for user confirmation...")

    waited = 0
    while waited < 300:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                points = data.get("points", [])
                if len(points) == 4:
                    print("✅ User confirmed rectangle via browser.")
                    return Rectangle(
                        top_left=(int(points[0]["x"]), int(points[0]["y"])),
                        top_right=(int(points[1]["x"]), int(points[1]["y"])),
                        bottom_right=(int(points[2]["x"]), int(points[2]["y"])),
                        bottom_left=(int(points[3]["x"]), int(points[3]["y"])),
                    )
        time.sleep(1)
        waited += 1

    print("❌ Timeout waiting for rectangle confirmation.")
    return None
