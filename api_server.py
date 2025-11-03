from pathlib import Path
from fastapi import FastAPI, UploadFile, BackgroundTasks, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import shutil
import asyncio
import os
import json
import numpy as np

# ניסיון לייבא torch רק אם קיים
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from dan.detect_table_rectangle import detect_table_rectangle
from dan.pipe_Line import start_pipe_line
from dan.build_table_from_image import start_build_table_from_img
<<<<<<< HEAD
from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH ,RECTANGLE_JSON_PATH
from analyzer_table.launcher_helper.data_to_rectangle import create_rectangle_from_data
from dan.detect_table_rectangle import update_table_size_from_rectangle
=======
from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH, RECTANGLE_JSON_PATH


# === Custom encoders for numpy / torch / path ===
CUSTOM_ENCODERS = {
    Path: lambda p: str(p),
    np.ndarray: lambda a: a.tolist(),
    np.int64: int, np.int32: int, np.int16: int, np.int8: int,
    np.uint64: int, np.uint32: int, np.uint16: int, np.uint8: int,
    np.float64: float, np.float32: float, np.float16: float,
}
if _HAS_TORCH:
    CUSTOM_ENCODERS.update({
        torch.Tensor: lambda t: t.detach().cpu().tolist(),
    })


# === FASTAPI APP ===
>>>>>>> 43ae133 (switching branch - changes for v4)
app = FastAPI()

# מחיקה אוטומטית של קובץ ניווט ישן בזמן עליית השרת
nav_file = Path(__file__).resolve().parent / "frontend_nav.json"
if nav_file.exists():
    print(f"🧹 Deleting leftover frontend_nav.json on startup: {nav_file}")
    nav_file.unlink()

# חשיפה של קבצי frontend כסטטיים
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/frontend_nav.json")
async def get_nav_file():
    nav_path = Path(__file__).resolve().parent / "frontend_nav.json"
    print(f"[DEBUG] Trying to serve nav file from: {nav_path}")

    if nav_path.exists():
        print("✅ Found frontend_nav.json — serving now.")
<<<<<<< HEAD
        # לא מוחקים כאן — זה שומר על יציבות
=======
        background_tasks.add_task(delete_file_safely, nav_path)
>>>>>>> 43ae133 (switching branch - changes for v4)
        return FileResponse(nav_path)
    else:
        # אין קובץ – מחזירים תשובה שקטה (לא שגיאה)
        return JSONResponse(status_code=204, content=None)



def delete_file_safely(path: Path):
    try:
        if path.exists():
            os.remove(path)
            print(f"🧹 Deleted old frontend_nav.json safely: {path}")
    except Exception as e:
        print(f"⚠️ Failed to delete nav file: {e}")


# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === PATHS ===
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUT_DIR = BASE_DIR / "photos/output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

executor = asyncio.get_event_loop().run_in_executor


# === API ROUTES ===
@app.post("/api/run_pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

<<<<<<< HEAD
    # הפעלת זיהוי מלבן ברקע — לא חוסם
    print("[DEBUG] start run_pipeline")

    loop = asyncio.get_event_loop()

    loop.run_in_executor(executor, detect_table_rectangle, str(file_path))
=======
    # הפעלת זיהוי מלבן ברקע
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, detect_table_rectangle, str(file_path))
>>>>>>> 43ae133 (switching branch - changes for v4)

    return {"status": "processing"}


@app.post("/api/confirm_rectangle")
async def confirm_rectangle(data: dict):
    """
    שלב שני: המשתמש אישר את המלבן או תיקן אותו -> מריץ את ה-pipeline המלא.
    מחזירים JSON 'טהור' בלבד.
    """
    try:
        print("[DEBUG] Received rectangle confirmation data:", data)
        rec=create_rectangle_from_data(data)
        update_table_size_from_rectangle(rec) if rec else None

        rect_path = Path(BASE_DIR / RECTANGLE_JSON_PATH)
        print("[DEBUG] Writing rectangle data to:", rect_path)
        rect_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        image_path = data.get("image_path")
        if not image_path:
            return JSONResponse({"error": "Missing image_path"}, status_code=400)

        print("[DEBUG] Starting full pipeline for image:", image_path)
        pipeline_result = await start_pipe_line(image_path)
        table_result = start_build_table_from_img()

        safe_payload = {
            "status": "ok",
            "pipeline": pipeline_result,
            "table": table_result,
        }

        # המרה לסוגים שניתנים לסריאליזציה
        safe = jsonable_encoder(safe_payload, custom_encoder=CUSTOM_ENCODERS)
        return JSONResponse(content=safe, status_code=200)

    except Exception as e:
        import traceback
<<<<<<< HEAD
        print("❌ Exception in confirm_rectangle:")
        traceback.print_exc()  # <== זה מדפיס את ה-stack trace המלא לקונסול
=======
        traceback.print_exc()
>>>>>>> 43ae133 (switching branch - changes for v4)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/get_image")
async def get_image(path: str):
    """שירות הצגת תמונה"""
    file_name = Path(path).name
    file_path = UPLOAD_DIR / file_name
    if not file_path.exists():
        return JSONResponse({"error": f"Image not found: {file_path}"}, status_code=404)
    return FileResponse(str(file_path))


@app.get("/api/get_output")
async def get_output():
    if OUTPUT_IMAGE_PATH.exists():
        return {"output_url": f"/static/{OUTPUT_IMAGE_PATH.name}"}
    return {"error": "No output image found"}


@app.get("/api/get_output_contact")
async def get_output_contact():
    if OUTPUT_CONTACT_VIEW_PATH.exists():
        return {"output_url": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}"}
    return {"error": "No output contact image found"}


# === STATIC FILES ===
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
