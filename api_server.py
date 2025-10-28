from pathlib import Path
from fastapi import  FastAPI, UploadFile,BackgroundTasks, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import asyncio
import os
import json
from dan.detect_table_rectangle import detect_table_rectangle  # פונקציה חדשה
from dan.pipe_Line import start_pipe_line
from dan.build_table_from_image import start_build_table_from_img
from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH ,RECTANGLE_JSON_PATH

app = FastAPI()
# מחיקה אוטומטית של קובץ ניווט ישן בזמן עליית השרת
nav_file = Path(__file__).resolve().parent / "frontend_nav.json"
if nav_file.exists():
    print(f"🧹 Deleting leftover frontend_nav.json on startup: {nav_file}")
    nav_file.unlink()


# חשיפה של קבצי frontend כסטטיים
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/frontend_nav.json")
async def get_nav_file(background_tasks: BackgroundTasks):
    nav_path = Path(__file__).resolve().parent / "frontend_nav.json"
    print(f"[DEBUG] Trying to serve nav file from: {nav_path}")

    if nav_path.exists():
        print("✅ Found frontend_nav.json — serving now.")
        # מחיקה מתוזמנת לאחר השליחה
        background_tasks.add_task(delete_file_safely, nav_path)
        return FileResponse(nav_path)
    else:
        print("❌ frontend_nav.json not found!")
        return JSONResponse(status_code=404, content={"error": "frontend_nav.json not found"})

def delete_file_safely(path: Path):
    try:
        if path.exists():
            os.remove(path)
            print(f"🧹 Deleted old frontend_nav.json safely: {path}")
    except Exception as e:
        print(f"⚠️ Failed to delete nav file: {e}")
        
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUT_DIR = BASE_DIR / "photos/output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

@app.post("/api/run_pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # הפעלת זיהוי מלבן ברקע — לא חוסם
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, detect_table_rectangle, str(file_path))

    # מיד מחזיר תגובה ללקוח
    return {"status": "processing"}

@app.post("/api/confirm_rectangle")
async def confirm_rectangle(data: dict):
    """
    שלב שני: המשתמש אישר את המלבן או תיקן אותו
    -> מריץ עכשיו את ה-pipeline המלא.
    """
    try:
        print("[DEBUG] Received rectangle confirmation data:", data)
        rect_path = Path(BASE_DIR / RECTANGLE_JSON_PATH)
        print("[DEBUG] Writing rectangle data to:", rect_path)
        rect_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        image_path = data.get("image_path")
        if not image_path:
            return JSONResponse({"error": "Missing image_path"}, status_code=400)

        # עכשיו נריץ את ה-pipeline המלא
        print("[DEBUG] Starting full pipeline for image:", image_path)
        pipeline_result = await start_pipe_line(image_path)
        table_result = start_build_table_from_img()

        return {"status": "ok", "pipeline": pipeline_result, "table": table_result}

    except Exception as e:
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



# Static
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
