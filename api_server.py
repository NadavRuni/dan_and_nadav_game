from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import shutil
import asyncio
from dataclasses import asdict
import os
import json
import requests
from analyzer_table.crop_table import crop_image_by_rectangle

from dan.detect_table_rectangle import detect_table_rectangle
from dan.pipe_Line import start_pipe_line
from dan.build_table_from_image import start_build_table_from_img
from const_numbers import *
from analyzer_table.launcher_helper.data_to_rectangle import create_rectangle_from_data
from dan.detect_table_rectangle import update_table_size_from_rectangle

# ✅ יצירת אפליקציה עם Response כברירת מחדל
app = FastAPI(default_response_class=Response)

# ✅ הפעלת CORS (פעם אחת בלבד)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ניקוי קובץ ניווט ישן
nav_file = Path(__file__).resolve().parent / "frontend_nav.json"
if nav_file.exists():
    print(f"🧹 Deleting leftover frontend_nav.json on startup: {nav_file}")
    nav_file.unlink()

# תיקיות
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
executor = ThreadPoolExecutor()

# ✅ קובץ ניווט — עכשיו זה באמת 100% תקין
@app.get("/frontend_nav.json")
async def get_nav_file():
    nav_path = Path(__file__).resolve().parent / "frontend_nav.json"
    print(f"[DEBUG] Trying to serve nav file from: {nav_path}")

    if nav_path.exists():
        print("✅ Found frontend_nav.json — serving now.")
        return FileResponse(nav_path)

    # 204 תקין — בלי שום תוכן או header נוסף
    return Response(status_code=204)

# ✅ הרצת פייפליין
@app.post("/api/run_pipeline")
async def run_pipeline(request: Request, file: UploadFile = None):
    try:
        file_path = None

        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            image_url = data.get("image_url")
            if not image_url:
                return JSONResponse({"error": "Missing 'image_url'"}, status_code=400)

            filename = image_url.split("/")[-1]
            file_path = UPLOAD_DIR / filename
            print(f"[DEBUG] Downloading image from URL: {image_url}")
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

        elif file is not None:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        else:
            return JSONResponse({"error": "No file or image_url provided"}, status_code=400)

        print(f"[DEBUG] start run_pipeline on {file_path}")
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, detect_table_rectangle, str(file_path))

        return JSONResponse({"status": "processing", "file_path": str(file_path)})

    except Exception as e:
        print("[ERROR]", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ✅ אישור מלבן
@app.post("/api/confirm_rectangle")
async def confirm_rectangle(data: dict):
    try:
        print("[DEBUG] Received rectangle confirmation data:", data)
        rec = create_rectangle_from_data(data)

        
        image_path = data.get("image_path")
        if not image_path:
            return JSONResponse({"error": "Missing image_path"}, status_code=400)

        cropped_path, rec_scaled = crop_image_by_rectangle(rec, image_path, UPLOAD_DIR, data)
        update_table_size_from_rectangle(rec_scaled)

        # שמירת המלבן החדש לקובץ JSON
        rect_path = Path(BASE_DIR / RECTANGLE_JSON_PATH)
        rect_path.write_text(json.dumps(asdict(rec_scaled), indent=2), encoding="utf-8")
        print(f"[DEBUG] ✅ Saved scaled rectangle to: {rect_path}")

     
        if not cropped_path:
            return JSONResponse({"error": "Failed to crop image"}, status_code=500)

        print("[DEBUG] Starting full pipeline for image:", cropped_path)
        pipeline_result = await start_pipe_line(str(cropped_path))
        table_result = start_build_table_from_img()

        return JSONResponse({
            "status": "ok",
            "pipeline": pipeline_result,
            "table": table_result
        })

    except Exception as e:
        import traceback
        print("❌ Exception in confirm_rectangle:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ✅ קבלת תמונה
@app.get("/api/get_image")
async def get_image(path: str):
    file_name = Path(path).name
    file_path = UPLOAD_DIR / file_name
    if not file_path.exists():
        return JSONResponse({"error": f"Image not found: {file_path}"}, status_code=404)
    return FileResponse(str(file_path))
@app.post("/api/get_output")
async def get_output(request: Request):
    """
    מחזיר קישור מלא לתמונה שנוצרה — נגיש גם דרך ngrok.
    כעת לא משתמש בנתיב /static אלא מפנה ישירות לתיקייה photos/output.
    """
    # נבדוק שהקובץ קיים
    if not OUTPUT_IMAGE_PATH.exists():
        print("[DEBUG] ❌ No output image found")
        return JSONResponse({"error": "No output image found"}, status_code=404)

    # זיהוי כתובת הבסיס
    base_url = str(request.base_url).rstrip("/")

    # אם הבקשה מגיעה מ-ngrok, נשתמש בכתובת הקבועה
    # if "ngrok" in base_url:
    #     base_url = "https://sunbeamed-spectrologically-kameron.ngrok-free.dev"

    # ניצור קישור מלא לפי התיקייה הנוכחית (ללא /static)
    public_url = f"{base_url}/api/output_image"
    print(f"[DEBUG] ✅ Returning direct public image URL: {public_url}")

    return JSONResponse({"output_url": public_url}, status_code=200)


@app.get("/api/get_output_contact")
async def get_output_contact():
    if OUTPUT_CONTACT_VIEW_PATH.exists():
        return JSONResponse({"output_url": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}"})
    return JSONResponse({"error": "No output contact image found"})

@app.get("/api/output_image")
async def output_image():
    """
    מגיש את התמונה ישירות ללקוח (עוקף את /static).
    """
    if not OUTPUT_IMAGE_PATH.exists():
        return JSONResponse({"error": "No image found"}, status_code=404)

    print(f"[DEBUG] ✅ Serving output image directly: {OUTPUT_IMAGE_PATH}")
    return FileResponse(
        path=str(OUTPUT_IMAGE_PATH),
        media_type="image/png",
        filename=OUTPUT_IMAGE_PATH.name
    )
# ✅ Static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
