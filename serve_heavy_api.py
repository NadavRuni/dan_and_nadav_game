from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os, asyncio
from pathlib import Path

# ייבוא הפונקציות שלך
from dan.detect_table_rectangle import detect_table_rectangle
from dan.pipe_Line import start_pipe_line
from dan.build_table_from_image import start_build_table_from_img
from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/run-heavy-process")
async def run_heavy_process(file: UploadFile = File(...)):
    try:
        # שמירה זמנית של התמונה
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # קריאה לפונקציות שלך (קיימות)
        detect_table_rectangle(str(file_path))
        pipeline_result = await asyncio.get_event_loop().run_in_executor(
            None, start_pipe_line, str(file_path)
        )
        table_result = start_build_table_from_img()

        return {
            "status": "ok",
            "pipeline": str(pipeline_result),
            "table": str(table_result),
            "output1": f"/static/{OUTPUT_IMAGE_PATH.name}",
            "output2": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
