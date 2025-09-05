from fastapi import FastAPI, UploadFile, File
import asyncio
import shutil
from pathlib import Path

from dan.pipe_Line import start_pipe_line
from dan.build_table_from_image import start_build_table_from_img
from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # מותר לכולם
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files from the uploads folder as static
app.mount("/static", StaticFiles(directory="photos/output"), name="static")


# Temporary upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def process_image(image_path: str):
    """Runs the full pipeline and returns a result"""
    pipeline_result = await start_pipe_line(image_path)
    table_result = start_build_table_from_img()
    return {"pipeline": pipeline_result, "table": table_result}


@app.post("/run_pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    # Save uploaded file locally
    file_path = UPLOAD_DIR / file.filename
    print("Saving uploaded file to:", file_path)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call the main processing function
    result = await process_image(str(file_path))
    return result


@app.get("/get_output")
async def get_output():
    if OUTPUT_IMAGE_PATH.exists():
        return {"output_url": f"/static/{OUTPUT_IMAGE_PATH.name}"}
    else:
        return {"error": "No output image found"}


@app.get("/get_output_contact")
async def get_output():
    if OUTPUT_CONTACT_VIEW_PATH.exists():
        return {"output_url": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}"}
    else:
        return {"error": "No output image found"}
