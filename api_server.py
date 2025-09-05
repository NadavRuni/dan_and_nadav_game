import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dan.pipe_Line import start_pipe_line
from dan.build_table_from_image import start_build_table_from_img
from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("photos/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path("frontend/src")  # where index.html lives

# Serve static files
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


async def process_image(image_path: str):
    pipeline_result = await start_pipe_line(image_path)
    table_result = start_build_table_from_img()
    return {"pipeline": pipeline_result, "table": table_result}


# Root route -> serve index.html
@app.get("/")
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "index.html not found in frontend/src"}


@app.post("/run_pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    print("Saving uploaded file to:", file_path)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await process_image(str(file_path))
    return result


@app.get("/get_output")
async def get_output():
    if OUTPUT_IMAGE_PATH.exists():
        return {"output_url": f"/static/{OUTPUT_IMAGE_PATH.name}"}
    return {"error": "No output image found"}


@app.get("/get_output_contact")
async def get_output_contact():
    if OUTPUT_CONTACT_VIEW_PATH.exists():
        return {"output_url": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}"}
    return {"error": "No output contact image found"}
