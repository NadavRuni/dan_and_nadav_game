"""
A simple FastAPI server for running heavy image processing tasks.

Warning:
    This file appears to be a duplicate or an alternative version of
    'api_server.py'. It also contains blocking calls that are not suitable for
    an async server. This file should likely be merged with 'api_server.py'
    or deleted.
"""

import asyncio
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from const_numbers import OUTPUT_IMAGE_PATH, OUTPUT_CONTACT_VIEW_PATH
from dan.build_table_from_image import start_build_table_from_img
from dan.detect_table_rectangle import detect_and_confirm_table_rectangle
from dan.pipe_Line import start_pipe_line

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/run-heavy-process")
async def run_heavy_process(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accepts an image upload and runs the full analysis pipeline.

    Warning:
        This endpoint contains a blocking call to detect and confirm the table
        rectangle, which will freeze the server. It should be run in an
        executor or redesigned.

    Args:
        file: The uploaded image file.

    Returns:
        A JSON response with the status and results of the analysis.
    """
    try:
        # Temporarily save the uploaded image
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the analysis functions
        # This is a blocking call and will freeze the server!
        detect_and_confirm_table_rectangle(str(file_path))

        # The pipeline is run in an executor to avoid blocking
        pipeline_result = await asyncio.get_event_loop().run_in_executor(
            None, start_pipe_line, str(file_path)
        )

        # This is likely a CPU-bound function and should also be in an executor
        table_result = start_build_table_from_img()

        return JSONResponse(
            {
                "status": "ok",
                "pipeline": str(pipeline_result),
                "table": str(table_result),
                "output1": f"/static/{OUTPUT_IMAGE_PATH.name}",
                "output2": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}",
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
