"""
Main API server for the Pool Game Analysis application.

This module provides the FastAPI backend that serves the frontend, handles file
uploads, and orchestrates the image analysis pipeline to suggest the best shots
in a game of pool.

Endpoints:
  - GET  /frontend_nav.json: Retrieves navigation state for the frontend.
  - POST /api/run_pipeline: Starts the table detection pipeline for a given image.
  - POST /api/confirm_rectangle: Confirms the table boundaries and runs the full
    analysis.
  - POST /api/best_shot_use_pocket: Runs analysis using user-defined pockets.
  - POST /api/get_pocket: Detects pockets in the uploaded image.
  - GET  /api/get_image: Retrieves the final processed image with shot suggestions.
  - GET  /api/get_output: Provides a public URL to the output image.
  - GET  /api/get_output_contact: Retrieves the contact points visualization.
  - GET  /api/output_image: Serves the output image file directly.
  - Static mounts for serving frontend, uploads, and output files.
"""

import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer_table.black_white_detect.pocket_api import pocket_detection_api
from analyzer_table.launcher_helper.data_to_rectangle import create_rectangle_from_data
from const_numbers import (
    UPLOAD_DIR,
    OUTPUT_DIR,
    RECTANGLE_JSON_PATH,
    OUTPUT_IMAGE_PATH,
    OUTPUT_CONTACT_VIEW_PATH,
    FRONTEND_DIR,
    set_ball_type,
    get_rectangle_croped,
    set_use_predicted_pockets,
    set_pocket_path,
    get_pocket_path,
    set_rectangle_croped,
)
from crop_table import crop_image_by_rectangle
from dan.build_table_from_image import start_build_table_from_img
from dan.detect_table_rectangle import (
    detect_and_confirm_table_rectangle,
    update_table_size_from_rectangle,
)
from dan.pipe_Line import start_pipe_line
from fetch_pockets import fetch_pockets_from_data

# --- Application Setup ---

# Initialize the FastAPI application
app = FastAPI(default_response_class=Response)

# Enable CORS for all origins to allow frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define base directory and ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Thread pool for running synchronous, CPU-bound tasks in the background
background_executor = ThreadPoolExecutor()


# --- Cleanup ---

# Clean up any old navigation files from previous runs on startup
nav_file = BASE_DIR / "frontend_nav.json"
if nav_file.exists():
    print(f"🧹 Deleting leftover frontend_nav.json on startup: {nav_file}")
    nav_file.unlink()


# --- API Endpoints ---


@app.get("/frontend_nav.json")
async def get_navigation_state() -> Response:
    """
    Serves the frontend navigation state file if it exists.

    This endpoint is polled by the frontend to get updates on the image
    processing status, allowing it to navigate to the correct view
    (e.g., from upload to confirmation).

    Returns:
        FileResponse: The navigation JSON file if it exists.
        Response: A 204 No Content response if the file does not exist.
    """
    nav_path = BASE_DIR / "frontend_nav.json"
    print(f"[DEBUG] Trying to serve navigation state file from: {nav_path}")

    if nav_path.exists():
        print("✅ Found frontend_nav.json — serving now.")
        return FileResponse(nav_path)

    # Return 204 No Content, which is a standard and efficient way to signal
    # that the resource is not yet available without causing frontend errors.
    return Response(status_code=204)


@app.post("/api/run_pipeline")
async def run_pipeline(
    request: Request,
    file: UploadFile = None,
) -> JSONResponse:
    """
    Accepts an image via URL or file upload and starts the table detection
    pipeline.

    This is the first step in the analysis, identifying the pool table's
    rectangle from the full image. The actual detection runs in a background
    thread to avoid blocking the server.

    Args:
        request: The incoming FastAPI request object.
        file: An optional uploaded file containing the image.

    Returns:
        JSONResponse: A status message indicating that processing has started,
                      along with the path to the saved file.
    """
    try:
        image_path = None
        content_type = request.headers.get("content-type", "")

        if content_type.startswith("application/json"):
            data = await request.json()
            image_url = data.get("image_url")
            if not image_url:
                return JSONResponse(
                    {"error": "Request body is missing 'image_url'"}, status_code=400
                )

            filename = Path(image_url).name
            image_path = UPLOAD_DIR / filename
            print(f"[DEBUG] Downloading image from URL: {image_url}")

            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            with open(image_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

        elif file:
            image_path = UPLOAD_DIR / file.filename
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            return JSONResponse(
                {"error": "No file or image_url provided in the request"},
                status_code=400,
            )

        print(f"[DEBUG] Starting table detection pipeline for {image_path}")
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            background_executor, detect_and_confirm_table_rectangle, str(image_path)
        )

        return JSONResponse({"status": "processing", "file_path": str(image_path)})

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in run_pipeline: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/confirm_rectangle")
async def confirm_rectangle(data: Dict[str, Any]) -> JSONResponse:
    """
    Confirms the detected table rectangle, crops the image, and runs the full
    shot analysis pipeline.

    Args:
        data: A dictionary containing the rectangle coordinates, the image path,
              and the player's ball type ('SOLID' or 'STRIPED').

    Returns:
        JSONResponse: A dictionary containing the status and results from the
                      analysis pipeline and table build process.
    """
    try:
        set_use_predicted_pockets(False)
        print("[DEBUG] Received rectangle confirmation data:", data)
        rectangle = create_rectangle_from_data(data)

        image_path = data.get("image_path")
        ball_type = data.get("ball_type")
        set_ball_type(ball_type)

        if not image_path:
            return JSONResponse(
                {"error": "Missing 'image_path' in request"}, status_code=400
            )

        cropped_path, scaled_rectangle = crop_image_by_rectangle(
            rectangle, image_path, UPLOAD_DIR, data
        )
        update_table_size_from_rectangle(scaled_rectangle)

        # Save the scaled rectangle to a JSON file for debugging and reuse.
        rect_path = BASE_DIR / RECTANGLE_JSON_PATH
        rect_path.write_text(
            json.dumps(asdict(scaled_rectangle), indent=2), encoding="utf-8"
        )
        print(f"[DEBUG] ✅ Saved scaled rectangle to: {rect_path}")

        if not cropped_path:
            return JSONResponse({"error": "Failed to crop image"}, status_code=500)

        print("[DEBUG] Starting full analysis pipeline for image:", cropped_path)
        # TODO: These are CPU-bound and should run in an executor
        pipeline_result = await start_pipe_line(str(cropped_path))
        table_result = start_build_table_from_img()

        return JSONResponse(
            {"status": "ok", "pipeline": pipeline_result, "table": table_result}
        )

    except Exception as e:
        import traceback

        print("❌ Exception in confirm_rectangle:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/best_shot_use_pocket")
async def best_shot_use_pocket(request: Request) -> JSONResponse:
    """
    Runs the analysis pipeline using user-submitted pocket locations instead of
    detecting them automatically.

    Args:
        request: The incoming FastAPI request object containing ball type and
                 pocket data.

    Returns:
        JSONResponse: A dictionary containing the status and analysis results.
    """
    try:
        data = await request.json()
        print("[DEBUG] Received best_shot_use_pocket data:", data)
        ball_type = data.get("ball_type")
        set_ball_type(ball_type)
        fetch_pockets_from_data(data)
        set_use_predicted_pockets(True)

        rect_path = BASE_DIR / RECTANGLE_JSON_PATH
        rect_path.write_text(
            json.dumps(asdict(get_rectangle_croped()), indent=2), encoding="utf-8"
        )

        # TODO: These are CPU-bound and should run in an executor
        pipeline_result = await start_pipe_line(str(get_pocket_path()))
        table_result = start_build_table_from_img()

        print("[DEBUG] best_shot_use_pocket completed successfully.")
        return JSONResponse(
            {"status": "ok", "pipeline": pipeline_result, "table": table_result}
        )

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch image from URL: {e}")
        return JSONResponse({"error": f"Failed to fetch image: {e}"}, status_code=500)
    except Exception as e:
        import traceback

        print("❌ Exception in best_shot_use_pocket:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/get_pocket")
async def get_pocket(request: Request) -> JSONResponse:
    """
    Downloads an image from a URL and runs the pocket detection API on it.

    Args:
        request: The incoming FastAPI request, containing the image URL.

    Returns:
        JSONResponse: On success, returns a URL to a visualization of the
                      detected pockets. On failure, returns an error.
    """
    print("[DEBUG] /api/get_pocket called")
    try:
        data = await request.json()
        image_url = data.get("image_url")

        if not image_url:
            return JSONResponse(
                {"error": "Missing 'image_url' in request"}, status_code=400
            )

        file_name = Path(image_url).name
        image_path = UPLOAD_DIR / file_name
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        with open(image_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"[DEBUG] Image downloaded and saved to: {image_path}")

        (
            img_path_with_circles,
            img_path_for_work,
            rectangle,
        ) = pocket_detection_api(str(image_path))

        set_pocket_path(img_path_for_work)
        set_rectangle_croped(rectangle)

        relative_path = os.path.relpath(img_path_with_circles, start=OUTPUT_DIR)
        file_url = f"/static/{relative_path.replace(os.sep, '/')}"

        print(f"[DEBUG] Returning file URL for pocket visualization: {file_url}")
        return JSONResponse({"file_url": file_url})

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch image from URL: {e}")
        return JSONResponse({"error": f"Failed to fetch image: {e}"}, status_code=500)
    except Exception as e:
        import traceback

        print("❌ Exception in get_pocket:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/get_image")
async def get_image() -> JSONResponse:
    """
    Provides a URL to the final, processed image showing the best shot.

    Returns:
        JSONResponse: A JSON object with a URL to the static image file.
    """
    try:
        file_path = OUTPUT_DIR / "img.png"
        if not file_path.exists():
            print(f"❌ Processed image not found at: {file_path}")
            return JSONResponse({"error": "Processed image not found"}, status_code=404)

        print(f"[DEBUG] Found final image: {file_path}")
        relative_path = os.path.relpath(file_path, start=OUTPUT_DIR)
        file_url = f"/static/{relative_path.replace(os.sep, '/')}"

        print(f"[DEBUG] Returning file URL: {file_url}")
        return JSONResponse({"file_url": file_url})

    except Exception as e:
        import traceback

        print("❌ Exception in get_image:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/get_output")
async def get_output(request: Request) -> JSONResponse:
    """
    Provides a full, publicly accessible URL to the final output image.

    This is useful for environments like ngrok where relative paths may not work.

    Args:
        request: The incoming FastAPI request object.

    Returns:
        JSONResponse: A JSON object with the absolute URL to the output image.
    """
    if not OUTPUT_IMAGE_PATH.exists():
        print("[DEBUG] ❌ No output image found to serve.")
        return JSONResponse({"error": "No output image found"}, status_code=404)

    base_url = str(request.base_url).rstrip("/")
    public_url = f"{base_url}/api/output_image"
    print(f"[DEBUG] ✅ Returning direct public image URL: {public_url}")

    return JSONResponse({"output_url": public_url}, status_code=200)


@app.get("/api/get_output_contact")
async def get_output_contact() -> JSONResponse:
    """
    Provides a URL to the contact points visualization image.

    Returns:
        JSONResponse: A JSON object with a URL to the static image file, or
                      an error if the file is not found.
    """
    if OUTPUT_CONTACT_VIEW_PATH.exists():
        return JSONResponse({"output_url": f"/static/{OUTPUT_CONTACT_VIEW_PATH.name}"})
    return JSONResponse({"error": "No output contact image found"}, status_code=404)


@app.get("/api/output_image")
async def serve_output_image() -> Response:
    """
    Serves the final output image file directly.

    This endpoint allows clients to fetch the image without relying on static
    file mounts, which is useful for the public URL generated by /api/get_output.

    Returns:
        FileResponse: The image file if found.
        JSONResponse: An error if the image is not found.
    """
    if not OUTPUT_IMAGE_PATH.exists():
        return JSONResponse({"error": "No image found"}, status_code=404)

    print(f"[DEBUG] ✅ Serving output image directly: {OUTPUT_IMAGE_PATH}")
    return FileResponse(
        path=str(OUTPUT_IMAGE_PATH),
        media_type="image/png",
        filename=OUTPUT_IMAGE_PATH.name,
    )


# --- Static File Mounting ---

# Mount directories to serve static files for the frontend, uploads, and results.
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
