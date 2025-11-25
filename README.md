# Pool Table Vision Analyzer

A computer vision project to analyze images of a pool table, identifying the location and type of all balls and pockets.

## Overview

This application provides a web interface for users to upload an image of a pool table. The backend then processes the image through a sophisticated pipeline to detect the game state, including the position of all balls (solids, stripes, 8-ball, cue ball) and the pockets. The result is displayed as an annotated image.

The project uses a Python FastAPI backend for the core logic and a simple HTML/JavaScript frontend for the user interface.

## Project Structure

- **/api_server.py**: The main FastAPI web server that handles requests and orchestrates the analysis.
- **/frontend/**: Contains the static HTML and JavaScript files for the client-side application.
- **/analyzer_table/**: The core computer vision library. It contains the detailed, multi-stage pipeline for detecting and classifying balls and pockets.
- **/dan/**: A high-level orchestration module that connects the web server to the analysis library and handles initial table detection.
- **/game_class/**: Defines the Python data classes used to represent the game state (e.g., Ball, Pocket, Table).
- **/uploads/**: Default directory for storing user-uploaded images.
- **/output/**: Default directory for storing the final, processed images.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd dan_and_nadav_game
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Node.js dependencies (for the script runner):**
    ```bash
    npm install
    ```

5.  **Start the application:**
    ```bash
    npm run all
    ```
    This command will start the backend server on `http://localhost:8000`. You can now open this URL in your browser.

## Core Usage

1.  Navigate to `http://localhost:8000` in your web browser.
2.  Upload an image of a pool table.
3.  The system will automatically detect the table boundaries. Adjust or confirm the rectangle on the confirmation page.
4.  Once confirmed, the full analysis will run.
5.  The final image, with all balls and pockets identified, will be displayed.
