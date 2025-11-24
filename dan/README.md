# Orchestration Module (`dan`)

## Purpose

This directory acts as the high-level orchestration layer for the application. It contains the code that connects the web server (`api_server.py`) to the core computer vision logic in the `analyzer_table/` directory. Its primary responsibilities are handling the initial, coarse detection of the pool table and managing the overall analysis pipeline.

## File Map

- **pipe_Line.py**: Contains the main pipeline orchestration function (`start_pipe_line`) that is called by the API server. It delegates the heavy lifting to `analyzer_table` and formats the final JSON output.
- **detect_table_rectangle.py**: Implements the first step of the analysis: finding the rectangular boundary of the pool table in the initial image.
- **build_table_from_image.py**: A supplementary script that appears to be part of the final result-building process.
