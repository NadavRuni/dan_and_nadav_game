# Core Analysis Library (`analyzer_table`)

## Purpose

This directory is the heart of the project's computer vision capabilities. It functions as a self-contained library dedicated to the detailed analysis of a cropped pool table image. It is responsible for finding and classifying every object on the table.

## File Map

- **balls_from_image.py**: Contains the `full_analyzer_pipeline` function, which is the master orchestrator for the entire analysis. It combines results from multiple detection and classification methods.
- **detect_ball/**: A sub-module containing various ball detection algorithms and utilities (e.g., `analyzer_runner.py`, `merge_utils.py`).
- **launcher_helper/**: Contains helper scripts and data models, including code for classifying balls by color and type.
- **pocket/**: Contains the logic for detecting the table's pockets (`pocket_detect.py`).
- **predict/**: Contains a machine learning model (`predict.py`) used to classify balls that cannot be identified through simple heuristics.
- **table/**: Contains utilities for handling the table geometry.
