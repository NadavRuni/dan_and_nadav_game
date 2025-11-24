import os
from pathlib import Path

DEBUG_DIR = "output/debug"


def get_output_path(file_path: str, sub_dir: str = "") -> str:
    """
    Creates the output directory and returns the full path for a given file.
    """
    if sub_dir:
        output_dir = os.path.join(DEBUG_DIR, sub_dir)
    else:
        output_dir = DEBUG_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return os.path.join(output_dir, os.path.basename(file_path))
