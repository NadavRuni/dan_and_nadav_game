"""
A utility for creating output paths for debug files.
"""

from pathlib import Path

# The base directory for all debug output.
DEBUG_DIR = Path("output/debug")


def get_output_path(file_name: str, sub_dir: str = "") -> str:
    """
    Constructs a full path for a debug file and ensures the directory exists.

    Args:
        file_name: The name of the file to be saved.
        sub_dir: An optional subdirectory within the main debug directory.

    Returns:
        The full, absolute path to the output file as a string.
    """
    if sub_dir:
        output_dir = DEBUG_DIR / sub_dir
    else:
        output_dir = DEBUG_DIR

    # Ensure the directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use pathlib to construct the final path and return it as a string.
    return str(output_dir / Path(file_name).name)
