"""
A script to run the full game analysis pipeline from start to finish.
"""

import asyncio
import sys

from dan.build_table_from_image import start_build_table_from_img
from dan.pipe_Line import start_pipe_line


async def start_game(image_path: str) -> None:
    """
    Asynchronously runs the main analysis pipeline and then builds the table.

    Args:
        image_path: The path to the image to be analyzed.
    """
    await start_pipe_line(image_path)
    start_build_table_from_img()


if __name__ == "__main__":
    # Default image path can be overridden by a command-line argument
    default_image = "/Users/nadavhershkovitz/Desktop/Runi/idea2app/dan_and_nadav_game/photos/img_start.jpeg"

    if len(sys.argv) > 1:
        image_to_process = sys.argv[1]
    else:
        image_to_process = default_image
        print(f"Usage: python full_flow.py [optional_image_path]")
        print(f"Using default image: {default_image}")

    asyncio.run(start_game(image_to_process))
