from dan.pipe_Line import start_pipe_line
import asyncio
from dan.build_table_from_image import start_build_table_from_img


async def start_game():
    await start_pipe_line(
        "/Users/nadavhershkovitz/Desktop/Runi/idea2app/dan_and_nadav_game/photos/img_start.jpeg"
    )
    start_build_table_from_img()


if __name__ == "__main__":
    asyncio.run(start_game())
