import os
from analyzer_table.black_white_detect import pocket_api
from game_class import C_bestShot_use_wall
from PIL import Image
import matplotlib.pyplot as plt

# Define the constant for the image path
IMAGE_PATH = "uploads/0e4940186_everything_white_like.jpeg"


def get_image(image_path: str):
    """
    Processes an image to detect pockets and returns the visualization image.
    """
    print(f"Analyzing pockets from image: {image_path}")
    # The function pocket_detection_api returns original_with_pockets_path, crop_photo_path, rect
    original_with_pockets_path, _, _ = pocket_api.pocket_detection_api(image_path)
    print("Pockets detected.")

    # 2. Call best_shot_use_pocket (using C_bestShot_use_wall as it seems to be the intended module)
    # This section remains as a placeholder for the actual best shot logic.
    print("Calculating best shot...")
    final_path = "mock/path/to/shot"
    plot_image = Image.open(original_with_pockets_path)

    # 3. Print the final path
    print(f"Final Path: {final_path}")

    return plot_image


def run_flow():
    # This function now orchestrates the flow and displays the final image.

    # 1. Get the processed image
    final_image = get_image(IMAGE_PATH)

    # 2. Print the plot that the backend creates
    print("Displaying plot...")
    plt.imshow(final_image)
    plt.title("Best Shot Visualization")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image path not found at '{IMAGE_PATH}'")
        # Create a dummy file for the script to run
        print("Creating a dummy image file.")
        dummy_img = Image.new("RGB", (100, 100), color="blue")
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        dummy_img.save(IMAGE_PATH)
        print(f"Dummy image created at {IMAGE_PATH}")

    run_flow()
