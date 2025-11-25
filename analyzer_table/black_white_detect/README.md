# Ball Detection Script (`mark_balls_v4.py`)

This script is designed to detect billiard balls in an image of a pool table. It uses a combination of color-based segmentation and geometric analysis to identify and locate the balls.

## How it Works

The script follows a multi-step pipeline to detect the balls:

1.  **Felt Detection**: It first identifies the green or blue felt of the pool table using color masking in the HSV color space.
2.  **Mask Inversion**: The felt mask is then inverted to isolate the areas where the balls are located.
3.  **Blob Detection**: The script finds connected components (blobs) in the inverted mask, which represent potential balls.
4.  **Hough Circle Refinement**: Each blob is further analyzed using a Hough Circle Transform to confirm if it has a circular shape and to refine its center and radius.
5.  **Filtering**: The detected circles are filtered based on their radius to remove false positives and keep only the detections that match the expected size of a billiard ball.
6.  **Output Generation**: Finally, the script draws the detected balls on a copy of the original image and saves it. It also saves an intermediate mask image.

## Main Function

The primary function in this script is `detect_balls_full_pipeline(input_path: str)`.

### Parameters

*   `input_path` (str): The file path of the input image.

### Returns

*   A list of `Ball` objects, where each object contains the center coordinates and radius of a detected ball.

## Usage

The script can be run from the command line. When executed directly, it will run a default example using a test image (`test_image9.jpeg`) located in the parent directory.

```bash
python analyzer_table/black_white_detect/mark_balls_v4.py
```

## Outputs

When the script is run, it will generate the following output files in the `output/debug/black_white_detect/` directory:

*   `01_felt_mask.jpg`: An image of the mask used to identify the table felt.
*   `output_marked_balls.jpg`: The original image with the detected balls marked with green circles.
