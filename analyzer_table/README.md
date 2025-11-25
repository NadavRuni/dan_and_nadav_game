# Table Border Cropping using Projection Profiles

This document explains the `remove_white_border` function located in `crop_table.py`.

## 1. Problem

When creating a binary mask of a billiard table, unwanted white borders, noise, and other artifacts can appear around the main table area. Simple cropping methods based on contours or flood-fills can be unreliable, either leaving artifacts behind or, more critically, cropping into the actual table surface if there is noise near the edges.

The goal is to precisely crop the binary mask to the boundaries of the black table surface, preserving all internal elements (like white balls and pockets) while discarding all external white artifacts.

## 2. Solution: Projection Profile Algorithm

To solve this robustly, we use a **pixel intensity projection profile** algorithm. Instead of looking at shapes, this method analyzes the image's structure by summing pixel values along rows and columns. This allows it to identify the main "block" of the table, making it highly resistant to noise or small gaps.

The function finds the largest continuous segment of rows and columns that have a high density of "table" pixels (black pixels in the original mask) and crops the image to those boundaries.

## 3. How It Works: Step-by-Step

1.  **Invert Mask:** The input `binary_mask` has a black table (value 0). We invert it so the table is white (value 255). This allows us to sum pixel intensities to find the table area.
2.  **Calculate Projections:**
    *   **Horizontal Projection:** The function sums the pixel values for each row, creating a 1D array where high values indicate a row is mostly part of the table.
    *   **Vertical Projection:** It does the same for each column, creating a second 1D array.
3.  **Find Table Segment:**
    *   A threshold is applied to each projection profile (e.g., 50% of the maximum projection value) to classify which rows/columns belong to the table.
    *   The algorithm then finds the start and end of the longest continuous segment of "table" rows and columns. This gives the `top`, `bottom`, `left`, and `right` boundaries of the core table area.
4.  **Validation:** Before cropping, the function checks if the detected table area is too close to the image edges. If it is, it assumes no border exists and returns the original image to avoid incorrect crops.
5.  **Crop Image:** The final `(top, bottom, left, right)` coordinates are used to crop the original `binary_mask`, ensuring all internal white pockets and balls are preserved.

## 4. Usage

Here is a simple example of how to use the function:

```python
import cv2
from analyzer_table.crop_table import remove_white_border

# Load your binary mask (table=black, background=white)
binary_mask = cv2.imread("path/to/your/mask.png", cv2.IMREAD_GRAYSCALE)

# Process the mask
cropped_mask, crop_coords = remove_white_border(binary_mask)

if crop_coords:
    print("Image was cropped!")
    print(f"New shape: {cropped_mask.shape}")
    print(f"Crop coordinates: {crop_coords}")
    cv2.imwrite("cropped_mask.png", cropped_mask)
else:
    print("No border was detected or cropped.")
```

## 5. Parameters

`remove_white_border(binary_mask, profile_threshold=0.5, edge_margin=5)`

*   `binary_mask` (np.ndarray): The input binary image where the table is black (0).
*   `profile_threshold` (float, optional): The ratio of the maximum projection value used to classify a row or column as part of the table. Defaults to `0.5`.
*   `edge_margin` (int, optional): The margin in pixels from the edge of the image. If the detected table falls within this margin, no cropping is performed. Defaults to `5`.

## 6. Return Value

The function returns a tuple: `(processed_mask, crop_coords)`

*   `processed_mask` (np.ndarray): The processed binary mask. It will be the cropped mask if a border was removed, otherwise it's the original mask.
*   `crop_coords` (Dict | None): A dictionary containing the crop coordinates `{'top', 'bottom', 'left', 'right'}` if a crop was performed. If no crop was done, this value is `None`.