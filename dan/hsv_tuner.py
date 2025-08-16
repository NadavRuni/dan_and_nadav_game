import cv2
import numpy as np

# --- Configuration ---
IMAGE_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-6.jpg" # The problematic image
# ----------------------

def nothing(x):
    pass

# Create a window for the trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 90, 179, nothing)  # Lower Hue
cv2.createTrackbar("L - S", "Trackbars", 50, 255, nothing)  # Lower Saturation
cv2.createTrackbar("L - V", "Trackbars", 50, 255, nothing)  # Lower Value
cv2.createTrackbar("U - H", "Trackbars", 130, 179, nothing) # Upper Hue
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing) # Upper Saturation
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing) # Upper Value

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("Adjust sliders to isolate the table color. Press 'q' to quit.")

while True:
    # Get current positions of the trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Create the mask with the new values
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Show the original image and the mask
    cv2.imshow("Original Image", img)
    cv2.imshow("Mask", mask)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nFinal HSV values:")
        print(f"LOWER = np.array([{l_h}, {l_s}, {l_v}])")
        print(f"UPPER = np.array([{u_h}, {u_s}, {u_v}])")
        break

cv2.destroyAllWindows()