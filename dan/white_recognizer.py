from output_utils import get_output_path
import cv2
import numpy as np

input_path = (
    "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-15.jpg"
)
output_path = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/white/table-15.jpg"
image = cv2.imread(input_path)
if image is None:
    raise ValueError("לא ניתן לטעון את התמונה מהנתיב המצוין.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=50,
)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    if len(circles) > 0:
        radii = [r for _, _, r in circles]
        median_r = np.median(radii)
    else:
        median_r = 0
    white_ball = None
    max_intensity = -float("inf")
    height, width = image.shape[:2]
    border_margin = 30
    for x, y, r in circles:
        if abs(r - median_r) > 5:
            continue
        if (
            x < border_margin
            or x > width - border_margin
            or y < border_margin
            or y > height - border_margin
        ):
            continue
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        roi = cv2.bitwise_and(image, image, mask=mask)
        mean_color = cv2.mean(roi, mask=mask)[:3]
        intensity = sum(mean_color)
        if intensity < 20:
            continue
        if intensity > max_intensity:
            max_intensity = intensity
            white_ball = (x, y, r)
    if white_ball is not None:
        x, y, r = white_ball
        cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
output_path = get_output_path("white_recognizer_output.jpg")
cv2.imwrite(output_path, image)
