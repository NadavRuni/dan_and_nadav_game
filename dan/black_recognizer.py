import cv2
import numpy as np

# הכנס כאן את הנתיבים הרצויים
input_path = '/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-14.jpg'  # שנה לנתיב התמונה הקלט שלך
output_path = '/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/output/black/table-14.jpg'  # שנה לנתיב התמונה הפלט הרצויה

# טען את התמונה
image = cv2.imread(input_path)
if image is None:
    raise ValueError("לא ניתן לטעון את התמונה מהנתיב המצוין.")

# המר לאפור לצורך זיהוי עיגולים
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# השתמש ב-HoughCircles לזיהוי עיגולים (כדורים)
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,  # מרחק מינימלי בין מרכזי עיגולים
    param1=50,
    param2=30,
    minRadius=10,  # רדיוס מינימלי לכדור
    maxRadius=50   # רדיוס מקסימלי לכדור
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    
    # חשב את הרדיוס החציוני כדי לסנן רדיוסים לא תואמים (למשל כיסים)
    if len(circles) > 0:
        radii = [r for _, _, r in circles]
        median_r = np.median(radii)
    else:
        median_r = 0
    
    black_ball = None
    min_intensity = float('inf')
    
    height, width = image.shape[:2]
    border_margin = 30  # מרווח מהגבולות כדי לסנן כיסים (התאם לפי הצורך)
    
    for (x, y, r) in circles:
        # סנן לפי רדיוס: רק אם קרוב לרדיוס החציוני
        if abs(r - median_r) > 5:  # סובלנות, התאם אם צריך
            continue
        
        # סנן אם קרוב מדי לגבולות התמונה (כיסים נמצאים בקצוות)
        if x < border_margin or x > width - border_margin or y < border_margin or y > height - border_margin:
            continue
        
        # חלץ את האזור של העיגול
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        roi = cv2.bitwise_and(image, image, mask=mask)
        
        # חשב את הצבע הממוצע באזור העיגול (בערוצי BGR)
        mean_color = cv2.mean(roi, mask=mask)[:3]  # B, G, R
        intensity = sum(mean_color)
        
        # סנן אם כהה מדי (כנראה כיס, לא כדור שחור)
        if intensity < 20:  # סף כהות, התאם אם צריך על סמך התמונה
            continue
        
        # בדוק אם זה הכדור הכהה ביותר
        if intensity < min_intensity:
            min_intensity = intensity
            black_ball = (x, y, r)
    
    if black_ball is not None:
        x, y, r = black_ball
        # סמן ריבוע סביב הכדור השחור
        cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)

# שמור את התמונה המעובדת
cv2.imwrite(output_path, image)