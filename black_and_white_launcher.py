#!/usr/bin/env python3
"""
black_and_white_launcher.py
מריץ שילוב של שני אלגוריתמים:
1. Color Based (V4) - מזהה מצוין צהוב וכדורים רגילים.
2. Geometric Based (Hough) - מזהה מצוין כדורים מוסווים (ירוק כהה).
"""

import cv2
import sys
import math
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "black_white_detect"))

# ייבוא האלגוריתם המקורי (זיהוי צבע)
from analyzer_table.black_white_detect.mark_balls_v4 import detect_balls_full_pipeline as detect_by_color
# ייבוא האלגוריתם החדש (זיהוי גיאומטרי לירוק)
from analyzer_table.black_white_detect.hough_detector import detect_balls_full_pipeline as detect_by_shape

from analyzer_table.launcher_helper.json_models import Ball

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_ball_detection(image_path: str):
    """
    מקבל תמונה, מריץ את שני הזיהויים ומאחד תוצאות.
    """
    print(f"--- Starting Dual Detection Mode ---")
    
    # 1. הרצת הזיהוי הקלאסי (V4)
    print("running V4 (Color)...")
    balls_v4 = detect_by_color(image_path)
    
    # 2. הרצת הזיהוי הגיאומטרי (Hough)
    print("running Hough (Geometric)...")
    balls_geo = detect_by_shape(image_path)

    # 3. איחוד התוצאות
    # מתחילים עם הרשימה של V4 כי היא הכי אמינה לרוב הכדורים
    final_balls = list(balls_v4)
    
    merged_count = 0
    
    # עוברים על הכדורים שמצאנו בשיטה הגיאומטרית
    for g_ball in balls_geo:
        is_new_ball = True
        
        # בודקים אם הכדור הזה כבר קיים ברשימה של V4
        for existing_ball in final_balls:
            dist = calculate_distance(g_ball.center, existing_ball.center)
            
            # אם המרחק קטן מ-20 פיקסלים, זה אותו כדור שכבר מצאנו -> מתעלמים
            if dist < 20:
                is_new_ball = False
                break
        
        # אם הכדור רחוק מכל מה שמצאנו קודם -> זה כנראה הירוק החסר! מוסיפים אותו.
        if is_new_ball:
            print(f"➕ Adding unique ball found by Geometry (Likely Green/Camouflaged): {g_ball.center}")
            final_balls.append(g_ball)
            merged_count += 1

    # הדפסה מסכמת
    print(f"✅ Total Detected: {len(final_balls)} (V4: {len(balls_v4)} + Unique Geometric: {merged_count})")
    
    return final_balls


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python black_and_white_launcher.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    balls = run_ball_detection(image_path)