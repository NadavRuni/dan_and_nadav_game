# בסיס קל משקל של Python
FROM python:3.11-slim

# תיקיית העבודה
WORKDIR /app

# העתקת כל הקבצים שלך לתוך הקונטיינר
COPY . .

# התקנת התלויות
RUN pip install --no-cache-dir -r requirements.txt

# יצירת תיקיות נחוצות מראש (למניעת שגיאות)
RUN mkdir -p uploads outputs photos/output

# הגדרת הפקודה שמריצה את השרת
CMD ["uvicorn", "serve_heavy_api:app", "--host", "0.0.0.0", "--port", "10000"]

