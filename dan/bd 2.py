# billiard_ball_classifier.py
# Detect balls using YOLOv8 and classify them using a custom MobileNetV2 model

import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Paths ===
YOLO_MODEL_PATH = "yolov8n.pt"  # or custom weights if you trained on billiard balls
BALL_CLASSIFIER_MODEL_PATH = "ball_classifier_model.h5"
TRAIN_CLASSES_DIR = "/Users/danbenzvi/Desktop/archive-2/train"  # used to get class names
TEST_IMAGE_PATH = "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-15.jpg"  # image with billiard balls to test

# === Load Models ===
yolo = YOLO(YOLO_MODEL_PATH)
classifier = load_model(BALL_CLASSIFIER_MODEL_PATH)
class_names = sorted(os.listdir(TRAIN_CLASSES_DIR))

# === Helper Function: Classify Balls ===
def classify_balls(image, boxes, model, class_names):
    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            predictions.append("unknown")
            continue
        crop = cv2.resize(crop, (224, 224))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        array = img_to_array(crop)
        array = preprocess_input(array)
        array = np.expand_dims(array, axis=0)
        pred = model.predict(array, verbose=0)
        predictions.append(class_names[np.argmax(pred)])
    return predictions

# === Load and Analyze Image ===
image = cv2.imread(TEST_IMAGE_PATH)
results = yolo(image)[0]

# === Extract Ball Detections ===
ball_boxes = []
for box in results.boxes:
    cls = int(box.cls[0])
    if cls == 32:  # class 32 in COCO = sports ball (adjust if needed)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ball_boxes.append((x1, y1, x2, y2))

# === Classify Balls ===
types = classify_balls(image, ball_boxes, classifier, class_names)

# === Annotate Image ===
for (x1, y1, x2, y2), label in zip(ball_boxes, types):
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# === Save Result ===
cv2.imwrite("output/annotated_balls.jpg", image)
print("✅ Image saved to output/annotated_balls.jpg")
