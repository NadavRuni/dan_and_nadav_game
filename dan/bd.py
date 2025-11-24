from output_utils import get_output_path
import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

YOLO_MODEL_PATH = "yolov8n.pt"
BALL_CLASSIFIER_MODEL_PATH = "ball_classifier_model.h5"
TRAIN_CLASSES_DIR = "/Users/danbenzvi/Desktop/archive-2/train"
TEST_IMAGE_PATH = (
    "/Users/danbenzvi/Desktop/dan_nadav_game/dan_and_nadav_game/dan/images/table-15.jpg"
)
yolo = YOLO(YOLO_MODEL_PATH)
classifier = load_model(BALL_CLASSIFIER_MODEL_PATH)
class_names = sorted(os.listdir(TRAIN_CLASSES_DIR))


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


image = cv2.imread(TEST_IMAGE_PATH)
results = yolo(image)[0]
ball_boxes = []
for box in results.boxes:
    cls = int(box.cls[0])
    if cls == 32:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ball_boxes.append((x1, y1, x2, y2))
types = classify_balls(image, ball_boxes, classifier, class_names)
for (x1, y1, x2, y2), label in zip(ball_boxes, types):
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
output_path = get_output_path("annotated_balls.jpg")
cv2.imwrite(output_path, image)
print(f"✅ Image saved to {output_path}")
