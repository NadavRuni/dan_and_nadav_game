import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# ==========================================================
# 📍 חישוב נתיב דינמי למודל
# ==========================================================
# המיקום של הקובץ:
# analyzer_table/launcher_helper/score_helper/white_tests/test_is_white_model.py

current_dir = os.path.dirname(os.path.abspath(__file__))  # white_tests
score_helper_dir = os.path.dirname(current_dir)           # score_helper
launcher_helper_dir = os.path.dirname(score_helper_dir)   # launcher_helper
analyzer_table_dir = os.path.dirname(launcher_helper_dir) # analyzer_table

# נתיב המודל: analyzer_table/predict/models/is_white_model.pth
MODEL_PATH = os.path.join(analyzer_table_dir, "predict", "models", "is_white_model.pth")

print(f"[DEBUG] White-Model Path: {MODEL_PATH}")

# ==========================================================
# ⚙️ הגדרות מודל
# ==========================================================
DEVICE = torch.device("cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

_model_cache = None


def load_model():
    """ טוען את המודל לזיהוי הכדור הלבן """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file missing at {MODEL_PATH}")
        return None

    try:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        _model_cache = model
        return model

    except Exception as e:
        print(f"❌ Error loading white model: {e}")
        return None


def get_white_score(ball_crop_bgr):
    """
    מקבל crop של כדור בפורמט OpenCV (BGR)
    ומחזיר ציון סבירות (0.0 עד 1.0) שהכדור הוא הלבן.
    """
    model = load_model()
    if model is None:
        return 0.0

    try:
        img_rgb = cv2.cvtColor(ball_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        img_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)[0].item()

        return prob

    except Exception as e:
        print(f"⚠️ Exception in get_white_score: {e}")
        return 0.0
