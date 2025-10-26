import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List
from analyzer_table.launcher_helper.json_models import Ball_Color , Ball

# ===========================
# 🧠 פונקציה אחת - עדכון כדורים
# ===========================
def update_undefined_balls(balls: List[Ball]) -> None:
    """
    מעדכנת את רשימת הכדורים במקום:
    - אם הכדור לבן או שחור → דילוג
    - אם UNDEFINED → הרצת מודלים (mobilenet, efficientnet, vit)
      והכרעה בין solid / striped
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_WEIGHTS = {
        "mobilenet": 1.0,
        "efficientnet": 1.0,
        "vit": 1.0,
    }

    # ===========================
    # עוזרים פנימיים
    # ===========================
    def create_model(name: str):
        if name == "mobilenet":
            model = models.mobilenet_v3_small(pretrained=False)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
        elif name == "efficientnet":
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        elif name == "vit":
            model = models.vit_b_16(pretrained=False)
            model.heads.head = nn.Linear(model.heads.head.in_features, 1)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model.to(DEVICE)

    def load_model(name: str):
        path = os.path.join(MODELS_DIR, f"{name}_best.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing model: {path}")
        model = create_model(name)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model

    def predict_ensemble(image_path: str) -> str:
        """מריץ את התמונה על שלושת המודלים ומחזיר 'solid' או 'striped'"""
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert("RGB")
        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

        model_names = ["mobilenet", "efficientnet", "vit"]
        weighted_sum = 0.0
        total_weight = 0.0
        votes = 0

        for name in model_names:
            weight = MODEL_WEIGHTS[name]
            model = load_model(name)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output)[0].item()
                pred = 1 if prob > 0.5 else 0
                votes += pred
                weighted_sum += prob * weight
                total_weight += weight

        vote_pred = 1 if votes >= 2 else 0
        return "striped" if vote_pred == 1 else "solid"

    # ===========================
    # לולאה ראשית
    # ===========================
    updated = 0
    skipped = 0

    for ball in balls:
        # ⚪ דלג אם הכדור לבן או שחור
        if ball.final_color in (Ball_Color.WHITE, Ball_Color.BLACK):
            skipped += 1
            continue

        # 🔵 דלג אם אין נתיב
        if not ball.single_ball_path or not os.path.exists(ball.single_ball_path):
            print(f"⚠️ Skipping (no path): {ball.single_ball_path}")
            continue

        # 🟣 עדכן רק אם UNDEFINED
        if ball.final_color == Ball_Color.UNDEFINED:
            try:
                prediction = predict_ensemble(ball.single_ball_path)
                ball.final_color = (
                    Ball_Color.STRIPED if prediction == "striped" else Ball_Color.SOLID
                )
                print(f"🎱 {os.path.basename(ball.single_ball_path)} → {ball.final_color}")
                updated += 1
            except Exception as e:
                print(f"❌ Error on {ball.single_ball_path}: {e}")
                ball.final_color = Ball_Color.UNDEFINED

    print(f"\n✅ Done. Updated={updated}, Skipped={skipped}")
