"""
Classifies undefined balls using an ensemble of deep learning models.

This module uses a set of pre-trained image classification models (MobileNet,
EfficientNet, and ViT) to predict whether a ball is 'solid' or 'striped'.
"""

import os
from typing import List, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from analyzer_table.launcher_helper.json_models import Ball, BallType

# --- Model and Global Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weights for combining the outputs of the different models in the ensemble.
MODEL_WEIGHTS = {"mobilenet": 1.0, "efficientnet": 1.0, "vit": 1.0}
MODELS: Dict[str, nn.Module] = {}


def _create_model(name: str) -> nn.Module:
    """Creates a model architecture and modifies its final classification layer."""
    if name == "mobilenet":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    elif name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif name == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model.to(DEVICE)


def _load_models() -> Dict[str, nn.Module]:
    """
    Loads all pre-trained model weights from disk into memory.

    Note:
        This is an expensive operation and should ideally be done only once
        at application startup.
    """
    if MODELS:
        return MODELS

    print("🔄 Loading prediction models...")
    loaded_models = {}
    for name in MODEL_WEIGHTS:
        path = os.path.join(BASE_DIR, f"{name}_best.pth")
        model = _create_model(name)
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            loaded_models[name] = model
        except FileNotFoundError:
            print(f"❌ Model weights not found at {path}. Skipping this model.")
    print(f"✅ {len(loaded_models)} models loaded into memory.")
    MODELS.update(loaded_models)
    return loaded_models


def _predict_ensemble(image_path: str, transform: transforms.Compose) -> str:
    """
    Predicts the ball type using a weighted ensemble of loaded models.

    Args:
        image_path: Path to the cropped ball image.
        transform: The torchvision transform to apply to the image.

    Returns:
        The predicted ball type ('solid' or 'striped').
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    weighted_sum = 0.0
    total_weight = 0.0
    votes = 0

    loaded_models = _load_models()

    for name, model in loaded_models.items():
        weight = MODEL_WEIGHTS.get(name, 0)
        with torch.no_grad():
            output = model(img_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            votes += prediction
            weighted_sum += probability * weight
            total_weight += weight

    # Final prediction is based on a majority vote.
    vote_prediction = 1 if votes >= 2 else 0
    return "striped" if vote_prediction == 1 else "solid"


def update_undefined_balls(balls: List[Ball]) -> None:
    """
    Iterates through a list of balls and classifies any 'undefined' ones.

    This function uses a pre-trained model ensemble to predict whether a ball
    is solid or striped. It modifies the 'final_color' attribute of the Ball
    objects in the input list directly.

    Args:
        balls: A list of Ball objects to be classified.
    """
    # Define the image transformation required by the models
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    updated_count = 0
    skipped_count = 0

    for ball in balls:
        if ball.final_color not in (BallType.UNDEFINED, None):
            skipped_count += 1
            continue

        if not ball.single_ball_path or not os.path.exists(ball.single_ball_path):
            print(f"⚠️ Skipping ball (no image path): {ball.center}")
            continue

        try:
            prediction = _predict_ensemble(ball.single_ball_path, val_transform)
            ball.final_color = (
                BallType.STRIPED if prediction == "striped" else BallType.SOLID
            )
            print(f"🎱 Prediction for ball at {ball.center} → {ball.final_color}")
            updated_count += 1
        except Exception as e:
            print(f"❌ Error predicting for ball at {ball.center}: {e}")
            ball.final_color = BallType.UNDEFINED

    print(
        f"\n✅ Ball classification finished. Updated={updated_count}, Skipped={skipped_count}"
    )
