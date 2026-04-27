import torch
import timm
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    NUM_CLASSES, MODEL_NAME, MODEL_DIR, CLASS_LIST, CLASSES,
    IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD,
)

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ToTensorV2(),
])


def load_model():
    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run train.py first.")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.4f})")
    return model, device


def predict(model, device, image: Image.Image) -> dict:
    img_array = np.array(image.convert("RGB"))
    transformed = transform(image=img_array)["image"]
    batch = transformed.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]
    sorted_indices = torch.argsort(probabilities, descending=True)

    top_idx = sorted_indices[0].item()
    top_class = CLASS_LIST[top_idx]
    top_conf = probabilities[top_idx].item()
    class_info = CLASSES[top_class]

    all_preds = [
        (CLASS_LIST[i.item()], round(probabilities[i.item()].item(), 4))
        for i in sorted_indices
    ]

    return {
        "class_code": top_class,
        "class_name": class_info["name"],
        "confidence": round(top_conf, 4),
        "severity": class_info["severity"],
        "urgency": class_info["urgency"],
        "all_predictions": all_preds,
    }