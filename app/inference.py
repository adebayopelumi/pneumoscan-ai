import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from PIL import Image

from app.transforms import inference_transform
from app.gradcam_utils import generate_gradcam

MODEL_PATH = "models/pneumonia_model.pth"
CLASS_NAMES = ["Normal", "Pneumonia"]
NUM_CLASSES = 2

_model = None
_device = None


def _load_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        model.to(_device)
        model.eval()
        _model = model
    return _model, _device


def predict_with_gradcam(pil_image: Image.Image):
    model, device = _load_model()

    img_tensor = inference_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(dim=1)

    predicted_class = predicted.item()
    confidence_score = confidence.item()
    label = CLASS_NAMES[predicted_class]

    rgb_float = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = generate_gradcam(model, img_tensor, predicted_class, rgb_float)

    return label, confidence_score, heatmap
