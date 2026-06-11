import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

from training.dataset import get_dataloaders
from training.config import MODEL_PATH, NUM_CLASSES, CLASS_NAMES


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = get_dataloaders()
    model = load_model(device)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    print("Test Metrics")
    print("-" * 30)
    print(f"  Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"  Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"  Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"  F1-Score:  {f1_score(all_labels, all_preds):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(all_labels, all_probs):.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(colormap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
