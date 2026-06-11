import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import f1_score

from training.dataset import get_dataloaders
from training.config import MODEL_PATH, MODEL_DIR, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES


def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, classes = get_dataloaders()
    print(f"Classes: {classes}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model (F1: {best_f1:.4f})")

        scheduler.step()

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
