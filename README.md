# Explainable Chest X-ray Classification Using CNNs and Grad-CAM

An educational deep learning application that classifies chest X-ray images as **Normal** or **Pneumonia** using transfer learning with ResNet18, with Grad-CAM heatmaps showing which regions of the image influenced the prediction.

> **Disclaimer:** This application is for educational and research purposes only. It is NOT intended for clinical diagnosis, treatment decisions, or patient triage.

---

## Features

- CNN-based binary image classification (Normal vs Pneumonia)
- Transfer learning with pretrained ResNet18
- Grad-CAM explainability heatmaps
- Streamlit web interface
- Docker deployment

## Tech Stack

| Component | Tool |
|-----------|------|
| Model training | PyTorch + Torchvision |
| CNN model | ResNet18 (transfer learning) |
| Explainability | Grad-CAM (`pytorch-grad-cam`) |
| Image processing | Pillow + OpenCV |
| Metrics | scikit-learn |
| Web app | Streamlit |
| Deployment | Docker |

---

## Project Structure

```
medical-image-classification-app/
├── app/
│   ├── streamlit_app.py   # Streamlit UI
│   ├── inference.py       # Single-image prediction + Grad-CAM
│   ├── gradcam_utils.py   # Grad-CAM wrapper
│   └── transforms.py      # Inference transforms
├── training/
│   ├── train.py           # Training loop
│   ├── dataset.py         # DataLoaders and augmentations
│   ├── evaluate.py        # Test metrics and confusion matrix
│   └── config.py          # Hyperparameters and paths
├── models/                # Saved model weights (not committed)
├── data/                  # Dataset (not committed — download separately)
│   ├── train/
│   ├── val/
│   └── test/
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## Dataset

Use the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

Organize images into:
```
data/
├── train/
│   ├── normal/
│   └── pneumonia/
├── val/
│   ├── normal/
│   └── pneumonia/
└── test/
    ├── normal/
    └── pneumonia/
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m training.train
```

The best model (by validation F1) is saved to `models/pneumonia_model.pth`.

### 3. Evaluate on the test set

```bash
python -m training.evaluate
```

Prints accuracy, recall, precision, F1, and ROC-AUC. Saves `confusion_matrix.png`.

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser, upload a chest X-ray, and see the prediction + Grad-CAM heatmap.

---

## Docker

```bash
# Build
docker build -t chest-xray-classifier .

# Run
docker run -p 8501:8501 chest-xray-classifier
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Recall | Pneumonia cases correctly caught (minimize false negatives) |
| Precision | Positive predictions that are actually pneumonia |
| F1-Score | Balanced precision/recall |
| ROC-AUC | Probability-based performance across thresholds |

---

## Build Roadmap

- [x] Phase 1 — Prepare dataset (train/val/test split)
- [x] Phase 2 — Train baseline ResNet18 model
- [x] Phase 3 — Evaluate with full metrics
- [x] Phase 4 — Single-image inference function
- [x] Phase 5 — Grad-CAM heatmap overlay
- [x] Phase 6 — Streamlit UI
- [x] Phase 7 — Docker container
- [x] Phase 8 — README
