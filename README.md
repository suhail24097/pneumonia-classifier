# Chest X-Ray Pneumonia Detection

Binary classification of chest X-rays (NORMAL vs PNEUMONIA) using a fine-tuned ResNet50 backbone, served via FastAPI, containerized with Docker.

---

## System Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Input X-Ray│───▶│  Preprocessing   │───▶│   ResNet50 Backbone │───▶│ Custom Head         │
│  (JPEG/PNG) │    │  224×224 resize  │    │  (ImageNet weights  │    │ GAP→BN→Dense(256)   │
└─────────────┘    │  preprocess_input│    │   top-30 fine-tuned)│    │ →Dropout(0.4)       │
                   └──────────────────┘    └─────────────────────┘    │ →Dense(64)→Drop(0.2)│
                                                                        │ →Sigmoid            │
                                                                        └──────┬──────────────┘
                                                                               │
                                                          ┌────────────────────▼──────────────┐
                                                          │  Threshold = 0.87                  │
                                                          │  prob ≥ 0.87 → PNEUMONIA           │
                                                          │  prob < 0.87 → NORMAL              │
                                                          └────────────────────────────────────┘
```

---

## Quick Start

### Docker (recommended)

```bash
docker compose up --build
```

API available at `http://localhost:8000`

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

---

## API Endpoints

| Method | Endpoint           | Description                                      |
|--------|--------------------|--------------------------------------------------|
| GET    | /health            | Server status and active threshold               |
| GET    | /model/info        | Model metadata (path, input size, classes)       |
| POST   | /predict           | Single image classification                      |
| POST   | /predict/explain   | Classification + Grad-CAM heatmap (base64 PNG)   |
| POST   | /predict/batch     | Batch classification (up to 20 images)           |

### Example — single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@xray.jpg"
```

Response:
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9512,
  "prob_pneumonia": 0.9512,
  "prob_normal": 0.0488,
  "threshold_used": 0.87,
  "inference_time_ms": 165.3
}
```

### Example — prediction with Grad-CAM

```bash
curl -X POST http://localhost:8000/predict/explain \
  -F "file=@xray.jpg" | python -m json.tool
```

The `grad_cam_image` field contains a base64-encoded PNG showing highlighted lung regions.

### Override threshold at runtime

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -F "file=@xray.jpg"
```

---

## Model Performance

| Metric             | Threshold 0.5 (default) | Threshold 0.87 (tuned) |
|--------------------|------------------------|------------------------|
| Accuracy           | 78%                    | **92%**                |
| AUC-ROC            | 0.9576                 | 0.9576                 |
| NORMAL Recall      | 41%                    | **85%**                |
| NORMAL Precision   | —                      | 96%                    |
| PNEUMONIA Recall   | 99%                    | **96%**                |
| PNEUMONIA Precision| —                      | 90%                    |
| F1 (NORMAL)        | —                      | 0.90                   |
| F1 (PNEUMONIA)     | —                      | 0.93                   |
| Sensitivity        | —                      | **95.9%**              |
| Specificity        | —                      | 85.0%                  |

**Confusion Matrix at threshold 0.87** (test set, 624 images):

|                  | Predicted NORMAL | Predicted PNEUMONIA |
|------------------|-----------------|---------------------|
| Actual NORMAL    | TN = 199        | FP = 35             |
| Actual PNEUMONIA | FN = 16         | TP = 374            |

---

## Dataset Setup

1. Download from Kaggle:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```

2. Run evaluation:
   ```bash
   python scripts/evaluate.py --test-dir data/chest_xray/test/
   ```

---

## Project Structure

```
pneumonia-classifier/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── README.md
├── models/
│   └── pneumonia_resnet50.h5        # Trained model (217 MB)
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                  # FastAPI app (5 endpoints)
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py         # ImageDataGenerator, balanced generator
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py             # PneumoniaPredictor class
│   │   └── grad_cam.py              # Grad-CAM implementation
│   └── models/
│       ├── __init__.py
│       └── architecture.py          # build_resnet50_classifier()
├── scripts/
│   ├── __init__.py
│   ├── evaluate.py                  # Threshold sweep + metrics + plots
│   ├── export_onnx.py               # Export model to ONNX
│   └── generate_report.py           # Generate technical report PDF
├── tests/
│   └── test_api.py                  # pytest API tests
├── notebooks/
│   └── pneumonia_classification.ipynb  # Training notebook
├── docs/
│   └── technical_report.pdf         # Full technical report
└── results/
    ├── metrics.json                 # Known model metrics
    └── test_samples/
        └── README.md                # How to download and test
```

---

## Bonus Features

| Feature         | Details                                                               |
|-----------------|-----------------------------------------------------------------------|
| Docker          | Full containerization with `docker-compose up --build`               |
| Grad-CAM        | Visual explainability — heatmap shows which lung regions matter       |
| ONNX Export     | `python scripts/export_onnx.py` for cross-platform deployment        |
| Batch API       | Up to 20 X-rays in a single POST request                             |
| Threshold API   | Override decision threshold per-request via query parameter          |
| Automated Tests | `pytest tests/` covers all endpoints                                 |
| Tech Report     | Full 7-section PDF in `docs/technical_report.pdf`                    |

---

## Updating the Model or Threshold

**Change threshold (without rebuilding):**
```bash
THRESHOLD=0.75 docker compose up
```

**Use a different model file:**
```bash
MODEL_PATH=models/new_model.h5 docker compose up
```

**Export to ONNX:**
```bash
python scripts/export_onnx.py \
  --model models/pneumonia_resnet50.h5 \
  --output models/pneumonia_resnet50.onnx
```

---

## Architecture Details

- **Backbone**: ResNet50 pretrained on ImageNet (frozen in Phase 1, top-30 layers fine-tuned in Phase 2)
- **Head**: GlobalAveragePooling2D → BatchNorm → Dense(256,relu) → Dropout(0.4) → Dense(64,relu) → Dropout(0.2) → Dense(1,sigmoid)
- **Preprocessing**: ResNet50's `preprocess_input` (subtracts ImageNet means, NOT a /255 rescale)
- **Grad-CAM target layer**: `conv5_block3_out` (final convolutional block, 7×7×2048)
- **Framework**: TensorFlow 2 / Keras
