# Chest X-Ray Pneumonia Detection

Binary classification of chest X-rays (NORMAL vs PNEUMONIA) using a fine-tuned ResNet50 backbone, served via FastAPI, containerized with Docker.

**Results:** Accuracy 92% · AUC-ROC 0.9576 · PNEUMONIA Recall 96% · NORMAL Recall 85% · Threshold 0.87

---

## Complete Setup & Run Guide (Start to End)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/suhail24097/pneumonia-classifier.git
cd pneumonia-classifier
```

### Step 2 — Download the Trained Model

The model file (`pneumonia_resnet50.h5`, 217 MB) is not included in the repo.  
Download it and place it in the `models/` folder:

```bash
mkdir -p models/
# Download from Google Drive (link provided separately)
# Place as: models/pneumonia_resnet50.h5
```

### Step 3 — Option A: Run with Docker (Recommended)

```bash
# Build and start
sudo docker build -t pneumonia-classifier .
sudo docker run -p 8000:8000 -v $(pwd)/models:/app/models pneumonia-classifier

# OR with docker compose
sudo docker compose up --build
```

API will be available at: `http://localhost:8000`

### Step 3 — Option B: Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
MODEL_PATH=models/pneumonia_resnet50.h5 uvicorn src.api.main:app --reload --port 8000
```

### Step 4 — Open Web UI

Open your browser and go to:
```
http://localhost:8000
```

- **Single Image tab** — Upload one X-ray, get prediction + Grad-CAM heatmap
- **Batch tab** — Upload multiple X-rays, get summary + per-image results

### Step 5 — Test via API

```bash
# Health check
curl http://localhost:8000/health

# Predict on a sample image
curl -X POST http://localhost:8000/predict \
  -F "file=@results/test_samples/PNEUMONIA/person101_bacteria_485.jpeg"

# Predict with Grad-CAM explanation
curl -X POST http://localhost:8000/predict/explain \
  -F "file=@results/test_samples/NORMAL/IM-0001-0001.jpeg"
```

Expected output for `/predict`:
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9998,
  "prob_pneumonia": 0.9998,
  "prob_normal": 0.0002,
  "threshold_used": 0.87,
  "inference_time_ms": 158.0
}
```

### Step 6 — Run Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

### Step 7 — Export to ONNX (Optional)

```bash
python scripts/export_onnx.py \
  --model models/pneumonia_resnet50.h5 \
  --output models/pneumonia_resnet50.onnx
```

### Step 8 — Re-train the Model (Optional)

Open the training notebook in Google Colab:
```
notebooks/pneumonia_classification.ipynb    # v1 — ResNet50 (trained, 92% accuracy)
notebooks/pneumonia_classification_v2.ipynb # v2 — DenseNet121 (improved architecture)
```

1. Upload notebook to [Google Colab](https://colab.research.google.com)
2. Set Runtime → T4 GPU
3. Upload `archive.zip` (Kaggle chest X-ray dataset) to Google Drive at:  
   `MyDrive/Colab Notebooks/dataset/archive.zip`
4. Run All cells

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

## Web UI

Open `http://localhost:8000` for the visual interface.

**Single Image tab**: Upload one X-ray → get prediction badge (NORMAL/PNEUMONIA), confidence score, probability bars, and Grad-CAM heatmap side-by-side with original.

**Batch tab**: Upload multiple X-rays at once → summary (total/PNEUMONIA count/NORMAL count/avg confidence) + expandable result cards with Grad-CAM for each image.

---

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

| Metric | Default (0.5) | Optimal (0.87) |
|--------|--------------|----------------|
| Accuracy | 78% | **92%** |
| AUC-ROC | 0.9576 | 0.9576 |
| PNEUMONIA Recall | 99% | 96% |
| NORMAL Recall | 41% | **85%** |
| F1 (macro avg) | 0.71 | **0.91** |

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
| Batch Web UI    | Upload multiple X-rays via browser — parallel calls + summary cards  |
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
