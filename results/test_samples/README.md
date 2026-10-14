# Test Samples

Place chest X-ray test images here to try the API or run evaluation.

## Dataset Download

The model was trained and evaluated on the **Kaggle Chest X-Ray Images (Pneumonia)** dataset.

1. Create a Kaggle account at https://www.kaggle.com
2. Download the dataset:
   ```
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```
   This creates `data/chest_xray/train/`, `data/chest_xray/val/`, `data/chest_xray/test/`
   with `NORMAL/` and `PNEUMONIA/` subdirectories in each.

## Running Evaluation

```bash
python scripts/evaluate.py --test-dir data/chest_xray/test/
```

This will:
- Run inference on all 624 test images
- Sweep thresholds from 0.1 to 0.9
- Print a classification report at the best threshold
- Save `results/metrics.json`, `results/confusion_matrix.png`, `results/roc_curve.png`

## Testing the API

### Start the server

```bash
# With Docker
docker compose up

# Or locally
uvicorn src.api.main:app --reload --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
```

### Single image prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/xray.jpg" \
  | python -m json.tool
```

### Prediction with Grad-CAM explanation

```bash
curl -X POST http://localhost:8000/predict/explain \
  -F "file=@path/to/xray.jpg" \
  | python -m json.tool
```

The response includes `grad_cam_image` — a base64-encoded PNG. To save it:

```python
import base64, json, requests

resp = requests.post(
    "http://localhost:8000/predict/explain",
    files={"file": open("xray.jpg", "rb")},
)
data = resp.json()
print(data["prediction"], data["prob_pneumonia"])

with open("grad_cam_output.png", "wb") as f:
    f.write(base64.b64decode(data["grad_cam_image"]))
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  | python -m json.tool
```
