import base64
import os
from contextlib import asynccontextmanager
from typing import Annotated
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference.predictor import PneumoniaPredictor, DEFAULT_THRESHOLD


MODEL_PATH = os.getenv("MODEL_PATH", "models/pneumonia_resnet50.h5")
THRESHOLD  = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

predictor: PneumoniaPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = PneumoniaPredictor(model_path=MODEL_PATH, threshold=THRESHOLD)
    print(f"Model loaded from: {MODEL_PATH}  |  threshold: {THRESHOLD}")
    yield
    predictor = None


app = FastAPI(
    title="Pneumonia Detection API",
    description="Chest X-ray binary classification: NORMAL vs PNEUMONIA using ResNet50.",
    version="1.0.0",
    lifespan=lifespan,
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_static_dir / "index.html"))


class PredictionResponse(BaseModel):
    prediction:        str
    confidence:        float
    prob_pneumonia:    float
    prob_normal:       float
    threshold_used:    float
    inference_time_ms: float


class ExplainPredictionResponse(BaseModel):
    prediction:        str
    confidence:        float
    prob_pneumonia:    float
    prob_normal:       float
    threshold_used:    float
    inference_time_ms: float
    grad_cam_image:    str  # base64-encoded PNG



class HealthResponse(BaseModel):
    status:    str
    model:     str
    threshold: float


def _validate_image(file: UploadFile) -> None:
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail="File too large. Maximum allowed size is 10 MB."
        )


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {
        "status":    "ok" if predictor else "model_not_loaded",
        "model":     MODEL_PATH,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: Annotated[UploadFile, File(description="Chest X-ray image (JPEG or PNG)")],
    threshold: float = Query(
        default=None,
        ge=0.0, le=1.0,
        description="Override classification threshold (default uses server setting)"
    ),
):
    _validate_image(file)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        if threshold is not None:
            predictor.threshold = threshold

        result = predictor.predict(image_bytes)

        if threshold is not None:
            predictor.threshold = THRESHOLD

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/explain", response_model=ExplainPredictionResponse, tags=["Inference"])
async def predict_explain(
    file: Annotated[UploadFile, File(description="Chest X-ray image (JPEG or PNG)")],
):
    """
    Run prediction and return a Grad-CAM explanation overlay.

    Returns all standard prediction fields plus `grad_cam_image`:
    a base64-encoded PNG showing which regions of the X-ray influenced
    the model's decision (red = high activation, blue = low activation).
    """
    _validate_image(file)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        result = predictor.predict_with_explanation(image_bytes)
        # Rename key to match API response model
        result["grad_cam_image"] = result.pop("grad_cam_b64")
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation inference failed: {str(e)}")




class BatchPredictionResponse(BaseModel):
    count:   int
    results: list[PredictionResponse]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(description="One or more chest X-ray images (JPEG or PNG)"),
):
    """Run prediction on multiple images and return per-image results plus total count."""
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded.")
    results = []
    for file in files:
        _validate_image(file)
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail=f"File '{file.filename}' is empty.")
        results.append(predictor.predict(image_bytes))
    return {"count": len(results), "results": results}


@app.get("/model/info", tags=["System"])
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_path":      MODEL_PATH,
        "threshold":       predictor.threshold,
        "input_size":      f"{predictor.img_size}x{predictor.img_size}",
        "classes":         ["NORMAL", "PNEUMONIA"],
        "framework":       "TensorFlow / Keras",
        "backbone":        "ResNet50 (ImageNet pretrained)",
    }
