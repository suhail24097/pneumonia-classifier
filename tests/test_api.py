import io
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app

client = TestClient(app)


def make_dummy_jpeg() -> bytes:
    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture(autouse=True)
def mock_predictor():
    mock = MagicMock()
    mock.threshold = 0.87
    mock.predict.return_value = {
        "prediction":        "PNEUMONIA",
        "confidence":        0.95,
        "prob_pneumonia":    0.95,
        "prob_normal":       0.05,
        "threshold_used":    0.87,
        "inference_time_ms": 180.0,
    }
    mock.predict_batch.return_value = [mock.predict.return_value]
    with patch("src.api.main.predictor", mock):
        yield mock


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_image():
    jpeg = make_dummy_jpeg()
    resp = client.post(
        "/predict",
        files={"file": ("xray.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= body["prob_pneumonia"] <= 1.0


def test_predict_invalid_content_type():
    resp = client.post(
        "/predict",
        files={"file": ("file.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert resp.status_code == 422


def test_predict_empty_file():
    resp = client.post(
        "/predict",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert resp.status_code == 422


def test_predict_custom_threshold():
    jpeg = make_dummy_jpeg()
    resp = client.post(
        "/predict?threshold=0.5",
        files={"file": ("xray.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert resp.status_code == 200


def test_batch_predict():
    jpeg = make_dummy_jpeg()
    resp = client.post(
        "/predict/batch",
        files=[
            ("files", ("a.jpg", io.BytesIO(jpeg), "image/jpeg")),
            ("files", ("b.jpg", io.BytesIO(jpeg), "image/jpeg")),
        ],
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_model_info():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    assert "backbone" in resp.json()
