import base64
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

from src.inference.grad_cam import generate_grad_cam, overlay_grad_cam


IMG_SIZE          = 224
DEFAULT_THRESHOLD = 0.87
LABELS            = {0: "NORMAL", 1: "PNEUMONIA"}


class PneumoniaPredictor:
    def __init__(self, model_path: str, threshold: float = DEFAULT_THRESHOLD):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model     = load_model(str(path))
        self.threshold = threshold
        self.img_size  = IMG_SIZE
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Must be a valid JPEG or PNG.")
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return preprocess_input(img)

    def predict(self, image_bytes: bytes) -> dict:
        t0             = time.time()
        tensor         = self.preprocess(image_bytes)
        prob_pneumonia = float(self.model.predict(tensor, verbose=0)[0][0])
        pred_class     = 1 if prob_pneumonia >= self.threshold else 0
        elapsed_ms     = round((time.time() - t0) * 1000, 1)

        return {
            "prediction":        LABELS[pred_class],
            "confidence":        round(prob_pneumonia if pred_class == 1 else 1 - prob_pneumonia, 4),
            "prob_pneumonia":    round(prob_pneumonia, 4),
            "prob_normal":       round(1 - prob_pneumonia, 4),
            "threshold_used":    self.threshold,
            "inference_time_ms": elapsed_ms,
        }

    def predict_batch(self, images: list[bytes]) -> list[dict]:
        tensors = np.vstack([self.preprocess(img) for img in images])
        probs   = self.model.predict(tensors, verbose=0).flatten()
        results = []
        for prob in probs:
            pred = 1 if prob >= self.threshold else 0
            results.append({
                "prediction":     LABELS[pred],
                "confidence":     round(float(prob) if pred == 1 else float(1 - prob), 4),
                "prob_pneumonia": round(float(prob), 4),
                "prob_normal":    round(float(1 - prob), 4),
            })
        return results

    def predict_with_explanation(self, image_bytes: bytes) -> dict:
        """
        Run prediction and generate a Grad-CAM explanation overlay.

        Args:
            image_bytes: Raw bytes of the input JPEG or PNG image.

        Returns:
            dict with all fields from predict() plus:
                grad_cam_b64 (str): Base64-encoded PNG of the Grad-CAM
                                    heatmap overlaid on the original image.
        """
        t0 = time.time()

        # Decode and keep original RGB for overlay
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image. Must be a valid JPEG or PNG.")

        img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb_orig, (self.img_size, self.img_size))

        # Preprocessed tensor for model input
        tensor = np.expand_dims(img_rgb_resized.astype(np.float32), axis=0)
        tensor = preprocess_input(tensor)

        # Model prediction
        prob_pneumonia = float(self.model.predict(tensor, verbose=0)[0][0])
        pred_class     = 1 if prob_pneumonia >= self.threshold else 0

        # Grad-CAM heatmap
        heatmap = generate_grad_cam(self.model, tensor)

        # Overlay on resized original image
        overlay_bgr = overlay_grad_cam(img_rgb_resized, heatmap, alpha=0.4)

        # Encode overlay as PNG -> base64
        success, buf = cv2.imencode(".png", overlay_bgr)
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM overlay as PNG.")
        grad_cam_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        result = {
            "prediction":        LABELS[pred_class],
            "confidence":        round(prob_pneumonia if pred_class == 1 else 1 - prob_pneumonia, 4),
            "prob_pneumonia":    round(prob_pneumonia, 4),
            "prob_normal":       round(1 - prob_pneumonia, 4),
            "threshold_used":    self.threshold,
            "inference_time_ms": elapsed_ms,
            "grad_cam_b64":      grad_cam_b64,
        }
        return result
