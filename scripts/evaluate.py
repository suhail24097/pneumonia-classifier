"""
Standalone evaluation script for the pneumonia classifier.

Loads the trained model, sweeps thresholds from 0.1 to 0.9, identifies
the best threshold by F1-score, prints a classification report, and
saves metrics + plots to the results/ directory.

Usage:
    python scripts/evaluate.py --test-dir data/chest_xray/test/

Directory structure expected under --test-dir:
    test/
        NORMAL/
        PNEUMONIA/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pneumonia classifier on a test dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pneumonia_resnet50.h5",
        help="Path to the Keras .h5 model file.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to test dataset directory (must contain NORMAL/ and PNEUMONIA/ sub-dirs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save metrics.json and plots.",
    )
    return parser.parse_args()


def load_test_data(test_dir: str, img_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from test_dir/NORMAL and test_dir/PNEUMONIA.

    Returns:
        images: Float32 array (N, img_size, img_size, 3) after preprocess_input.
        labels: Int array (N,) with 0=NORMAL, 1=PNEUMONIA.
    """
    import cv2
    from tensorflow.keras.applications.resnet50 import preprocess_input

    class_map = {"NORMAL": 0, "PNEUMONIA": 1}
    images, labels = [], []

    for class_name, label in class_map.items():
        class_dir = Path(test_dir) / class_name
        if not class_dir.exists():
            print(f"WARNING: Directory not found: {class_dir}", file=sys.stderr)
            continue

        img_files = sorted(
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        print(f"  {class_name}: {len(img_files)} images")

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: Could not read {img_path}", file=sys.stderr)
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    images = preprocess_input(images)
    return images, labels


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, dict]:
    """
    Sweep thresholds and return the best threshold (by macro F1) and metrics.

    Returns:
        best_threshold: Threshold value with highest macro F1.
        sweep_results:  Dict mapping threshold -> {accuracy, f1_macro, ...}.
    """
    from sklearn.metrics import f1_score, accuracy_score

    sweep_results = {}
    best_f1        = -1.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, average="macro", zero_division=0)
        acc   = accuracy_score(labels, preds)
        sweep_results[round(float(t), 2)] = {"f1_macro": round(f1, 4), "accuracy": round(acc, 4)}

        if f1 > best_f1:
            best_f1        = f1
            best_threshold = float(t)

    return best_threshold, sweep_results


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)

    # Suppress TF logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    print(f"Loading model from: {model_path}")
    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))

    print(f"\nLoading test images from: {test_dir}")
    images, labels = load_test_data(str(test_dir))
    print(f"  Total: {len(labels)} images  (NORMAL={int((labels==0).sum())}, PNEUMONIA={int((labels==1).sum())})")

    if len(images) == 0:
        print("ERROR: No images loaded.", file=sys.stderr)
        sys.exit(1)

    print("\nRunning inference...")
    probs = model.predict(images, batch_size=args.batch_size, verbose=1).flatten()

    # AUC-ROC
    from sklearn.metrics import (
        roc_auc_score, classification_report, confusion_matrix, roc_curve
    )
    auc = roc_auc_score(labels, probs)
    print(f"\nAUC-ROC: {auc:.4f}")

    # Threshold sweep
    print("\nRunning threshold sweep (0.1 to 0.9)...")
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_threshold, sweep_results = threshold_sweep(probs, labels, thresholds)
    print(f"Best threshold (max macro F1): {best_threshold:.2f}")

    # Classification report at best threshold
    preds_best = (probs >= best_threshold).astype(int)
    print(f"\n--- Classification Report at threshold={best_threshold:.2f} ---")
    print(classification_report(labels, preds_best, target_names=["NORMAL", "PNEUMONIA"]))

    # Confusion matrix
    cm = confusion_matrix(labels, preds_best)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\nSensitivity (PNEUMONIA recall): {sensitivity:.4f}")
    print(f"Specificity (NORMAL recall):    {specificity:.4f}")

    # Save metrics.json
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    metrics = {
        "model":        str(model_path),
        "test_dir":     str(test_dir),
        "auc_roc":      round(auc, 4),
        "best_threshold": best_threshold,
        "sensitivity":  round(sensitivity, 4),
        "specificity":  round(specificity, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "at_best_threshold": {
            "accuracy":           round(accuracy_score(labels, preds_best), 4),
            "f1_macro":           round(f1_score(labels, preds_best, average="macro"), 4),
            "normal_precision":   round(precision_score(labels, preds_best, pos_label=0), 4),
            "normal_recall":      round(recall_score(labels, preds_best, pos_label=0), 4),
            "pneumonia_precision":round(precision_score(labels, preds_best, pos_label=1), 4),
            "pneumonia_recall":   round(recall_score(labels, preds_best, pos_label=1), 4),
        },
        "threshold_sweep": sweep_results,
    }

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (threshold={best_threshold:.2f})")
        cm_path = results_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrix saved to: {cm_path}")

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="blue")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        roc_path = results_dir / "roc_curve.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"ROC curve saved to: {roc_path}")

    except ImportError:
        print("WARNING: matplotlib/seaborn not installed — skipping plots.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
