"""
Export the trained Keras (.h5) pneumonia classifier to ONNX format.

Requires:
    pip install tf2onnx onnx

Usage:
    python scripts/export_onnx.py \
        --model models/pneumonia_resnet50.h5 \
        --output models/pneumonia_resnet50.onnx
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Keras .h5 model to ONNX format using tf2onnx."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pneumonia_resnet50.h5",
        help="Path to the input Keras .h5 model file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pneumonia_resnet50.onnx",
        help="Path to save the output .onnx file.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to use (default: 13).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path  = Path(args.model)
    output_path = Path(args.output)

    # Validate input
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Print model input/output info
    print("\n--- Model Summary ---")
    for inp in model.inputs:
        print(f"  Input  : name={inp.name}, shape={inp.shape}, dtype={inp.dtype}")
    for out in model.outputs:
        print(f"  Output : name={out.name}, shape={out.shape}, dtype={out.dtype}")
    print(f"  Total parameters: {model.count_params():,}")
    print("---------------------\n")

    print(f"Converting to ONNX (opset={args.opset})...")

    try:
        import tf2onnx
        import onnx

        # Convert using tf2onnx
        input_signature = [
            tf.TensorSpec(
                shape=inp.shape,
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in model.inputs
        ]

        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=args.opset,
            output_path=str(output_path),
        )

    except ImportError:
        print(
            "ERROR: tf2onnx not installed. Run: pip install tf2onnx onnx",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: ONNX conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify the exported model
    try:
        import onnx
        onnx_model_loaded = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model_loaded)
        print("ONNX model verification: PASSED")
    except Exception as e:
        print(f"WARNING: ONNX model verification failed: {e}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExport complete!")
    print(f"  Output file : {output_path}")
    print(f"  File size   : {file_size_mb:.1f} MB")
    print(f"\nTo run inference with ONNX Runtime:")
    print("    pip install onnxruntime")
    print("    import onnxruntime as ort, numpy as np")
    print(f"    session = ort.InferenceSession('{output_path}')")
    print("    inputs  = {session.get_inputs()[0].name: img_array}  # (1,224,224,3) float32")
    print("    prob    = session.run(None, inputs)[0][0][0]")


if __name__ == "__main__":
    main()
