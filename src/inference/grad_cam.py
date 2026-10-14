"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet50.

Target layer: conv5_block3_out  (final convolutional feature map, 7x7x2048)

Usage:
    heatmap = generate_grad_cam(model, img_array)
    overlay = overlay_grad_cam(original_img_rgb, heatmap)
"""

import numpy as np
import cv2
import tensorflow as tf


TARGET_LAYER = "conv5_block3_out"


def generate_grad_cam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    layer_name: str = TARGET_LAYER,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        model:      Loaded Keras model (the full pneumonia classifier).
        img_array:  Preprocessed image tensor with shape (1, 224, 224, 3).
                    Must already have ResNet50's preprocess_input applied.
        layer_name: Name of the convolutional layer to target.
                    Default is 'conv5_block3_out' (final ResNet50 conv layer).

    Returns:
        heatmap: Float32 numpy array with shape (H, W), values in [0, 1].
                 H and W match the spatial dimensions of the target layer
                 (7x7 for conv5_block3_out with 224x224 input).
    """
    # For nested models loaded from .h5, conv5_block3_out lives inside the 'resnet50'
    # sublayer and its output tensor is not directly connected to the outer model's
    # input graph. We handle this by running the base model eagerly to get conv
    # features, then watching them as a tf.Variable while running the remaining
    # head layers to compute the final score.
    img_tensor = tf.constant(img_array, dtype=tf.float32)

    # Identify the backbone submodel
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find a nested backbone model inside the classifier.")

    # Forward pass through backbone → conv features (no tape needed here)
    conv_features = base_model(img_tensor, training=False)  # (1, 7, 7, 2048)
    conv_var = tf.Variable(conv_features, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(conv_var)
        x = conv_var
        # Run the remaining head layers (everything after the backbone)
        for layer in model.layers:
            if layer.name in ('input_layer_1', base_model.name):
                continue
            if hasattr(layer, 'name') and 'input' in layer.name:
                continue
            x = layer(x, training=False)
        class_score = x[:, 0]

    grads = tape.gradient(class_score, conv_var)           # (1, 7, 7, 2048)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (2048,)

    conv_out_np = conv_features[0].numpy()                 # (7, 7, 2048)
    heatmap = conv_out_np @ pooled_grads.numpy()[..., np.newaxis]  # (7, 7, 1)
    heatmap = np.squeeze(heatmap)                          # (7, 7)
    heatmap = np.maximum(heatmap, 0)                       # ReLU

    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.astype(np.float32)


def overlay_grad_cam(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        original_img_rgb: Original image as RGB uint8 array (H, W, 3).
                          Can be any spatial size — the heatmap is resized
                          to match.
        heatmap:          Grad-CAM heatmap from generate_grad_cam(), values
                          in [0, 1], shape (h, w).
        alpha:            Blending weight for the heatmap overlay.
                          0 = original image only, 1 = heatmap only.

    Returns:
        overlay_bgr: BGR uint8 image (H, W, 3) with colored heatmap blended
                     onto the original.  BGR format is used for compatibility
                     with OpenCV's imencode.
    """
    h, w = original_img_rgb.shape[:2]

    # Resize heatmap to match the original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to uint8 in [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply JET colormap (blue=cold/low activation, red=hot/high activation)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    # Convert original from RGB to BGR for OpenCV blending
    original_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)

    # Blend
    overlay_bgr = cv2.addWeighted(
        colored_heatmap, alpha,
        original_bgr,    1.0 - alpha,
        0,
    )

    return overlay_bgr.astype(np.uint8)
