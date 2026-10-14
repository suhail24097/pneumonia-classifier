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
    # Build a sub-model that outputs both the target conv layer activations
    # and the final prediction.
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        # Cast to float32 and watch the input
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        # For binary sigmoid output: class score is the raw prediction
        class_score = predictions[:, 0]

    # Gradients of the class score w.r.t. conv layer feature maps
    grads = tape.gradient(class_score, conv_outputs)  # (1, H, W, C)

    # Global average pooling over spatial dims to get neuron importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Extract the conv feature maps for this image (H, W, C)
    conv_outputs = conv_outputs[0]

    # Weighted combination of feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                           # (H, W)

    # ReLU: keep only positive activations (features that increase the score)
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    heatmap = heatmap.numpy()
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
