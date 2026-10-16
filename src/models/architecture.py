"""
Pneumonia classifier architectures — ResNet50 (v1) and DenseNet121 (v2).

v1 — ResNet50:
    ResNet50 (ImageNet) → GAP → BN → Dense(256) → Dropout(0.4)
    → Dense(64) → Dropout(0.2) → Sigmoid

v2 — DenseNet121:
    DenseNet121 (ImageNet) → GAP → BN → Dense(512) → Dropout(0.5)
    → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)
    Training: Focal Loss + Cosine LR Decay + 2-phase fine-tuning
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50, DenseNet121


IMG_SIZE = 224


def build_densenet121_classifier(trainable_base: bool = False) -> Model:
    """
    Build DenseNet121-based pneumonia classifier (v2).

    Improvements over v1:
        - DenseNet121 backbone: denser feature reuse, fewer parameters
        - Larger head (Dense 512→256) for more capacity
        - Designed for Focal Loss + Cosine LR training

    Args:
        trainable_base: If False (Phase 1), base frozen; if True (Phase 2),
                        caller unfreezes top layers selectively.
    Returns:
        Uncompiled Keras Model.
    """
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn")(x)

    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.5, name="dropout_05")(x)

    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_03")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="pneumonia_densenet121")


def unfreeze_top_layers_densenet(model: Model, n_layers: int = 50) -> Model:
    """Unfreeze top n_layers of DenseNet121 base for Phase 2 fine-tuning."""
    base_model = model.get_layer("densenet121")
    base_model.trainable = True
    freeze_until = len(base_model.layers) - n_layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    return model


def build_resnet50_classifier(trainable_base: bool = False) -> Model:
    """
    Build and return the ResNet50-based pneumonia classifier.

    Args:
        trainable_base: If False (Phase 1), all ResNet50 layers are frozen.
                        If True (Phase 2), all layers are trainable —
                        caller should then selectively freeze lower layers
                        (e.g. keep only top 30 unfrozen).

    Returns:
        Compiled Keras Model ready for training.
    """
    # Load ResNet50 without top classification layers
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = trainable_base

    # Custom classification head
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn")(x)

    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout_04")(x)

    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.2, name="dropout_02")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="pneumonia_resnet50")
    return model


def unfreeze_top_layers(model: Model, n_layers: int = 30) -> Model:
    """
    Unfreeze the top n_layers of the ResNet50 base for Phase 2 fine-tuning.

    Layers below the threshold remain frozen to preserve low-level features
    learned from ImageNet.

    Args:
        model:    The model returned by build_resnet50_classifier.
        n_layers: Number of layers from the end of the base to unfreeze.

    Returns:
        The same model with selected layers made trainable.
    """
    base_model = model.get_layer("resnet50")
    base_model.trainable = True

    # Freeze all but the last n_layers
    freeze_until = len(base_model.layers) - n_layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    return model


def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Compile the model with Adam optimizer and binary cross-entropy loss.

    Metrics include AUC-ROC (used for early stopping) and standard accuracy.

    Args:
        model:         Keras Model to compile.
        learning_rate: Adam learning rate.

    Returns:
        Compiled model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_callbacks(
    checkpoint_path: str = "models/best_model.h5",
    monitor: str = "val_auc",
) -> list:
    """
    Return standard training callbacks: EarlyStopping + ReduceLROnPlateau +
    ModelCheckpoint.

    Args:
        checkpoint_path: Where to save the best model weights.
        monitor:         Metric to monitor ('val_auc' recommended).

    Returns:
        List of Keras callbacks.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=5,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        mode="max",
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    return [early_stop, reduce_lr, checkpoint]
