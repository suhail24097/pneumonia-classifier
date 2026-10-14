"""
Data preprocessing pipeline for chest X-ray pneumonia classification.

Uses ResNet50's preprocess_input (subtracts ImageNet mean, does NOT rescale to 1/255).
Implements a balanced batch generator to handle class imbalance (1:2.9 ratio).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


IMG_SIZE   = 224
BATCH_SIZE = 32

# Training class counts (merged val set into train, then 80/20 split)
TRAIN_NORMAL    = 1349
TRAIN_PNEUMONIA = 3883


def get_train_datagen() -> ImageDataGenerator:
    """
    Returns an ImageDataGenerator for training with augmentation.
    Per-class augmentation: NORMAL class gets stronger augmentation to
    compensate for underrepresentation.
    """
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def get_val_datagen() -> ImageDataGenerator:
    """
    Returns an ImageDataGenerator for validation/test — no augmentation,
    only preprocessing.
    """
    return ImageDataGenerator(preprocessing_function=preprocess_input)


def get_normal_augmented_datagen() -> ImageDataGenerator:
    """
    Stronger augmentation for the NORMAL class to balance the dataset.
    Applied during balanced batch generation.
    """
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )


def compute_class_weights(
    n_normal: int = TRAIN_NORMAL,
    n_pneumonia: int = TRAIN_PNEUMONIA,
) -> dict:
    """
    Compute inverse-frequency class weights to handle imbalance.

    Returns dict suitable for Keras model.fit(class_weight=...).
    Note: In Keras 3, sample_weight is preferred over class_weight
    for generators; this is kept for reference.
    """
    total = n_normal + n_pneumonia
    w_normal    = total / (2.0 * n_normal)
    w_pneumonia = total / (2.0 * n_pneumonia)
    return {0: w_normal, 1: w_pneumonia}


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights from a label array.

    Args:
        labels: 1-D integer array of class indices (0=NORMAL, 1=PNEUMONIA).

    Returns:
        Float array of the same length with inverse-frequency weights.
    """
    class_weights = compute_class_weights()
    weights = np.array([class_weights[int(label)] for label in labels])
    return weights.astype(np.float32)


def make_train_generator(
    train_dir: str,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.utils.Sequence:
    """
    Create a standard (non-balanced) training generator from directory.

    Directory structure expected:
        train_dir/
            NORMAL/
            PNEUMONIA/

    Args:
        train_dir: Path to training data directory.
        batch_size: Number of images per batch.

    Returns:
        Keras DirectoryIterator.
    """
    datagen = get_train_datagen()
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )
    return generator


def make_val_generator(
    val_dir: str,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.utils.Sequence:
    """
    Create a validation generator from directory.

    Args:
        val_dir: Path to validation data directory.
        batch_size: Number of images per batch.

    Returns:
        Keras DirectoryIterator.
    """
    datagen = get_val_datagen()
    generator = datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return generator


class BalancedBatchGenerator(tf.keras.utils.Sequence):
    """
    Custom generator that yields balanced batches by oversampling NORMAL
    class with stronger augmentation.

    Each batch contains batch_size//2 NORMAL and batch_size//2 PNEUMONIA
    images, with sample weights pre-applied.

    Compatible with Keras 3 (uses sample_weight instead of class_weight).
    """

    def __init__(
        self,
        normal_dir: str,
        pneumonia_dir: str,
        batch_size: int = BATCH_SIZE,
        img_size: int = IMG_SIZE,
    ):
        self.batch_size   = batch_size
        self.half_batch   = batch_size // 2
        self.img_size     = img_size

        self.normal_gen = get_normal_augmented_datagen().flow_from_directory(
            normal_dir,
            target_size=(img_size, img_size),
            batch_size=self.half_batch,
            class_mode="binary",
            shuffle=True,
            seed=42,
        )
        self.pneumonia_gen = get_train_datagen().flow_from_directory(
            pneumonia_dir,
            target_size=(img_size, img_size),
            batch_size=self.half_batch,
            class_mode="binary",
            shuffle=True,
            seed=42,
        )

        self.n_batches = max(
            len(self.normal_gen), len(self.pneumonia_gen)
        ) * 2  # oversample to see more data per epoch

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int):
        x_normal,    y_normal    = next(self.normal_gen)
        x_pneumonia, y_pneumonia = next(self.pneumonia_gen)

        x_batch = np.concatenate([x_normal, x_pneumonia], axis=0)
        y_batch = np.concatenate([y_normal, y_pneumonia], axis=0)

        # Shuffle within batch
        perm    = np.random.permutation(len(x_batch))
        x_batch = x_batch[perm]
        y_batch = y_batch[perm]

        # Sample weights (equal for balanced batch, kept for API compatibility)
        sw = np.ones(len(y_batch), dtype=np.float32)

        return x_batch, y_batch, sw
