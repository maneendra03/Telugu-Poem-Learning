"""
Training Pipeline for CNN Telugu Poem Classification System.

Handles training for both single-task (chandas only) and multi-task
(chandas + source) CNN models. Uses callbacks for early stopping,
learning rate reduction, and model checkpointing.

Optimized for NVIDIA H200 GPU with large batch sizes and mixed precision.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import config
from model import build_cnn_model, build_multitask_cnn, configure_gpu
from data_preprocessing import prepare_dataset
from feature_engineering import prepare_features


def get_callbacks(model_path: str, monitor: str = 'val_loss') -> list:
    """
    Build training callbacks optimized for H200 GPU training.

    Args:
        model_path: Path to save the best model
        monitor: Metric to monitor for early stopping

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Stop if no improvement for EARLY_STOP_PATIENCE epochs
        EarlyStopping(
            monitor=monitor,
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        # Save best model
        ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),

        # Reduce LR on plateau
        ReduceLROnPlateau(
            monitor=monitor,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    return callbacks


def compute_class_weights(y_labels: np.ndarray) -> dict:
    """
    Compute class weights to handle imbalanced chandas distribution.
    Poems like 'aataveladi' (995) vs 'saardulamu' (290) need balancing.

    Args:
        y_labels: One-hot encoded labels array

    Returns:
        Dictionary mapping class index → weight
    """
    from sklearn.utils.class_weight import compute_class_weight

    class_indices = np.argmax(y_labels, axis=1)
    unique_classes = np.unique(class_indices)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=class_indices
    )

    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    print(f"[Training] Class weights: {class_weight_dict}")
    return class_weight_dict


def train_single_task(features: dict) -> dict:
    """
    Train the single-task CNN for chandas prediction.

    Args:
        features: Dictionary from prepare_features()

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING: Single-Task CNN (Chandas Prediction)")
    print("=" * 60)

    n_classes = features['y_chandas_train'].shape[1]
    print(f"  Classes: {n_classes}")
    print(f"  Train samples: {len(features['X_train'])}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max epochs: {config.EPOCHS}")

    # Build model
    model = build_cnn_model(n_classes=n_classes)
    model.summary()

    # Class weights for imbalanced data
    class_weights = compute_class_weights(features['y_chandas_train'])

    # Callbacks
    callbacks = get_callbacks(config.CHANDAS_MODEL_PATH)

    # Train
    print(f"\n[Training] Starting on H200 GPU (batch={config.BATCH_SIZE})...")
    history = model.fit(
        features['X_train'],
        features['y_chandas_train'],
        validation_data=(features['X_val'], features['y_chandas_val']),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    with open(config.HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n[Training] Model saved to {config.CHANDAS_MODEL_PATH}")
    print(f"[Training] History saved to {config.HISTORY_PATH}")

    return history.history


def train_multitask(features: dict) -> dict:
    """
    Train the multi-task CNN for chandas + source prediction.

    Args:
        features: Dictionary from prepare_features()

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING: Multi-Task CNN (Chandas + Source)")
    print("=" * 60)

    n_chandas = features['y_chandas_train'].shape[1]
    n_source = features['y_source_train'].shape[1]
    print(f"  Chandas classes: {n_chandas}")
    print(f"  Source classes: {n_source}")
    print(f"  Train samples: {len(features['X_train'])}")

    # Build model
    model = build_multitask_cnn(n_chandas=n_chandas, n_source=n_source)
    model.summary()

    # Callbacks
    callbacks = get_callbacks(
        config.MULTITASK_MODEL_PATH,
        monitor='val_chandas_output_loss'
    )

    # Train with both labels
    print(f"\n[Training] Starting multi-task on H200 GPU...")
    history = model.fit(
        features['X_train'],
        {
            'chandas_output': features['y_chandas_train'],
            'source_output': features['y_source_train']
        },
        validation_data=(
            features['X_val'],
            {
                'chandas_output': features['y_chandas_val'],
                'source_output': features['y_source_val']
            }
        ),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    mt_history_path = config.HISTORY_PATH.replace('.pkl', '_multitask.pkl')
    with open(mt_history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n[Training] Model saved to {config.MULTITASK_MODEL_PATH}")

    return history.history


def run_training(mode: str = 'both'):
    """
    Full training pipeline: data → features → model → train → save.

    Args:
        mode: 'single' for chandas only, 'multi' for multi-task, 'both' for both
    """
    # Configure GPU
    configure_gpu()

    # Prepare data
    print("\n[Pipeline] Step 1/3: Loading & preprocessing data...")
    train_df, val_df, test_df = prepare_dataset()

    # Prepare features
    print("\n[Pipeline] Step 2/3: Feature engineering...")
    features = prepare_features(train_df, val_df, test_df)

    # Train
    print("\n[Pipeline] Step 3/3: Training...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if mode in ('single', 'both'):
        train_single_task(features)

    if mode in ('multi', 'both'):
        train_multitask(features)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run_training(mode='both')
