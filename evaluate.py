"""
Evaluation Module for CNN Telugu Poem Classification System.

Generates comprehensive evaluation metrics and visualizations:
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- Training curves (accuracy + loss over epochs)
- Overall accuracy score
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
import config
from model import configure_gpu


def load_model_and_encoders(model_path: str) -> tuple:
    """Load saved model, tokenizer, and label encoders."""
    model = tf.keras.models.load_model(model_path)
    print(f"[Eval] Loaded model from {model_path}")

    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)
    with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
        source_encoder = pickle.load(f)

    return model, tokenizer, chandas_encoder, source_encoder


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list, save_path: str,
                          title: str = "Confusion Matrix"):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true: True label indices
        y_pred: Predicted label indices
        labels: Class label names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Confusion matrix saved to {save_path}")


def plot_training_curves(history: dict, save_path: str,
                         title_prefix: str = ""):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        title_prefix: Prefix for plot titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Find the correct key names (they vary based on single vs multi-task)
    acc_keys = [k for k in history.keys() if 'accuracy' in k and 'val' not in k]
    val_acc_keys = [k for k in history.keys() if 'accuracy' in k and 'val' in k]
    loss_keys = [k for k in history.keys() if 'loss' in k and 'val' not in k]
    val_loss_keys = [k for k in history.keys() if 'loss' in k and 'val' in k]

    # Accuracy plot
    for k in acc_keys:
        axes[0].plot(history[k], label=f'Train {k}')
    for k in val_acc_keys:
        axes[0].plot(history[k], label=f'{k}', linestyle='--')
    axes[0].set_title(f'{title_prefix}Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    for k in loss_keys:
        axes[1].plot(history[k], label=f'Train {k}')
    for k in val_loss_keys:
        axes[1].plot(history[k], label=f'{k}', linestyle='--')
    axes[1].set_title(f'{title_prefix}Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Training curves saved to {save_path}")


def evaluate_single_task(features: dict):
    """
    Evaluate the single-task CNN model on the test set.

    Args:
        features: Dictionary from prepare_features()
    """
    print("\n" + "=" * 60)
    print("EVALUATION: Single-Task CNN (Chandas)")
    print("=" * 60)

    configure_gpu()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load model
    model, _, chandas_encoder, _ = load_model_and_encoders(config.CHANDAS_MODEL_PATH)

    # Predict
    y_pred_proba = model.predict(features['X_test'], batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(features['y_chandas_test'], axis=1)
    labels = list(chandas_encoder.classes_)

    # Classification report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=labels,
                                   digits=4, zero_division=0)
    print(report)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n--- Overall Metrics ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels, config.CONFUSION_MATRIX_PATH,
                          title="Chandas Classification")

    # Training curves
    if os.path.exists(config.HISTORY_PATH):
        with open(config.HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        plot_training_curves(history, config.TRAINING_CURVES_PATH,
                             title_prefix="Chandas CNN — ")

    # Save report to file
    report_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("CNN Telugu Poem Classification — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config.CHANDAS_MODEL_PATH}\n")
        f.write(f"Test samples: {len(y_true)}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy:  {acc:.4f}\n")
        f.write(f"Overall Precision: {prec:.4f}\n")
        f.write(f"Overall Recall:    {rec:.4f}\n")
        f.write(f"Overall F1-Score:  {f1:.4f}\n")
    print(f"\n[Eval] Report saved to {report_path}")


def evaluate_multitask(features: dict):
    """
    Evaluate the multi-task CNN model on the test set.
    """
    print("\n" + "=" * 60)
    print("EVALUATION: Multi-Task CNN (Chandas + Source)")
    print("=" * 60)

    if not os.path.exists(config.MULTITASK_MODEL_PATH):
        print("[Eval] Multi-task model not found — skipping.")
        return

    configure_gpu()
    model = tf.keras.models.load_model(config.MULTITASK_MODEL_PATH)

    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)
    with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
        source_encoder = pickle.load(f)

    # Predict
    chandas_pred_proba, source_pred_proba = model.predict(
        features['X_test'], batch_size=config.BATCH_SIZE
    )

    # Chandas evaluation
    chandas_pred = np.argmax(chandas_pred_proba, axis=1)
    chandas_true = np.argmax(features['y_chandas_test'], axis=1)
    chandas_labels = list(chandas_encoder.classes_)

    print("\n--- Chandas Classification Report ---")
    print(classification_report(chandas_true, chandas_pred,
                                target_names=chandas_labels, digits=4, zero_division=0))

    # Source evaluation
    source_pred = np.argmax(source_pred_proba, axis=1)
    source_true = np.argmax(features['y_source_test'], axis=1)
    source_labels = list(source_encoder.classes_)

    print("--- Source Classification Report ---")
    print(classification_report(source_true, source_pred,
                                target_names=source_labels, digits=4, zero_division=0))

    # Confusion matrices
    plot_confusion_matrix(chandas_true, chandas_pred, chandas_labels,
                          config.MULTITASK_CONFUSION_PATH,
                          title="Multi-Task Chandas")

    # Training curves
    mt_history_path = config.HISTORY_PATH.replace('.pkl', '_multitask.pkl')
    if os.path.exists(mt_history_path):
        with open(mt_history_path, 'rb') as f:
            history = pickle.load(f)
        plot_training_curves(history, config.MULTITASK_CURVES_PATH,
                             title_prefix="Multi-Task CNN — ")


if __name__ == "__main__":
    from data_preprocessing import prepare_dataset
    from feature_engineering import prepare_features

    train_df, val_df, test_df = prepare_dataset()
    features = prepare_features(train_df, val_df, test_df)
    evaluate_single_task(features)
    evaluate_multitask(features)
