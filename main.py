"""
Main Entry Point for CNN Telugu Poem Classification System.

Usage:
    python main.py --mode train       # Train both models
    python main.py --mode evaluate    # Evaluate on test set
    python main.py --mode predict     # Interactive prediction
    python main.py --mode all         # Train + Evaluate
    python main.py --mode single      # Train single-task only
    python main.py --mode multi       # Train multi-task only
"""

import argparse
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from model import configure_gpu
from data_preprocessing import prepare_dataset, clean_text
from feature_engineering import prepare_features
from train import train_single_task, train_multitask
from evaluate import evaluate_single_task, evaluate_multitask
from interpretation import get_interpretation


def run_train(mode: str = 'both'):
    """Run the full training pipeline."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("CNN TELUGU POEM CLASSIFICATION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"  GPU Config: Mixed Precision = {config.MIXED_PRECISION}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Embedding: {config.EMBEDDING_DIM}d")
    print(f"  Vocab: {config.VOCAB_SIZE}")
    print(f"  Max Seq Length: {config.MAX_SEQ_LEN}")

    # Step 1: Prepare data
    print("\n[Pipeline] Step 1/3: Data Preprocessing...")
    train_df, val_df, test_df = prepare_dataset()

    # Step 2: Feature engineering
    print("\n[Pipeline] Step 2/3: Feature Engineering...")
    features = prepare_features(train_df, val_df, test_df)

    # Step 3: Train
    print("\n[Pipeline] Step 3/3: Training Models...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if mode in ('single', 'both'):
        train_single_task(features)

    if mode in ('multi', 'both'):
        train_multitask(features)

    print("\n✅ Training complete!")
    return features


def run_evaluate():
    """Run the full evaluation pipeline."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("CNN TELUGU POEM CLASSIFICATION — EVALUATION")
    print("=" * 60)

    # Prepare data (needed for test set)
    train_df, val_df, test_df = prepare_dataset()
    features = prepare_features(train_df, val_df, test_df)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    evaluate_single_task(features)
    evaluate_multitask(features)

    print("\n✅ Evaluation complete! Check outputs/ for visualizations.")


def run_predict():
    """Interactive prediction mode."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("CNN TELUGU POEM CLASSIFICATION — INTERACTIVE PREDICTION")
    print("=" * 60)

    # Load models
    if not os.path.exists(config.TOKENIZER_PATH):
        print("❌ Models not found! Run training first: python main.py --mode train")
        sys.exit(1)

    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)

    model = None
    if os.path.exists(config.CHANDAS_MODEL_PATH):
        model = tf.keras.models.load_model(config.CHANDAS_MODEL_PATH)

    multitask = None
    source_encoder = None
    if os.path.exists(config.MULTITASK_MODEL_PATH):
        multitask = tf.keras.models.load_model(config.MULTITASK_MODEL_PATH)
        with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
            source_encoder = pickle.load(f)

    print("\nEnter Telugu poem text (press Enter twice to predict, 'quit' to exit):\n")

    while True:
        lines = []
        print("─" * 40)
        while True:
            line = input()
            if line.strip().lower() == 'quit':
                print("👋 Exiting.")
                sys.exit(0)
            if line == '' and lines:
                break
            lines.append(line)

        poem_text = '\n'.join(lines)
        cleaned = clean_text(poem_text)

        # Tokenize
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=config.MAX_SEQ_LEN,
                               padding='post', truncating='post')

        print("\n📊 Results:")
        print("─" * 40)

        # Single-task prediction
        if model:
            pred = model.predict(padded, verbose=0)[0]
            chandas_idx = np.argmax(pred)
            chandas = chandas_encoder.classes_[chandas_idx]
            confidence = pred[chandas_idx]
            print(f"  Chandas:    {chandas} ({confidence*100:.1f}%)")

            # Map to class
            chandas_to_class = {
                'seesamu': 'vupajaathi', 'teytageethi': 'vupajaathi',
                'aataveladi': 'vupajaathi',
                'mattebhamu': 'vruttamu', 'champakamaala': 'vruttamu',
                'vutpalamaala': 'vruttamu', 'saardulamu': 'vruttamu',
                'kandamu': 'jaathi'
            }
            print(f"  Class:      {chandas_to_class.get(chandas, 'unknown')}")

        # Multi-task prediction
        if multitask and source_encoder:
            chandas_pred, source_pred = multitask.predict(padded, verbose=0)
            source_idx = np.argmax(source_pred[0])
            source = source_encoder.classes_[source_idx]
            source_conf = source_pred[0][source_idx]
            print(f"  Source:     {source} ({source_conf*100:.1f}%)")

        # Interpretation
        interp = get_interpretation(poem_text)
        method = "Extracted" if interp['method'] == 'extracted' else "Keywords"
        print(f"\n  📖 Interpretation ({method}):")
        print(f"     {interp['interpretation']}")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CNN Telugu Poem Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train       Train both models
  python main.py --mode evaluate    Evaluate on test set
  python main.py --mode predict     Interactive prediction
  python main.py --mode all         Train + Evaluate
  python main.py --mode single      Train single-task CNN only
  python main.py --mode multi       Train multi-task CNN only
        """
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['train', 'evaluate', 'predict', 'all', 'single', 'multi'],
        help='Operation mode'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        run_train(mode='both')
    elif args.mode == 'single':
        run_train(mode='single')
    elif args.mode == 'multi':
        run_train(mode='multi')
    elif args.mode == 'evaluate':
        run_evaluate()
    elif args.mode == 'predict':
        run_predict()
    elif args.mode == 'all':
        features = run_train(mode='both')
        evaluate_single_task(features)
        evaluate_multitask(features)


if __name__ == "__main__":
    main()
