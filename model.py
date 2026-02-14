"""
CNN Model Architecture Module for Telugu Poem Classification.

Provides two model architectures optimized for NVIDIA H200 GPU:
1. Single-task CNN — predicts chandas (meter type)
2. Multi-task CNN — predicts chandas + source simultaneously

Architecture leverages deep convolutions with increasing abstraction
to capture both local syllable patterns (kernel=5) and broader
rhythmic structures (kernel=3) in Telugu poetry.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import config


def configure_gpu():
    """Configure GPU settings for optimal H200 performance."""
    # Enable mixed precision (FP16) for faster computation on H200
    if config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[GPU] Mixed precision (FP16) enabled for H200")

    # Dynamic memory growth
    if config.GPU_MEMORY_GROWTH:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        else:
            print("[GPU] No GPU detected — running on CPU")


def build_cnn_model(n_classes: int, name: str = "chandas_cnn") -> Model:
    """
    Build a single-task CNN model for poem classification.

    Architecture (H200-optimized):
        Embedding(30000, 200) → input_length=400
        Conv1D(256, kernel=5, relu) → MaxPool(3)
        Conv1D(128, kernel=3, relu) → MaxPool(3)
        Conv1D(64, kernel=3, relu) → GlobalMaxPool
        Dropout(0.4)
        Dense(256, relu) → Dropout(0.3)
        Dense(128, relu)
        Dense(n_classes, softmax)

    This three-layer convolution stack mirrors how humans process
    poetic rhythm: first detecting syllable-level patterns (Conv1),
    then phrase-level meter (Conv2), then overall structure (Conv3).

    Args:
        n_classes: Number of output classes
        name: Model name string

    Returns:
        Compiled Keras Model
    """
    model = tf.keras.Sequential([
        # Embedding: learn dense representations for Telugu tokens
        layers.Embedding(
            input_dim=config.VOCAB_SIZE,
            output_dim=config.EMBEDDING_DIM,
            input_length=config.MAX_SEQ_LEN,
            name='embedding'
        ),

        # Conv Block 1: detect local syllable patterns (5-gram)
        layers.Conv1D(
            filters=config.CONV1_FILTERS,
            kernel_size=config.CONV1_KERNEL,
            activation='relu',
            padding='same',
            name='conv1'
        ),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling1D(pool_size=config.POOL_SIZE, name='pool1'),

        # Conv Block 2: detect phrase-level meter patterns (3-gram)
        layers.Conv1D(
            filters=config.CONV2_FILTERS,
            kernel_size=config.CONV2_KERNEL,
            activation='relu',
            padding='same',
            name='conv2'
        ),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling1D(pool_size=config.POOL_SIZE, name='pool2'),

        # Conv Block 3: detect overall rhythmic structure
        layers.Conv1D(
            filters=config.CONV3_FILTERS,
            kernel_size=config.CONV3_KERNEL,
            activation='relu',
            padding='same',
            name='conv3'
        ),
        layers.BatchNormalization(name='bn3'),
        layers.GlobalMaxPooling1D(name='global_pool'),

        # Classification head
        layers.Dropout(config.DROPOUT_RATE, name='dropout1'),
        layers.Dense(config.DENSE1_UNITS, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(config.DENSE2_UNITS, activation='relu', name='dense2'),

        # Output — use float32 for numerical stability with mixed precision
        layers.Dense(n_classes, activation='softmax', dtype='float32', name='output')
    ], name=name)

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_multitask_cnn(n_chandas: int, n_source: int,
                        name: str = "multitask_cnn") -> Model:
    """
    Build a multi-task CNN model with shared feature extraction
    and two separate output heads for chandas and source prediction.

    Architecture:
        Shared Backbone:
            Embedding → Conv1D(256) → MaxPool → Conv1D(128) → MaxPool
            → Conv1D(64) → GlobalMaxPool → Dropout

        Chandas Branch:
            Dense(128, relu) → Dropout → Dense(n_chandas, softmax)

        Source Branch:
            Dense(128, relu) → Dropout → Dense(n_source, softmax)

    Multi-task learning forces the shared backbone to learn features
    useful for BOTH tasks, improving generalization.

    Args:
        n_chandas: Number of chandas classes (8)
        n_source: Number of source/satakam classes

    Returns:
        Compiled Keras Model with two outputs
    """
    # Shared input
    input_layer = Input(shape=(config.MAX_SEQ_LEN,), name='text_input')

    # Shared embedding
    x = layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        input_length=config.MAX_SEQ_LEN,
        name='shared_embedding'
    )(input_layer)

    # Shared convolution backbone
    x = layers.Conv1D(config.CONV1_FILTERS, config.CONV1_KERNEL,
                      activation='relu', padding='same', name='shared_conv1')(x)
    x = layers.BatchNormalization(name='shared_bn1')(x)
    x = layers.MaxPooling1D(config.POOL_SIZE, name='shared_pool1')(x)

    x = layers.Conv1D(config.CONV2_FILTERS, config.CONV2_KERNEL,
                      activation='relu', padding='same', name='shared_conv2')(x)
    x = layers.BatchNormalization(name='shared_bn2')(x)
    x = layers.MaxPooling1D(config.POOL_SIZE, name='shared_pool2')(x)

    x = layers.Conv1D(config.CONV3_FILTERS, config.CONV3_KERNEL,
                      activation='relu', padding='same', name='shared_conv3')(x)
    x = layers.BatchNormalization(name='shared_bn3')(x)
    shared_features = layers.GlobalMaxPooling1D(name='shared_global_pool')(x)
    shared_features = layers.Dropout(config.DROPOUT_RATE, name='shared_dropout')(shared_features)

    # --- Chandas prediction branch ---
    chandas_x = layers.Dense(config.DENSE2_UNITS, activation='relu',
                             name='chandas_dense1')(shared_features)
    chandas_x = layers.Dropout(0.3, name='chandas_dropout')(chandas_x)
    chandas_output = layers.Dense(n_chandas, activation='softmax',
                                  dtype='float32', name='chandas_output')(chandas_x)

    # --- Source prediction branch ---
    source_x = layers.Dense(config.DENSE2_UNITS, activation='relu',
                            name='source_dense1')(shared_features)
    source_x = layers.Dropout(0.3, name='source_dropout')(source_x)
    source_output = layers.Dense(n_source, activation='softmax',
                                 dtype='float32', name='source_output')(source_x)

    # Build model
    model = Model(
        inputs=input_layer,
        outputs=[chandas_output, source_output],
        name=name
    )

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss={
            'chandas_output': 'categorical_crossentropy',
            'source_output': 'categorical_crossentropy'
        },
        loss_weights={
            'chandas_output': config.CHANDAS_LOSS_WEIGHT,
            'source_output': config.SOURCE_LOSS_WEIGHT
        },
        metrics={
            'chandas_output': ['accuracy'],
            'source_output': ['accuracy']
        }
    )

    return model


if __name__ == "__main__":
    configure_gpu()

    print("\n" + "=" * 60)
    print("SINGLE-TASK CNN (Chandas Prediction)")
    print("=" * 60)
    m1 = build_cnn_model(n_classes=8)
    m1.summary()

    print("\n" + "=" * 60)
    print("MULTI-TASK CNN (Chandas + Source)")
    print("=" * 60)
    m2 = build_multitask_cnn(n_chandas=8, n_source=28)
    m2.summary()
