"""
Model definitions.

    build_custom_cnn()  -> Model A: a small CNN trained from scratch (32x32 input)
    build_mobilenet()   -> Model B: MobileNetV2 transfer learning (added in Phase 5)
"""

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from src import config


def _conv_block(model, filters, use_batchnorm):
    """Add two 3x3 convolutions, then max pooling and dropout."""
    for _ in range(2):
        model.add(layers.Conv2D(filters, (3, 3), activation="relu"))
        if use_batchnorm:
            # BatchNorm keeps activations at a stable scale, which often speeds up training
            model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))   # halve width and height
    model.add(layers.Dropout(0.25))                     # randomly drop 25% to reduce overfitting


def build_custom_cnn(use_batchnorm=False):
    """
    Build and compile the custom CNN (Model A).

    Input (32, 32, 3)
    Conv(32) -> Conv(32) -> MaxPool -> Dropout(0.25)
    Conv(64) -> Conv(64) -> MaxPool -> Dropout(0.25)
    Flatten -> Dense(256) -> Dropout(0.5) -> Dense(43, softmax)
    """
    name = "custom_cnn_bn" if use_batchnorm else "custom_cnn"
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.NUM_CHANNELS)))

    # Feature extractor: learns edges, shapes, then sign parts
    _conv_block(model, 32, use_batchnorm)
    _conv_block(model, 64, use_batchnorm)

    # Classifier: turns the learned features into 43 class probabilities
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(config.NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
