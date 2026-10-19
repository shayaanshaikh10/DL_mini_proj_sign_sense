"""
Model definitions.

    build_custom_cnn()  -> Model A: a small CNN trained from scratch (32x32 input)
    build_mobilenet()   -> Model B: MobileNetV2 transfer learning (added in Phase 5)
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
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

    _compile(model, config.LEARNING_RATE)
    return model


def _compile(model, learning_rate):
    """Compile a model with Adam, categorical cross-entropy and accuracy."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_mobilenet():
    """
    Build Model B: MobileNetV2 pre-trained on ImageNet, with a new classifier head.

    Input (96, 96, 3) -> MobileNetV2 base (frozen)
    -> GlobalAveragePooling2D -> Dropout(0.3) -> Dense(43, softmax)

    Returns (model, base_model). base_model is needed later for fine-tuning.
    """
    size = config.MOBILENET_IMG_SIZE
    base_model = MobileNetV2(weights="imagenet", include_top=False,
                             input_shape=(size, size, config.NUM_CHANNELS))
    base_model.trainable = False   # freeze: keep the ImageNet features unchanged at first

    inputs = layers.Input(shape=(size, size, config.NUM_CHANNELS))
    # training=False keeps the base's BatchNorm layers in inference mode,
    # which matters when we unfreeze some layers for fine-tuning later
    features = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(features)   # average each feature map to one number
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2")
    _compile(model, config.LEARNING_RATE)
    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=config.FINE_TUNE_LAYERS):
    """Unfreeze the last `num_layers` layers of the base and recompile with a small learning rate."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    # A tiny learning rate nudges the pre-trained weights instead of destroying them
    _compile(model, config.FINE_TUNE_LR)
