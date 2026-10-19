"""
Central configuration for SignSense.

All paths and hyperparameters live here, so the rest of the code
has no "magic numbers" and no hardcoded absolute paths.
"""

from pathlib import Path

# ---------------------------------------------------------------
# Paths (all relative to the project root, so the code runs anywhere)
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # project root folder
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "Train"                      # Train/<class_id>/*.png
TEST_DIR = DATA_DIR / "Test"                        # Test/*.png
TEST_CSV = DATA_DIR / "Test.csv"                    # labels for test images
META_DIR = DATA_DIR / "Meta"                        # reference image of each sign
SIGNNAMES_CSV = DATA_DIR / "signnames.csv"          # ClassId -> SignName
MODELS_DIR = BASE_DIR / "models"                    # saved .keras models
RESULTS_DIR = BASE_DIR / "results"                  # plots, tables, reports

# Saved model files
CUSTOM_CNN_PATH = MODELS_DIR / "custom_cnn.keras"         # baseline from the brief (98.05%)
CUSTOM_CNN_BN_PATH = MODELS_DIR / "custom_cnn_bn.keras"   # + BatchNorm (98.60%)
MOBILENET_PATH = MODELS_DIR / "mobilenetv2.keras"

# "Model A" used in the demo app and the final comparison: the BatchNorm CNN
MODEL_A_PATH = CUSTOM_CNN_BN_PATH

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
NUM_CLASSES = 43            # GTSRB has 43 sign classes (0 to 42)
KAGGLE_DATASET = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

# ---------------------------------------------------------------
# Image sizes
# ---------------------------------------------------------------
IMG_SIZE = 32               # custom CNN input: 32x32 pixels
MOBILENET_IMG_SIZE = 96     # MobileNetV2 needs larger images
NUM_CHANNELS = 3            # RGB colour images

# ---------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------
BATCH_SIZE = 64             # images per training step
EPOCHS = 20                 # max epochs for the custom CNN
MOBILENET_EPOCHS = 10       # epochs for training the MobileNetV2 head
# Fine-tuning settings. Run 1 used 20 layers / 1e-5 / 5 epochs and reached only
# 66.75% test accuracy, so run 2 fine-tunes more layers, faster and for longer.
FINE_TUNE_EPOCHS = 15       # extra epochs when fine-tuning MobileNetV2
FINE_TUNE_LAYERS = 60       # how many top MobileNetV2 layers to unfreeze (of ~154)
LEARNING_RATE = 1e-3        # Adam learning rate
FINE_TUNE_LR = 1e-4         # smaller learning rate for fine-tuning
VAL_SPLIT = 0.2             # 20% of training data used for validation
RANDOM_SEED = 42            # fixed seed for reproducible results
SHUFFLE_BUFFER = 5000       # images kept in the tf.data shuffle buffer (MobileNetV2)

# Callbacks
EARLY_STOP_PATIENCE = 5     # stop if val_loss does not improve for 5 epochs
REDUCE_LR_PATIENCE = 3      # halve the learning rate after 3 flat epochs
REDUCE_LR_FACTOR = 0.5

# ---------------------------------------------------------------
# Data augmentation (NO flips: flipping changes a sign's meaning,
# e.g. "turn left" would become "turn right")
# ---------------------------------------------------------------
ROTATION_RANGE = 10         # degrees
ZOOM_RANGE = 0.15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1

# ---------------------------------------------------------------
# Prediction / demo app
# ---------------------------------------------------------------
LOW_CONFIDENCE = 0.5        # below 50% the app shows a "not sure" warning
TOP_K = 3                   # number of top predictions to display
