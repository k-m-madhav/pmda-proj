# All settings and mappings:
# new category mapping dict, relevant hyperparameters, input/output directories

"""
Project configuration: category mappings, directories, training parameters, etc.
"""

# src/config.py
import torch

# =========== MODEL SETTINGS ===========
PRETRAINED_MODEL_NAME = "facebook/dinov2-base" # dinov3-vit7b16-pretrain-lvd1689m

# =========== CATEGORY MAPPINGS ===========

# Original class indices (as per order in rs19-config.json)
# You should generate this from your dataframe for your actual dataset
ORIGINAL_CLASS_NAMES = [
    "road", "sidewalk", "construction", "tram-track", "fence", "pole",
    "traffic-light", "traffic-sign", "vegetation", "terrain", "sky", "human",
    "rail-track", "car", "truck", "trackbed", "on-rails", "rail-raised", "rail-embedded"
]

# Final three categories (index: class name)
COMBINED_LABELS = {
    0: "Track area",
    1: "Scene context",
    2: "Object"
}

VOID_LABEL = 255  # Mask "ignore" value

# Map from original idx to combined category idx
ORIG_TO_COMBINED = {
    0: 1,    # road -> Scene context
    1: 1,    # sidewalk -> Scene context
    2: 1,    # construction -> Scene context
    3: 0,    # tram-track -> Track area
    4: 1,    # fence -> Scene context
    5: 1,    # pole -> Scene context
    6: 1,    # traffic-light -> Scene context
    7: 1,    # traffic-sign -> Scene context
    8: 1,    # vegetation -> Scene context
    9: 1,    # terrain -> Scene context
    10: 1,   # sky -> Scene context
    11: 2,   # human -> Object
    12: 0,   # rail-track -> Track area
    13: 2,   # car -> Object
    14: 2,   # truck -> Object
    15: 0,   # trackbed -> Track area
    16: 0,   # on-rails -> Track area
    17: 0,   # rail-raised -> Track area
    18: 0,   # rail-embedded -> Track area
    # Others (e.g. 255) can map to VOID_LABEL or be ignored
}

COMBINED_COLORS = {
    0: [255, 0, 0], # Track area - Red
    1: [0, 255, 0], # Scene context - Green
    2: [0, 0, 255], # Object - Blue
    255: [0, 0, 0], # Void - Black
}

# =========== DEFAULT DIRECTORIES ===========

DATA_DIR = "data/railsem19"
IMG_DIR = f"{DATA_DIR}/jpgs/rs19_val"
MASK_DIR = f"{DATA_DIR}/uint8/rs19_val"

PROCESSED_MASK_DIR = f"{DATA_DIR}/processed_masks"  # For remapped masks

# =========== TRAINING SETTINGS ===========

# Dataset
SUBSET_SIZE = 100  # Start small for CPU training
VAL_RATIO = 0.2

# Training
NUM_EPOCHS = 10  # Reasonable for initial training
BATCH_SIZE = 4   # Smaller for CPU
LEARNING_RATE = 1e-3

# Class imbalance (from 1000-sample analysis)
CLASS_WEIGHTS = [1.0, 1.0, 2.72]

# Monitoring
LOG_INTERVAL = 5      # Log every 10 batches
SAVE_INTERVAL = 5      # Save checkpoint every 5 epochs
EARLY_STOPPING = True
PATIENCE = 10          # Stop if no improvement for 10 epochs

# Paths
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# =========== HYPERPARAMETERS ===========

RANDOM_SEED = 42
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
NUM_COMBINED_CLASSES = len(COMBINED_LABELS)
INPUT_SIZE = 518
PATCH_GRID_SIZE = 37
FEATURE_DIM = 768

# =========== OTHER SETTINGS ===========

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
