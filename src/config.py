# src/config.py

"""
Project configuration: category mappings, directories, training parameters, etc.
"""

import numpy as np
import torch

# =========== MODEL SETTINGS ===========
PRETRAINED_MODEL_NAME = "facebook/dinov2-base" # dinov3-vit7b16-pretrain-lvd1689m

# =========== CATEGORY MAPPINGS ===========

# Original class indices (as per order in rs19-config.json)
ORIGINAL_CLASS_NAMES = [
    "road", "sidewalk", "construction", "tram-track", "fence", "pole",
    "traffic-light", "traffic-sign", "vegetation", "terrain", "sky", "human",
    "rail-track", "car", "truck", "trackbed", "on-rails", "rail-raised", "rail-embedded"
]

# Final 6 categories (index: class name)
FINAL_CLASS_NAMES = {
    0: "Built env",
    1: "Sky",
    2: "Vegetation",
    3: "Object",
    4: "Sign",
    5: "Track Rail",
    255: "Void"
}

VOID_LABEL = 255  # Mask "ignore" value

# Map from original idx to combined category idx
ORIG_TO_COMBINED = {
    0: 0,    # road -> Built env
    1: 0,    # sidewalk -> Built env
    2: 0,    # construction -> Built env
    3: 5,    # tram-track -> Track area
    4: 0,    # fence -> Built env
    5: 4,    # pole -> Sign
    6: 4,    # traffic-light -> Sign
    7: 4,    # traffic-sign -> Sign
    8: 2,    # vegetation -> Vegetation
    9: 0,    # terrain -> Built env
    10: 1,   # sky -> Sky
    11: 3,   # human -> Object
    12: 5,   # rail-track -> Track Rail
    13: 3,   # car -> Object
    14: 3,   # truck -> Object
    15: 5,   # trackbed -> Track Rail
    16: 5,   # on-rails -> Track Rail
    17: 5,   # rail-raised -> Track Rail
    18: 5,   # rail-embedded -> Track Rail
    # Others (e.g. 255) can map to VOID_LABEL or be ignored
}

# RGB format (0-255) for Streamlit HTML/CSS
CLASS_COLORS_RGB = {
    0: [102, 51, 0], # Built env - Brown
    1: [0, 0, 255], # Sky - Blue
    2: [0, 255, 0], # Vegetation - Green
    3: [127, 0, 255], # Object - Purple
    4: [255, 255, 0], # Sign - Yellow
    5: [255, 0, 0], # Track Rail - Red
    255: [0, 0, 0], # Void - Black
}

# Normalized format (0-1) for matplotlib/visualization
CLASS_COLORS_NORMALIZED = {
    k: np.array(v) / 255.0
    for k, v in CLASS_COLORS_RGB.items()
}

# =========== DEFAULT DIRECTORIES ===========

DATA_DIR = "data/railsem19"
IMG_DIR = f"{DATA_DIR}/jpgs/rs19_val"
MASK_DIR = f"{DATA_DIR}/uint8/rs19_val"

PROCESSED_MASK_DIR = f"{DATA_DIR}/processed_masks"  # For remapped masks

# =========== TRAINING SETTINGS ===========

# Dataset
SUBSET_SIZE = 100      # Start small for CPU training
VAL_RATIO = 0.2

# Training
NUM_EPOCHS = 10        # Reasonable for initial training
BATCH_SIZE = 4         # Smaller for CPU
LEARNING_RATE = 1e-3

# Class imbalance (from 1000-sample analysis). Object class is getting a higher weight
CLASS_WEIGHTS = [
    1.0,    # Built env
    1.0,    # Sky
    1.0,    # Vegetation
    5.0,    # Object (underrepresented)
    2.0,    # Sign
    1.0     # Track Rail
]

# Monitoring
LOG_INTERVAL = 5       # Log every 10 batches
SAVE_INTERVAL = 5      # Save checkpoint every 5 epochs
EARLY_STOPPING = True
PATIENCE = 10          # Stop if no improvement for 10 epochs

# Paths
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
BEST_MODEL_PATH = "checkpoints/best_model.pth"

# =========== HYPERPARAMETERS ===========

RANDOM_SEED = 42
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
NUM_COMBINED_CLASSES = len(FINAL_CLASS_NAMES) - 1
INPUT_SIZE = 518
PATCH_GRID_SIZE = 37
FEATURE_DIM = 768

# =========== OTHER SETTINGS ===========

CONFIDENCE_THRESHOLD_DEFAULT = 0.5
OVERLAY_ALPHA_DEFAULT = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
