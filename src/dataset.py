# src/dataset.py

"""
The goal of this script is to define a PyTorch Dataset class that:
    Loads images and preprocessed masks from disk
    Applies DINOv2 preprocessing to images
    Resizes masks to match DINOv2's patch grid
    Returns PyTorch tensors ready for training
"""
import os
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset

from src.config import (
    IMG_DIR, PROCESSED_MASK_DIR,
    PRETRAINED_MODEL_NAME,
    INPUT_SIZE, PATCH_GRID_SIZE
)

from src.utils import get_logger

logger = get_logger("dataset")

class RailSem19Dataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing RailSem19 data.
    
    Responsibilities:
    1. Load RGB images from disk
    2. Load preprocessed 6-class masks from disk
    3. Apply DINOv2 image preprocessing (resize, normalize)
    4. Resize masks to 37x37 (patch grid)
    5. Return (image_tensor, mask_tensor) pairs
    
    Used by:
    - DataLoader in train.py
    - Evaluation in evaluate.py
    """

    def __init__(self,
                  img_dir=IMG_DIR,
                    mask_dir=PROCESSED_MASK_DIR,
                      model_name=PRETRAINED_MODEL_NAME,
                        input_size=INPUT_SIZE,
                            mask_size=PATCH_GRID_SIZE):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.mask_size = mask_size
        
        # Load image processor with explicit size
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, use_fast=True
        )
        logger.info(f"Loaded processor from {model_name}")

        # Get files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        assert len(self.img_files) == len(self.mask_files), \
            f"Mismatch: {len(self.img_files)} images vs {len(self.mask_files)} masks"
        
        logger.info(f"Initialized dataset with {len(self.img_files)} samples")

    def __len__(self):
        """Tells Pytorch how many samples are in your dataset"""
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Defines how to load one sample given its index

        Returns:
            image: Tensor of shape (3, 518, 518) - preprocessed by DINOv2 processor; (color_channels, h, w)
            mask: Tensor of shape (37, 37) with values in [0, 1, 2, 3, 4, 5, 255]; 
                  518 / 14 = 37, where 14 is patch_size
        """

        # Load image (1920x1080)
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path)

        # Load preprocessed mask (only 6 classes)
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        msk = np.array(Image.open(mask_path), dtype=np.int64)

        # Preprocess image with DINOv2 processor
        inputs = self.image_processor(
            images=img,
            return_tensors='pt',
            size={'height': self.input_size, 'width': self.input_size}
        )
        image_tensor = inputs['pixel_values'].squeeze(0) # removes the batch size value: [3, 518, 518]

        # Resize mask to match patch grid (37x37) since segmentation happens at patch-level (37×37),
        # not pixel-level (1920×1080), which is why the mask needs to match that resolution.
        # Image.Resampling.NEAREST preserves discrete labels i.e. values are still
        # [0, 1, 2, 3, 4, 5, 255] even after resizing the mask
        mask_resized = Image.fromarray(msk.astype(np.uint8))
        mask_resized = mask_resized.resize((self.mask_size, self.mask_size), Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized, dtype=np.int64)).long()

        return image_tensor, mask_tensor

# Test function
if __name__ == "__main__":
    logger.info("Testing RailSem19Dataset with 518x518 input...")
  
    # Create dataset
    dataset = RailSem19Dataset()
    logger.info(f"Dataset size: {len(dataset)}")
   
    # Load one sample
    image_tsr, mask_tsr = dataset[0] # internally calls dataset.__getitem__(0)
    logger.info(f"\n=== Sample Data ===")
    logger.info(f"Image shape: {image_tsr.shape}")  # Should be [3, 518, 518]
    logger.info(f"Mask shape: {mask_tsr.shape}")    # Should be [37, 37]
    logger.info(f"Mask unique values: {torch.unique(mask_tsr)}")
    logger.info(f"Image value range (min/max): {image_tsr.min():.3f} / {image_tsr.max():.3f}")

    # --- Verify with actual DINOv2 model ---
    logger.info("\n=== Loading DINOv2 for verification ===")
    model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME).eval()

    logger.info("Running forward pass...")
    with torch.no_grad():
        outputs = model(image_tsr.unsqueeze(0)) # Add batch dimension
    
    # Get features
    features = outputs.last_hidden_state # [1, num_tokens, feature_dim]

    logger.info(f"\n=== DINOv2 Output ===")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"  [batch, tokens, feature_dim]")
    logger.info(f"  [{features.shape[0]}, {features.shape[1]}, {features.shape[2]}]")

    # Verification
    expected_patches = (518 // 14) ** 2
    expected_tokens = expected_patches + 1 # +1 for CLS token
    actual_tokens = features.shape[1]
    feature_spatial = int(np.sqrt(actual_tokens - 1))
    
    logger.info(f"\n=== Verification ===")
    checks = [
        ("Expected patches", expected_patches, feature_spatial*feature_spatial),
        ("Expected tokens", expected_tokens, actual_tokens),
        ("Feature dimension", 768, features.shape[2]),
        ("Mask spatial size", feature_spatial, mask_tsr.shape[0]),
    ]
    
    all_passed = True
    for name, expected, actual in checks:
        passed = (expected == actual)
        symbol = "[OK]" if passed else "[FAIL]"
        logger.info(f"{symbol} {name}: expected={expected}, actual={actual}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n All verifications passed! Dataset is ready for training.")
    else:
        logger.error("\n Some verifications failed. Check configuration.")
    
    logger.info("\nDataset test complete!")
