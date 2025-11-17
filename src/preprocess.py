# Script to remap all original masks to your 3-category version

# src/preprocess.py

"""
Script to remap original mask classes to combined categories and save processed masks.
Uses mappings from config.py
Only need to run this script once!
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.config import MASK_DIR, PROCESSED_MASK_DIR, ORIG_TO_COMBINED, VOID_LABEL
from src.utils import get_logger

logger = get_logger("preprocess")

def remap_mask(mask_arr):
    """
    Remaps original class indices to combined categories. Preserves VOID_LABEL.
    """
    # Create a copy filled with VOID_Label
    remapped = np.full_like(mask_arr, fill_value=VOID_LABEL)

    # Find positions where mask_arr equals each original index
    # and assign the corresponding new index
    # This uses broadcasting instead of a Python loop
    for orig_idx, new_idx in ORIG_TO_COMBINED.items():
        remapped[mask_arr == orig_idx] = new_idx

    # Safety net to ensure void pixels remain untouched
    remapped[mask_arr == VOID_LABEL] = VOID_LABEL
   
    return remapped

def main():
    os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    logger.info(f"Found {len(mask_files)} mask files in {MASK_DIR}")

    for mask_file in tqdm(mask_files, desc="Mask remapping"): # tested using the first 5 masks; mask_files[:5]
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask_arr = np.array(Image.open(mask_path))
        remapped = remap_mask(mask_arr)

        # Save processed mask (PNG, uint8)
        out_path = os.path.join(PROCESSED_MASK_DIR, mask_file)
        Image.fromarray(remapped.astype(np.uint8)).save(out_path)
        logger.debug(f"Remapped and saved {mask_file} → {out_path}")

    logger.info(f"Finished processing {len(mask_files)} masks.")
    logger.info(f"Processed masks saved to {PROCESSED_MASK_DIR}")

if __name__ == "__main__":
    main()
