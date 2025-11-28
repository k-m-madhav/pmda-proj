# Overlay predicted masks, ground truth vs. prediction plots, pretty class color maps

# src/visualize.py

"""
Visualization script for segmentation predictions.
Displays original image, ground truth mask, and predicted mask side-by-side.
"""

import random
import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import (
    DEVICE, FINAL_CLASS_NAMES,
    CLASS_COLORS_NORMALIZED
)
from src.model import load_model
from src.dataset import RailSem19Dataset
from src.utils import get_logger

logger = get_logger("visualize")

def create_colored_mask(mask,
                         colors=CLASS_COLORS_NORMALIZED):
    """
    Convert class mask to RGB visualization.
    
    Args:
        mask: numpy array of shape (H, W) with class indices
        colors: dict mapping class_id to RGB color
    
    Returns:
        RGB image of shape (H, W, 3)
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    for class_id, color in colors.items():
        colored[mask == class_id] = color

    return colored

def visualize_prediction(image,
                          gt_mask,
                            pred_mask,
                              save_path=None,
                                show=True):
    """
    Create side-by-side visualization of original image, ground truth, and prediction.
    
    Args:
        image: PIL Image or numpy array (H, W, 3)
        gt_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        save_path: Path to save figure (optional)
        show: Whether to display figure
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Create colored masks
    gt_colored = create_colored_mask(gt_mask)
    pred_colored = create_colored_mask(pred_mask)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Original image
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add legend
    legend_elements = [
        Patch(facecolor=CLASS_COLORS_NORMALIZED[0], label=FINAL_CLASS_NAMES[0]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[1], label=FINAL_CLASS_NAMES[1]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[2], label=FINAL_CLASS_NAMES[2]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[3], label=FINAL_CLASS_NAMES[3]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[4], label=FINAL_CLASS_NAMES[4]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[5], label=FINAL_CLASS_NAMES[5]),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=12, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_overlay(image,
                       pred_mask,
                         alpha=0.4,
                           save_path=None,
                             show=True):
    """
    Create overlay visualization with prediction mask on original image.
    
    Args:
        image: PIL Image or numpy array (H, W, 3)
        pred_mask: Predicted mask (H, W)
        alpha: Transparency (0=invisible, 1=opaque)
        save_path: Path to save figure
        show: Whether to display figure
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Create colored mask
    pred_colored = create_colored_mask(pred_mask)
    
    # Blend image and mask
    overlay = (1 - alpha) * image + alpha * pred_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(overlay)
    ax.set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        Patch(facecolor=CLASS_COLORS_NORMALIZED[0], label=FINAL_CLASS_NAMES[0]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[1], label=FINAL_CLASS_NAMES[1]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[2], label=FINAL_CLASS_NAMES[2]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[3], label=FINAL_CLASS_NAMES[3]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[4], label=FINAL_CLASS_NAMES[4]),
        Patch(facecolor=CLASS_COLORS_NORMALIZED[5], label=FINAL_CLASS_NAMES[5]),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=12, frameon=True, fancybox=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved overlay to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def predict_from_dataset(model,
                          dataset,
                            index,
                              device=DEVICE):
    """
    Get prediction for a specific dataset sample.
    
    Args:
        model: Trained segmentation model
        dataset: RailSem19Dataset instance
        index: Sample index
        device: Device to run inference on
    
    Returns:
        image (PIL), gt_mask (numpy), pred_mask (numpy)
    """
    model.eval()

    # Get sample from dataset
    image_tensor, gt_mask = dataset[index]

    # Load original image for visualization
    img_path = os.path.join(dataset.img_dir, dataset.img_files[index])
    original_image = Image.open(img_path).convert('RGB')

    # Get prediction
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device) # Add batch dim
        outputs = model(image_batch)
        pred_mask = outputs.argmax(dim=1)[0].cpu().numpy() # [37, 37]

    # Upsample masks to original image size
    gt_mask_np = gt_mask.numpy()

    # Resize to original image size for better visualization
    orig_w, orig_h = original_image.size

    pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_full = pred_mask_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
    pred_mask_full = np.array(pred_mask_full)

    gt_mask_img = Image.fromarray(gt_mask_np.astype(np.uint8))
    gt_mask_full = gt_mask_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
    gt_mask_full = np.array(gt_mask_full)

    return original_image, gt_mask_full, pred_mask_full

def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Display visualizations interactively')
    parser.add_argument('--overlay', action='store_true',
                       help='Also create overlay visualizations')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                       help='Specific sample indices to visualize (e.g., --indices 0 5 10)')
    
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Segmentation Visualization")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    device = torch.device(DEVICE)
    rail_seg_model = load_model(args.checkpoint, device=device)
    rail_seg_model.eval()
    logger.info("Model loaded successfully!")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = RailSem19Dataset()
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Determine which samples to visualize
    if args.indices:
        indices = args.indices
        logger.info(f"Visualizing specific indices: {indices}")
    else:
        # Random samples
        random.seed(42)
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        logger.info(f"Visualizing {len(indices)} random samples")
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        logger.info(f"\nProcessing sample {i+1}/{len(indices)} (index {idx})...")

        try:
            # Get prediction
            image, gt_mask, pred_mask = predict_from_dataset(rail_seg_model, dataset, idx, device)

            # Compute accuracy for this sample
            valid_mask = gt_mask != 255
            correct = ((pred_mask == gt_mask) & valid_mask).sum()
            total = valid_mask.sum()
            accuracy = correct / total if total > 0 else 0

            logger.info(f"  Sample accuracy: {accuracy:.2%}")
            logger.info(f"  Unique GT classes: {np.unique(gt_mask[gt_mask != 255])}")
            logger.info(f"  Unique pred classes: {np.unique(pred_mask)}")

            # Save side-by-side visualization
            save_path = os.path.join(output_dir, f"upd_sample_{idx:04d}_comparison.png")
            visualize_prediction(image, gt_mask, pred_mask,
                               save_path=save_path, show=args.show)

            # Save overlay if requested
            if args.overlay:
                overlay_path = os.path.join(output_dir, f"upd_sample_{idx:04d}_overlay.png")
                visualize_overlay(image, pred_mask, alpha=0.5,
                                save_path=overlay_path, show=args.show)
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue
   
    logger.info("\n" + "=" * 60)
    logger.info("Visualization Complete!")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
