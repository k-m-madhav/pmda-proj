# Model evaluation, metrics (IoU, pixel accuracy, confusion matrix)

# src/evaluate.py

"""
Comprehensive evaluation script for segmentation model.
Computes IoU, precision, recall, F1, and confusion matrix.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from src.config import (
    SUBSET_SIZE, VAL_RATIO,
    RANDOM_SEED, NUM_COMBINED_CLASSES,
    VOID_LABEL, DEVICE, FINAL_CLASS_NAMES
)
from src.dataset import RailSem19Dataset
from src.model import load_model
from src.train import create_stratified_split, compute_iou, compute_accuracy
from src.utils import get_logger

logger = get_logger("evaluate")

# CLASS_NAMES = {
#     0: "Track area",
#     1: "Scene context",
#     2: "Object"
# }

def compute_confusion_matrix(pred,
                              target,
                                num_classes=NUM_COMBINED_CLASSES,
                                  ignore_index=VOID_LABEL):
    """
    Compute confusion matrix for segmentation.
    
    Args:
        pred: Predicted labels [N]
        target: Ground truth labels [N]
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    # Flatten predictions and target
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Remove ignore index
    valid_mask = target_flat != ignore_index
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]

    # Compute confusion matrix
    cm = confusion_matrix(
        target_valid.cpu().numpy(),
        pred_valid.cpu().numpy(),
        labels=list(range(num_classes))
    )

    return cm

def compute_per_class_metrics(cm):
    """
    Compute precision, recall, F1 from confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
    
    Returns:
        Dictionary with metrics per class
    """
    num_classes = cm.shape[0]
    metrics = {}

    for i in range(num_classes):
        # True positives: diagonal element
        tp = cm[i, i]
        
        # False positives: sum of column i (excluding diagonal)
        fp = cm[:, i].sum() - tp
        
        # False negatives: sum of row i (excluding diagonal)
        fn = cm[i, :].sum() - tp
        
        # True negatives: all other predictions
        tn = cm.sum() - tp - fp - fn
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': cm[i, :].sum()  # Total ground truth instances
        }
    
    return metrics

def plot_confusion_matrix(cm,
                           class_names,
                             save_path=None,
                               normalize=False):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize by row (ground truth)
    """
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm) # handle div by 0
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
        data = cm_norm
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
        data = cm

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap='viridis',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Normalized' if normalize else 'Count'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()

def evaluate_model(model,
                    dataloader,
                      device):
    """
    Evaluate model on given dataloader.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()

    all_preds = []
    all_targets = []
    running_ious = {i: [] for i in range(NUM_COMBINED_CLASSES)}
    running_acc = 0.0

    logger.info('Running evaluation...')

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            pred = outputs.argmax(dim=1)

            # Metrics
            acc = compute_accuracy(pred, masks)
            ious = compute_iou(pred, masks)

            running_acc += acc
            for c in range(NUM_COMBINED_CLASSES):
                if not np.isnan(ious[c]):
                    running_ious[c].append(ious[c])

            # Collect predictions and targets for Confusion Matrix
            all_preds.append(pred.cpu())
            all_targets.append(masks.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute overall metrics
    mean_acc = running_acc / len(dataloader)

    class_ious = {
        c: np.mean(running_ious[c]) if running_ious[c] else 0.0
        for c in range(NUM_COMBINED_CLASSES)
    }

    mean_iou = np.mean([v for v in class_ious.values() if v > 0])

    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_targets)

    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(cm)

    results = {
        'accuracy': mean_acc,
        'mean_iou': mean_iou,
        'class_ious': class_ious,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics
    }

    return results

def print_evaluation_results(results,
                              split_name='Validation'):
    """
    Print formatted evaluation results.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"{split_name} Set Evaluation Results")
    logger.info("=" * 70)

    # Overall metrics
    logger.info("\n Overall Metrics:")
    logger.info(f"  Pixel Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"  Mean IoU:       {results['mean_iou']:.4f} ({results['mean_iou']*100:.2f}%)")

    # Per-class IoU
    logger.info("\n Per-Class IoU:")
    for class_id, iou in results['class_ious'].items():
        class_name = FINAL_CLASS_NAMES[class_id]
        logger.info(f"  {class_name:20s} (class {class_id}): {iou:.4f} ({iou*100:.2f}%)")

    # Per-class detailed metrics
    logger.info("\n Per-Class Detailed Metrics:")
    logger.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    logger.info("-" * 70)
    
    for class_id, metrics in results['per_class_metrics'].items():
        class_name = FINAL_CLASS_NAMES[class_id]
        logger.info(
            f"{class_name:<20} "
            f"{metrics['precision']:>11.4f} "
            f"{metrics['recall']:>11.4f} "
            f"{metrics['f1']:>11.4f} "
            f"{int(metrics['support']):>11d}"
        )
    
    # Confusion matrix
    logger.info("\n Confusion Matrix:")
    cm = results['confusion_matrix']

    # Header
    header = "True\\Pred    "
    for i in range(NUM_COMBINED_CLASSES):
        header += f"{FINAL_CLASS_NAMES[i][:12]:>14s}"
    logger.info(header)
    logger.info("-" * len(header))

    # Rows
    for i in range(NUM_COMBINED_CLASSES):
        row = f"{FINAL_CLASS_NAMES[i][:12]:<13}"
        for j in range(NUM_COMBINED_CLASSES):
            row += f"{cm[i, j]:>14d}"
        logger.info(row)

    # Analysis
    logger.info("\n Key Insights:")
    
    # Best performing class
    best_class = max(results['class_ious'].items(), key=lambda x: x[1])
    logger.info(f"  Best class: {FINAL_CLASS_NAMES[best_class[0]]} (IoU: {best_class[1]:.2%})")
    
    # Worst performing class
    worst_class = min(results['class_ious'].items(), key=lambda x: x[1])
    logger.info(f"  Needs improvement: {FINAL_CLASS_NAMES[worst_class[0]]} (IoU: {worst_class[1]:.2%})")

    # Main confusions
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    logger.info("\n  Common misclassifications:")
    for i in range(NUM_COMBINED_CLASSES):
        for j in range(NUM_COMBINED_CLASSES):
            if i != j and cm_norm[i, j] > 0.1:  # >10% confusion
                logger.info(
                    f"    {FINAL_CLASS_NAMES[i]} -> {FINAL_CLASS_NAMES[j]}: "
                    f"{cm_norm[i, j]:.1%} ({cm[i, j]} pixels)"
                )
    
    logger.info("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--subset_size', type=int, default=SUBSET_SIZE,
                       help='Subset size used during training')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                       help='Which split to evaluate')
    
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 70)
    
    # Device
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    model.eval()
    logger.info("Model loaded successfully!")

    # Load dataset and create splits
    logger.info("\nLoading dataset...")
    full_dataset = RailSem19Dataset()
    train_dataset, val_dataset = create_stratified_split(
        full_dataset,
        subset_size=args.subset_size,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED
    )
    
    # Create dataloaders    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Evaluate
    if args.split in ['val', 'both']:
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating on VALIDATION set")
        logger.info("=" * 70)
        
        val_results = evaluate_model(model, val_loader, device)
        print_evaluation_results(val_results, split_name="Validation")
        
        # Save confusion matrices
        plot_confusion_matrix(
            val_results['confusion_matrix'],
            [FINAL_CLASS_NAMES[i] for i in range(NUM_COMBINED_CLASSES)],
            save_path=os.path.join(output_dir, 'upd_confusion_matrix_val.png'),
            normalize=False
        )
        
        plot_confusion_matrix(
            val_results['confusion_matrix'],
            [FINAL_CLASS_NAMES[i] for i in range(NUM_COMBINED_CLASSES)],
            save_path=os.path.join(output_dir, 'upd_confusion_matrix_val_normalized.png'),
            normalize=True
        )

    if args.split in ['train', 'both']:
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating on TRAINING set")
        logger.info("=" * 70)
        
        train_results = evaluate_model(model, train_loader, device)
        print_evaluation_results(train_results, split_name="Training")
        
        # Save confusion matrices
        plot_confusion_matrix(
            train_results['confusion_matrix'],
            [FINAL_CLASS_NAMES[i] for i in range(NUM_COMBINED_CLASSES)],
            save_path=os.path.join(output_dir, 'upd_confusion_matrix_train.png'),
            normalize=False
        )
        
        plot_confusion_matrix(
            train_results['confusion_matrix'],
            [FINAL_CLASS_NAMES[i] for i in range(NUM_COMBINED_CLASSES)],
            save_path=os.path.join(output_dir, 'upd_confusion_matrix_train_normalized.png'),
            normalize=True
        )
    
    # Compare train vs val if both evaluated
    if args.split == 'both':
        logger.info("\n" + "=" * 70)
        logger.info("Train vs. Validation Comparison")
        logger.info("=" * 70)
        
        logger.info("\nAccuracy:")
        logger.info(f"  Train: {train_results['accuracy']:.4f} ({train_results['accuracy']*100:.2f}%)")
        logger.info(f"  Val:   {val_results['accuracy']:.4f} ({val_results['accuracy']*100:.2f}%)")
        
        logger.info("\nMean IoU:")
        logger.info(f"  Train: {train_results['mean_iou']:.4f} ({train_results['mean_iou']*100:.2f}%)")
        logger.info(f"  Val:   {val_results['mean_iou']:.4f} ({val_results['mean_iou']*100:.2f}%)")
        
        logger.info("\nPer-Class IoU Comparison:")
        for class_id in range(NUM_COMBINED_CLASSES):
            train_iou = train_results['class_ious'][class_id]
            val_iou = val_results['class_ious'][class_id]
            diff = train_iou - val_iou
            logger.info(
                f"  {FINAL_CLASS_NAMES[class_id]:20s}: "
                f"Train={train_iou:.4f}, Val={val_iou:.4f}, "
                f"Diff={diff:+.4f}"
            )

        # Overfitting check
        iou_gap = train_results['mean_iou'] - val_results['mean_iou']
        if iou_gap > 0.10:
            logger.warning(f"\n Large train-val gap ({iou_gap:.2%}) - possible overfitting!")
        elif iou_gap > 0.05:
            logger.info(f"\n Moderate train-val gap ({iou_gap:.2%}) - acceptable")
        else:
            logger.info(f"\n Small train-val gap ({iou_gap:.2%}) - good generalization")
    
    logger.info(f"\n Evaluation complete! Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
