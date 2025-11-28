# Training loop, loss, dataloading, checkpoint save/load

# src/train.py

"""
Training script for DINOv2 rail track segmentation model.
Includes stratified split, class weighting, IoU metrics, and checkpointing.
"""

import random
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    SUBSET_SIZE, VAL_RATIO,
    RANDOM_SEED, NUM_EPOCHS,
    BATCH_SIZE, LEARNING_RATE,
    NUM_WORKERS, CLASS_WEIGHTS,
    VOID_LABEL, LOG_INTERVAL,
    SAVE_INTERVAL, CHECKPOINT_DIR,
    DEVICE, NUM_COMBINED_CLASSES
)

from src.dataset import RailSem19Dataset
from src.model import RailSegmentationModel
from src.utils import get_logger

logger = get_logger("train")

def compute_iou(pred,
                 target,
                   num_classes=NUM_COMBINED_CLASSES,
                     ignore_index=VOID_LABEL):   
    """
    Compute IoU for each class.
    
    Args:
        pred: Predicted class indices [B, H, W]
        target: Ground truth class indices [B, H, W]
        num_classes: Number of combined classes (6)
        ignore_index: Index to ignore (255)
    
    Returns:
        dict: IoU per class {0: iou_0, 1: iou_1, ..., 5: iou_5}
    """
    ious = {}

    for class_id in range(num_classes):
        # Binary masks for this class
        pred_mask = pred == class_id
        target_mask = target == class_id
        valid_mask = target != ignore_index

        # Apply valid mask
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        # Intersection and Union
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou = float('nan') # class not present
        else:
            iou = (intersection / union).item() # convert from tensor to float

        ious[class_id] = iou

    return ious

def compute_accuracy(pred,
                      target,
                        ignore_index=VOID_LABEL):
    """
    Compute pixel accuracy ignoring VOID pixels.
    """
    valid_mask = target != ignore_index
    correct = ((pred == target) & valid_mask).sum()
    total = valid_mask.sum()

    if total == 0:
        return 0.0
    
    return (correct.float() / total.float()).item()

def create_stratified_split(dataset,
                             subset_size=None,
                               val_ratio=VAL_RATIO,
                                 random_seed=RANDOM_SEED):
    """
    Create stratified train/val split based on object presence.
    
    Args:
        dataset: Full dataset
        subset_size: Use random N samples (for faster training)
        val_ratio: Validation set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset
    """
    logger.info("Creating stratified train/val split...")

    # Use subset if specified
    if subset_size and subset_size < len(dataset):
        random.seed(random_seed)
        indices = random.sample(range(len(dataset)), subset_size)
        logger.info(f"Using random subset of {subset_size} samples out of {len(dataset)}")
    else:
        indices = list(range(len(dataset)))
        logger.info(f"Using full dataset: {len(dataset)} samples")

    # Check which images have the 'Object' class (class 3)
    logger.info("Analyzing objects presence for stratification...")
    has_objects = []
    for i in tqdm(indices, desc='Checking object presence'):
        _, mask = dataset[i]
        has_objects.append((mask == 3).any().item())

    object_count = sum(has_objects)
    logger.info(f"Images with objects: {object_count}/{len(indices)} ({object_count/len(indices):.1%})")

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=has_objects,
        random_state=random_seed
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def train_one_epoch(model,
                     dataloader,
                       criterion,
                         optimizer,
                           device,
                             epoch):
    """Train for one epoch"""
    model.train()
    model.backbone.eval() # Keep backbone in eval mode

    running_loss = 0.0
    running_acc = 0.0
    running_ious = {i: [] for i in range(NUM_COMBINED_CLASSES)}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pred = outputs.argmax(dim=1)
        acc = compute_accuracy(pred, masks)
        ious = compute_iou(pred, masks)

        running_loss += loss.item()
        running_acc += acc
        for c in range(NUM_COMBINED_CLASSES):
            if not torch.isnan(torch.tensor(ious[c])):
                running_ious[c].append(ious[c])

        # Update progress bar
        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })

    # Epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_ious = {
        c: sum(running_ious[c]) / len(running_ious[c]) if running_ious[c] else 0.0
        for c in range(NUM_COMBINED_CLASSES)
    }

    return epoch_loss, epoch_acc, epoch_ious

def validate(model,
              dataloader,
                criterion,
                  device):
    """Validate the model"""
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    running_ious = {i: [] for i in range(NUM_COMBINED_CLASSES)}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            pred = outputs.argmax(dim=1)
            acc = compute_accuracy(pred, masks)
            ious = compute_iou(pred, masks)

            running_loss += loss.item()
            running_acc += acc
            for c in range(NUM_COMBINED_CLASSES):
                if not torch.isnan(torch.tensor(ious[c])):
                    running_ious[c].append(ious[c])

    val_loss = running_loss / len(dataloader)
    val_acc = running_acc / len(dataloader)
    val_ious = {
        c: sum(running_ious[c]) / len(running_ious[c]) if running_ious[c] else 0.0
        for c in range(NUM_COMBINED_CLASSES)
    }

    return val_loss, val_acc, val_ious

def save_checkpoint(model,
                     optimizer,
                       epoch,
                         val_loss,
                           val_ious,
                             filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_ious': val_ious,
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

def main(resume_from_checkpoint_path=None):
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Starting Rail Track Segmentation Training")
    logger.info("=" * 60)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # Dataset
    logger.info("\n=== Loading Dataset ===")
    full_dataset = RailSem19Dataset()
    train_dataset, val_dataset = create_stratified_split(
        full_dataset, subset_size=SUBSET_SIZE
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Model
    logger.info("\n=== Initializing Model ===")
    rail_seg_model = RailSegmentationModel(freeze_backbone=True)
    rail_seg_model = rail_seg_model.to(device)

    # Loss functions with class weights
    class_weights = torch.tensor(CLASS_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=VOID_LABEL)
    logger.info(f"Class weights: {class_weights}")

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, rail_seg_model.parameters()),
        lr=LEARNING_RATE
    )
    logger.info(f"Optimizer: Adam, LR = {LEARNING_RATE}")

    # Training loop
    logger.info("\n=== Starting Training ===")
    logger.info(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")

    best_val_loss = float('inf')
    best_mean_iou = 0.0
    training_history = []

    start_time = time.time()

    start_epoch = 0
    if resume_from_checkpoint_path:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location=device)
        rail_seg_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, train_ious = train_one_epoch(
            rail_seg_model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, val_ious = validate(
            rail_seg_model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        # Compute mean IoU (excluding NaN)
        train_mean_iou = sum(v for v in train_ious.values() if v > 0) / sum(1 for v in train_ious.values() if v > 0)
        val_mean_iou = sum(v for v in val_ious.values() if v > 0) / sum(1 for v in val_ious.values() if v > 0)

        # Logging
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.1f}s")
        logger.info(f"{'='*60}")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
        logger.info(f"Train mIoU: {train_mean_iou:.4f} | Val mIoU: {val_mean_iou:.4f}")
        logger.info("\nPer-Class IoU:")
        logger.info(f"  Built env (0):   Train={train_ious[0]:.4f} | Val={val_ious[0]:.4f}")
        logger.info(f"  Sky (1): Train={train_ious[1]:.4f} | Val={val_ious[1]:.4f}")
        logger.info(f"  Vegetation (2): Train={train_ious[2]:.4f} | Val={val_ious[2]:.4f}")
        logger.info(f"  Object (3): Train={train_ious[3]:.4f} | Val={val_ious[3]:.4f}")
        logger.info(f"  Sign (4): Train={train_ious[4]:.4f} | Val={val_ious[4]:.4f}")
        logger.info(f"  Track rail (5): Train={train_ious[5]:.4f} | Val={val_ious[5]:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_mean_iou': train_mean_iou,
            'val_mean_iou': val_mean_iou,
            'train_ious': train_ious,
            'val_ious': val_ious
        })

        # Save best model
        if val_mean_iou > best_mean_iou:
            best_mean_iou = val_mean_iou
            best_val_loss = val_loss
            save_checkpoint(
                rail_seg_model, optimizer, epoch, val_loss, val_ious,
                os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )
            logger.info(f"New best model! Val mIoU: {best_mean_iou:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(
                rail_seg_model, optimizer, epoch, val_loss, val_ious,
                os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            )
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best validation mIoU: {best_mean_iou:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    save_checkpoint(
        rail_seg_model, optimizer, NUM_EPOCHS-1, val_loss, val_ious,
        os.path.join(CHECKPOINT_DIR, "final_model.pth")
    )

    logger.info(f"\nCheckpoints saved in: {CHECKPOINT_DIR}")
    # logger.info("Best model: best_model.pth")
    # logger.info("Final model: final_model.pth")

if __name__ == "__main__":
    main()
