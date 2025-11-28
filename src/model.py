# Functions/classes to load DINOv2/DINOv3, build your segmentation head, load from checkpoint

# src/model.py

"""
DINOv2-based segmentation model for rail track segmentation.
Combines frozen DINOv2 backbone with trainable segmentation head.
"""

import torch
from torch import nn
from transformers import AutoModel

from src.config import (
    PRETRAINED_MODEL_NAME,
    PATCH_GRID_SIZE, NUM_COMBINED_CLASSES,
    FEATURE_DIM, DEVICE
)

from src.utils import get_logger

logger = get_logger("model")

class SegmentationHead(nn.Module):
    """
    Lightweight segmentation head for DINOv2 features.
    Maps 768-dim features to 6-class predictions per patch.
    """

    def __init__(self,
                  in_channels=FEATURE_DIM,
                    num_classes=NUM_COMBINED_CLASSES,
                      hidden_dim=256):
        super().__init__()

        self.head = nn.Sequential(
            # First Conv: Reduce Dimensions
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # Second Conv: Refine Features
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # Final Conv: Map to Classes
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

        logger.info(f"Initialized SegmentationHead: {in_channels} -> {hidden_dim} -> {num_classes}")

    def forward(self, x):
        """
        Args:
            x: [B, 768, 37, 37] - DINOv2 spatial features
        Returns:
            [B, 6, 37, 37] - Class logits per patch
        """
        return self.head(x)
  
class RailSegmentationModel(nn.Module):
    """
    Complete segmentation model: DINOv2 backbone + Segmentation head.
    
    Architecture:
    1. DINOv2 extracts features (frozen)
    2. Remove CLS (Classifier) token and reshape to spatial grid
    3. Segmentation head predicts classes
    """
    def __init__(self,
                    model_name=PRETRAINED_MODEL_NAME,
                        freeze_backbone=True):
        super().__init__()

        # Load DINOv2 backbone
        logger.info(f"Loading DINOv2 backbone from {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("DINOv2 backbone frozen (not trainable)")
        else:
            logger.info("DINOv2 backbone unfrozen (trainable)")

        # Segmentation Head
        self.seg_head = SegmentationHead()

        # Store config
        self.patch_grid_size = PATCH_GRID_SIZE
        self.num_classes = NUM_COMBINED_CLASSES

        logger.info(f"Model initialized: {self.count_parameters()} total parameters")
        logger.info(f"  Trainable: {self.count_parameters(trainable_only=True)}")

    def forward(self, x):
        """
        Forward pass through complete model.
        
        Args:
            x: [B, 3, 518, 518] - Preprocessed images
        
        Returns:
            [B, 3, 37, 37] - Class logits per patch
        """
        batch_size = x.shape[0]

        # 1. Extract features with DINOv2
        with torch.set_grad_enabled(self.training and self.backbone.training):
            outputs = self.backbone(x)
            features = outputs.last_hidden_state # [B, 1370, 768]

        # 2. Remove CLS token (first token)
        patch_features = features[:, 1:, :] # [B, 1369, 768]

        # 3. Reshape to spatial grid
        # 1369 tokens = 37x37 patches
        spatial_features = patch_features.reshape(
            batch_size,
            self.patch_grid_size,
            self.patch_grid_size,
            -1
        ) # [B, 37, 37, 768]

        # 4. Permute to channel-first format for conv layers
        spatial_features = spatial_features.permute(0, 3, 1, 2) # [B, 768, 37, 37]

        # 5. Segmentation Head
        logits = self.seg_head(spatial_features)

        return logits
    
    def count_parameters(self, trainable_only=False):
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self):
        """Freeze DINOv2 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze DINOv2 backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

def load_model(checkpoint_path=None, device=DEVICE):
    """
    Load model from checkpoint or initialize new model.
    
    Args:
        checkpoint_path: Path to saved checkpoint (optional)
        device: Device to load model on
    
    Returns:
        model: Initialized model
    """
    model = RailSegmentationModel()
    
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded")
    
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    return model

# Test function
if __name__ == "__main__":
    logger.info("Testing RailSegmentationModel...")
    
    # Create model
    rail_model = RailSegmentationModel()
    rail_model.eval()
    
    # Create dummy input
    DUMMY_BATCH_SIZE = 2
    dummy_input = torch.randn(DUMMY_BATCH_SIZE, 3, 518, 518)
    logger.info(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        output = rail_model(dummy_input)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Expected: [{DUMMY_BATCH_SIZE}, {NUM_COMBINED_CLASSES}, {PATCH_GRID_SIZE}, {PATCH_GRID_SIZE}]")

    # Verify shapes
    assert output.shape == (DUMMY_BATCH_SIZE, NUM_COMBINED_CLASSES, PATCH_GRID_SIZE, PATCH_GRID_SIZE), \
        f"Shape mismatch! Got {output.shape}"

    # Check value ranges
    logger.info(f"Output value range: [{output.min():.3f}, {output.max():.3f}]")

    # --- Test with actual sample from dataset ---
    logger.info("\n=== Testing with Real Data ===")
    from src.dataset import RailSem19Dataset

    dataset = RailSem19Dataset()
    image, mask = dataset[0]

    logger.info(f"Real image shape: {image.shape}")
    logger.info(f"Real mask shape: {mask.shape}")

    # Add batch dimension
    image_batch = image.unsqueeze(0)  # [1, 3, 518, 518]

    with torch.no_grad():
        predictions = rail_model(image_batch)
    
    logger.info(f"Prediction shape: {predictions.shape}")

    # Get predicted classes
    predicted_classes = predictions.argmax(dim=1)  # [1, 37, 37]
    logger.info(f"Predicted classes shape: {predicted_classes.shape}")
    logger.info(f"Unique predicted classes: {torch.unique(predicted_classes)}")
    logger.info(f"Ground truth classes: {torch.unique(mask)}")

    # Parameter summary
    logger.info("\n=== Model Summary ===")
    logger.info(f"Total parameters: {rail_model.count_parameters():,}")
    logger.info(f"Trainable parameters: {rail_model.count_parameters(trainable_only=True):,}")
 
    backbone_params = sum(p.numel() for p in rail_model.backbone.parameters())
    head_params = sum(p.numel() for p in rail_model.seg_head.parameters())
    logger.info(f"Backbone parameters: {backbone_params:,}")
    logger.info(f"Seg head parameters: {head_params:,}")

    logger.info("\n Model test complete!")
