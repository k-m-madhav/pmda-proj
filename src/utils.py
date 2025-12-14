# Misc helpers (logging, color conversion, etc.)

# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import logging
import os
from datetime import datetime

from src.config import INPUT_SIZE, DEVICE, VOID_LABEL, CLASS_COLORS_NORMALIZED

def get_logger(name, level=logging.INFO, log_dir='logs'):
    """Returns a logger that writes to stdout and a rotating log file."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    # Create log directory if missing
    os.makedirs(log_dir, exist_ok=True)

    # Unique log file name for each run (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{name}_{timestamp}.log"

    # File handler
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.setLevel(level)
    logger.propagate = False
    return logger

def preprocess_image(image, processor, input_size=INPUT_SIZE, device=DEVICE):
    """
    Preprocess a PIL Image for model inference.
    
    Args:
        image: PIL Image
        processor: DINOv2 image processor (from transformers)
        input_size: Dimension to convert the image into
        device: torch device
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model
    """
    # Use the same DINOv2 image processor 
    inputs = processor(
        images=image,
        return_tensors='pt',
        size={'height': input_size, 'width': input_size}
    )
    image_tensor = inputs['pixel_values'].to(device).squeeze(0)

    return image_tensor

def run_inference_and_upscale(original_image, image_tensor, model, device, class_weights=None):
    """
    Run inference and return predictions + confidence scores
    
    Args:
        original_image: Input Image
        image_tensor: Processed image converted to a tensor
        model: Trained combination of DINOv2 + Segmentation Head
        device: torch device

    Returns:
        Predicted Mask upscaled to the original image's dimensions
        Confidence scores also upscaled to the original image's dimensions
    """
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_batch)  # [1, num_classes, H, W]

        # Optional class weighting to bias certain classes
        if class_weights is not None:
            weight_t = torch.tensor(class_weights, device=device).view(1, -1, 1, 1)
            outputs = outputs * weight_t
        
        # Get probabilities
        probabilities = torch.softmax(outputs, dim=1)[0]  # [num_classes, H, W]
        max_probs, pred_mask = torch.max(probabilities, dim=0)  # [H, W], [H, W]

        # Original image dimensions to scale to
        orig_w, orig_h = original_image.size

        # --- Resize predicted mask (nearest, integers) ---
        pred_mask_up = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0).float(), # [1, 1, H, W]
            size=(orig_h, orig_w),
            mode="nearest"
        )[0, 0].cpu().numpy().astype(np.uint8)

        # --- Resize confidence map (bilinear, floats) ---
        max_probs_up = F.interpolate(
            max_probs.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )[0, 0].cpu().numpy()
        
        return pred_mask_up, max_probs_up

def postprocess_mask(pred_mask, confidence_scores, threshold):
    """
    Filter a predicted segmentation mask using per-pixel confidence scores.

    This function overwrites low-confidence pixels in the predicted mask with a
    designated `VOID_LABEL`, based on a user-defined confidence threshold.
    Pixels whose confidence scores fall below the threshold
    are treated as invalid or unreliable predictions.

    Args:
        pred_mask (np.ndarray): The predicted segmentation mask (HxW), where each
            element represents a class ID.
        confidence_scores (np.ndarray): Array of per-pixel confidence values
            (HxW), typically in the range [0, 1].
        threshold (float): Minimum confidence required for a pixel to be kept.
            Pixels with confidence < threshold are replaced with `VOID_LABEL`.

    Returns:
        np.ndarray: A copy of `pred_mask` in which low-confidence pixels have been
        replaced by `VOID_LABEL`.

    Notes:
        - Both `pred_mask` and `confidence_scores` must have the same spatial shape.
    """
    filtered_mask = pred_mask.copy()
    filtered_mask[confidence_scores < threshold] = VOID_LABEL  # or background_id
    return filtered_mask

def create_colored_mask(mask, class_colors=CLASS_COLORS_NORMALIZED):
    """
    Convert class mask to RGB visualization.
    
    Args:
        mask: numpy array of shape (H, W) with class indices
        colors: dict mapping class_id to RGB color
    
    Returns:
        RGB image of shape (H, W, 3)
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)

    for class_id, color in class_colors.items():
        colored_mask[mask == class_id] = color

    return colored_mask

def create_overlay(original_image, pred_mask, alpha, class_colors=CLASS_COLORS_NORMALIZED):
    """
    Create a visualization by blending an original image with a colorized
    segmentation mask.

    This function converts the input image to a NumPy array (if needed),
    normalizes it to the [0, 1] range, maps class IDs in the predicted mask
    to RGB colors, and blends the two using an alpha value to control
    transparency.

    Args:
        original_image (np.ndarray or PIL.Image.Image):
            The input image to be visualized. Can be a NumPy array (HxWx3) or
            a PIL image. If the pixel values are in [0, 255], they are normalized
            to [0, 1].
        pred_mask (np.ndarray):
            Predicted segmentation mask (HxW), where each pixel contains a class ID.
        alpha (float):
            Blending factor between 0 and 1. Higher values make the segmentation
            mask more visible; lower values preserve more of the original image.
        class_colors (dict or np.ndarray, optional):
            A mapping from class IDs to normalized RGB colors in [0, 1].
            Defaults to `CLASS_COLORS_NORMALIZED`.

    Returns:
        np.ndarray: The blended overlay image (HxWx3) with pixel values in [0, 1].

    Notes:
        - Uses `create_colored_mask` to convert the mask into an RGB image.
    """
    # Convert image to numpy if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Normalize image to [0, 1]
    if original_image.max() > 1.0:
        original_image = original_image / 255.0

    # Create colored mask
    pred_colored = create_colored_mask(pred_mask, class_colors)

    # Blend image and mask
    overlay = (1 - alpha) * original_image + alpha * pred_colored
    overlay = np.clip(overlay, 0, 1)

    return overlay

def check_track_suitability(scenario, brightness, contrast, edge_density):
    """
    Decide if visibility is suitable for rail operations.
    Returns dict with status in {GO, CAUTION, STOP} and a human-readable reason.
    """
    scenario_l = (scenario or "").lower()
    thresholds = {
        "brightness": {"go": 60.0, "caution": 40.0},
        "contrast": {"go": 25.0, "caution": 15.0},
        "edge_density": {"go": 10.0, "caution": 6.0},
    }

    # Scenario-specific adjustments
    if "night" in scenario_l:
        thresholds["brightness"] = {"go": 50.0, "caution": 30.0}
        thresholds["contrast"]["caution"] = 10.0
        thresholds["edge_density"]["caution"] = 4.0
    elif "fog" in scenario_l or "mist" in scenario_l:
        thresholds["contrast"]["go"] = 35.0
        thresholds["edge_density"]["go"] = 12.0
    elif "rain" in scenario_l:
        thresholds["contrast"]["go"] = 30.0
        thresholds["edge_density"]["go"] = 12.0
    elif "snow" in scenario_l:
        thresholds["brightness"]["go"] = 70.0
        thresholds["contrast"]["go"] = 30.0
        thresholds["edge_density"]["go"] = 12.0

    def to_float(val):
        try:
            return float(val)
        except Exception:
            return None

    vals = {
        "brightness": to_float(brightness),
        "contrast": to_float(contrast),
        "edge_density": to_float(edge_density),
    }

    status = "GO"
    reasons = []

    for metric, val in vals.items():
        go_thr = thresholds[metric]["go"]
        caution_thr = thresholds[metric]["caution"]

        if val is None:
            if status != "STOP":
                status = "CAUTION"
            reasons.append(f"{metric} missing")
            continue

        if val < caution_thr:
            status = "STOP"
            reasons.append(f"{metric} too low ({val:.1f} < {caution_thr})")
        elif val < go_thr:
            if status != "STOP":
                status = "CAUTION"
            reasons.append(f"{metric} marginal ({val:.1f} < {go_thr})")

    if not reasons and status == "GO":
        reasons.append("Visibility sufficient for rail operations")

    return {"status": status, "reason": "; ".join(reasons)}
