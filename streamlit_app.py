import streamlit as st
from PIL import Image
import numpy as np

from transformers import AutoImageProcessor

from src.model import load_model
from src.utils import preprocess_image, run_inference_and_upscale, postprocess_mask, create_overlay
from src.config import (
    PRETRAINED_MODEL_NAME, BEST_MODEL_PATH, DEVICE,
    NUM_COMBINED_CLASSES, CLASS_COLORS_RGB,
    CLASS_COLORS_NORMALIZED, CONFIDENCE_THRESHOLD_DEFAULT,
    OVERLAY_ALPHA_DEFAULT, VOID_LABEL, FINAL_CLASS_NAMES
)

# Page configuration
st.set_page_config(
    page_title="Segmentation Demo",
    page_icon="🚈",
    layout="wide"
)

# Cache processor and model loading
@st.cache_resource
def load_processor():
    """Load and cache DINOv2 image processor"""
    return AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)

@st.cache_resource
def load_cached_model():
    """Load model once and cache it"""
    model = load_model(BEST_MODEL_PATH, DEVICE)
    model.eval()
    return model

# Load processor and model
processor = load_processor()
best_model = load_cached_model()

# Title and description
st.title("Image Segmentation Demo")
st.markdown("Upload an image to see segmentation results")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Check if image exists in session
    has_image = 'pred_mask' in st.session_state
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD_DEFAULT,
        step=0.05,
        disabled=not has_image,
        help=(
            "Upload an image to enable" if not has_image 
            else "Filter low-confidence predictions"
        )
    )
    
    # Opacity slider for overlay
    overlay_alpha = st.slider(
        "Overlay Opacity",
        min_value=0.0,
        max_value=1.0,
        value=OVERLAY_ALPHA_DEFAULT,
        step=0.1,
        disabled=not has_image,
        help=(
            "Upload an image to enable" if not has_image
            else "Transparency of the segmentation overlay"
        )
    )

    if not has_image:
        st.info("Upload an image to adjust settings")
    
    st.divider()
    st.header("Model Info")
    st.info(
        f"**Model:** DINOv2 + Segmentation Head\n"
        f"**Classes detectable:** {NUM_COMBINED_CLASSES}\n"
        f"**Device:** {DEVICE}"
    )

def display_legend_badges(filtered_mask, class_colors, id_to_name):
    """Display classes as compact badges"""
    unique_classes = np.unique(filtered_mask)
    unique_classes = unique_classes[unique_classes != VOID_LABEL]
    
    if len(unique_classes) == 0:
        return
    
    st.subheader("🎨 Detected Classes")
    
    # Create HTML for badges
    badges_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
    
    for class_id in sorted(unique_classes):
        class_name = id_to_name.get(class_id, f"Class {class_id}")
        color = class_colors.get(class_id, [128, 128, 128])
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
        
        # Calculate text color (white or black) based on background brightness
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        text_color = '#000000' if brightness > 128 else '#FFFFFF'
        
        # Build badge with explicit style properties
        style = (
            f"background-color: {color_hex}; "
            f"color: {text_color}; "
            f"padding: 4px 12px; "
            f"border-radius: 12px; "
            f"font-size: 13px; "
            f"font-weight: 500; "
            f"display: inline-block; "
            f"margin: 2px;"
        )
        
        badges_html += f'<span style="{style}">{class_name}</span>'
  
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

# --- Main content area ---
# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload any image for segmentation"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Generate a unique key for this image
    image_key = uploaded_file.name + str(uploaded_file.size)

    # Check if this is a new image
    is_new_image = (
        'last_image_key' not in st.session_state or 
        st.session_state.last_image_key != image_key
    )
    
    # Only run inference if this is a new image
    if is_new_image:
        with st.spinner("🔄 Running inference..."):
            # Preprocess
            image_tensor = preprocess_image(image, processor)

            # Run inference
            pred_mask_upscaled, confidence_scores = run_inference_and_upscale(image, image_tensor, best_model, DEVICE)

            # Store in session state
            st.session_state.pred_mask = pred_mask_upscaled
            st.session_state.confidence_scores = confidence_scores
            st.session_state.last_image_key = image_key
            st.session_state.original_image = image

            # Force rerun to enable sliders
            st.rerun()
    
    # Retrieve cached results
    pred_mask = st.session_state.pred_mask
    confidence_scores = st.session_state.confidence_scores
    filtered_mask = postprocess_mask(pred_mask, confidence_scores, confidence_threshold)

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, width=None)
    
    with col2:
        st.subheader("🎯 Segmentation Overlay")
        overlay = create_overlay(image, filtered_mask, overlay_alpha, CLASS_COLORS_NORMALIZED)
        st.image(overlay, width=None)
    
    # Show legend
    st.divider()
    display_legend_badges(filtered_mask, CLASS_COLORS_RGB, FINAL_CLASS_NAMES)

# Footer
st.divider()
st.markdown("""
### 📝 Notes
- Adjust the **confidence threshold** to filter out low-confidence predictions
- Adjust the **overlay opacity** to see more or less of the original image
- Upload any image to test the model's generalization capability
""")
