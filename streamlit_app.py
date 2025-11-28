import streamlit as st
import time
from PIL import Image
import numpy as np

from transformers import AutoImageProcessor

# Ensure your local src folder structure exists
from src.model import load_model
from src.utils import preprocess_image, run_inference_and_upscale, postprocess_mask, create_overlay
from src.config import (
    PRETRAINED_MODEL_NAME, BEST_MODEL_PATH, DEVICE,
    NUM_COMBINED_CLASSES, CLASS_COLORS_RGB,
    CLASS_COLORS_NORMALIZED, CONFIDENCE_THRESHOLD_DEFAULT,
    OVERLAY_ALPHA_DEFAULT, VOID_LABEL, FINAL_CLASS_NAMES
)

# --- 1. Page Config & Custom CSS ---
st.set_page_config(
    page_title="Segmentation Demo",
    page_icon="🚈",
    layout="wide"
)

def inject_custom_css():
    """
    Inject CSS for smooth image transitions.
    Optimized for fast slideshows (0.5s) to avoid strobing.
    """
    st.markdown("""
        <style>
            img {
                /* Animation duration: 0.3s (fits comfortably inside 0.5s sleep) */
                animation: fadeIn 0.3s ease-in-out;
            }
            
            @keyframes fadeIn {
                /* Start at 50% opacity instead of 0% to reduce flashing/strobing */
                0% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 2. Session State Initialization ---
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = {}

# --- 3. Cache & Model Loading ---
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

processor = load_processor()
best_model = load_cached_model()

# --- 4. UI Layout & Title ---
st.title("Image Segmentation Demo")
st.markdown("Upload a batch of images to see segmentation results or run a slideshow.")

# --- 5. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    # Check if we have processed results
    has_results = len(st.session_state.batch_results) > 0
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD_DEFAULT,
        step=0.05,
        disabled=not has_results,
        help="Filter low-confidence predictions"
    )
    
    # Opacity slider
    overlay_alpha = st.slider(
        "Overlay Opacity",
        min_value=0.0,
        max_value=1.0,
        value=OVERLAY_ALPHA_DEFAULT,
        step=0.1,
        disabled=not has_results,
        help="Transparency of the segmentation overlay"
    )

    if not has_results:
        st.info("Upload images to enable settings")
    
    st.divider()
    
    # Slideshow specific controls
    st.header("🎞️ Slideshow Settings")
    slideshow_speed = st.slider(
        "Speed (seconds per slide)",
        min_value=0.1,
        max_value=3.0,
        value=0.5, # Default to 0.5s as requested
        step=0.1,
        help="Time to display each image"
    )
    
    st.divider()
    st.header("Model Info")
    st.info(
        f"**Model:** DINOv2 + Segmentation Head\n"
        f"**Classes detectable:** {NUM_COMBINED_CLASSES}\n"
        f"**Device:** {DEVICE}"
    )

# --- 6. Helper Functions ---
def display_legend_badges(filtered_mask, class_colors, id_to_name):
    """Display classes as compact badges"""
    unique_classes = np.unique(filtered_mask)
    unique_classes = unique_classes[unique_classes != VOID_LABEL]
    
    if len(unique_classes) == 0:
        return
    
    st.subheader("🎨 Detected Classes")
    
    badges_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
    for class_id in sorted(unique_classes):
        class_name = id_to_name.get(class_id, f"Class {class_id}")
        color = class_colors.get(class_id, [128, 128, 128])
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
        
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        text_color = '#000000' if brightness > 128 else '#FFFFFF'
        
        style = (
            f"background-color: {color_hex}; color: {text_color}; "
            f"padding: 4px 12px; border-radius: 12px; font-size: 13px; "
            f"font-weight: 500; display: inline-block; margin: 2px;"
        )
        badges_html += f'<span style="{style}">{class_name}</span>'
  
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

def render_result_view(img, mask, scores, conf_thresh, alpha):
    """
    Helper to render the two-column view inside a placeholder.
    Note: For manual view, we generate overlay on the fly.
    """
    # Apply filtering
    filt_mask = postprocess_mask(mask, scores, conf_thresh)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("🎯 Segmentation Overlay")
        # Generate overlay
        ov = create_overlay(img, filt_mask, alpha, CLASS_COLORS_NORMALIZED)
        st.image(ov, use_container_width=True)
    
    st.divider()
    display_legend_badges(filt_mask, CLASS_COLORS_RGB, FINAL_CLASS_NAMES)

# --- 7. Main Logic ---

uploaded_files = st.file_uploader(
    "Choose images...",
    type=["jpg", "jpeg", "png"],
    help="Upload images for segmentation",
    accept_multiple_files=True
)

if uploaded_files:
    # --- PROCESSING PHASE ---
    current_file_names = [f.name for f in uploaded_files]
    
    # Identify new files
    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.batch_results]
    
    if files_to_process:
        with st.spinner(f"🔄 Processing {len(files_to_process)} new images..."):
            for file in files_to_process:
                try:
                    image = Image.open(file).convert("RGB")
                    image_key = file.name
                    
                    # Inference
                    image_tensor = preprocess_image(image, processor)
                    pred_mask_upscaled, confidence_scores = run_inference_and_upscale(
                        image, image_tensor, best_model, DEVICE
                    )

                    # Store results
                    st.session_state.batch_results[image_key] = {
                        "image": image,
                        "pred_mask": pred_mask_upscaled,
                        "confidence_scores": confidence_scores
                    }
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

    # --- CONTROLS PHASE ---
    # Selector for manual view
    selected_file_name = st.sidebar.selectbox(
        "📂 Select Image to View",
        options=current_file_names,
        index=0
    )
    
    # Slideshow toggle button
    start_slideshow = st.sidebar.button(
        "▶️ Play Slideshow", 
        disabled=len(current_file_names) < 2
    )

    # --- DISPLAY PHASE ---
    main_display = st.empty()

    if start_slideshow:
        # --- SLIDESHOW MODE ---
        # 1. Pre-calculation Phase (Crucial for smooth 0.5s transitions)
        slideshow_data = []
        progress_bar = st.progress(0, text="⏳ Pre-rendering frames for smooth playback...")
        
        for i, fname in enumerate(current_file_names):
            if fname in st.session_state.batch_results:
                data = st.session_state.batch_results[fname]
                
                # We calculate everything NOW so the loop is instant
                img = data["image"]
                mask = data["pred_mask"]
                scores = data["confidence_scores"]
                filt_mask = postprocess_mask(mask, scores, confidence_threshold)
                
                # Generate the overlay image here
                final_overlay = create_overlay(img, filt_mask, overlay_alpha, CLASS_COLORS_NORMALIZED)
                
                slideshow_data.append({
                    "name": fname,
                    "original": img,
                    "overlay": final_overlay,
                    "mask": filt_mask
                })
            progress_bar.progress((i + 1) / len(current_file_names))
        
        progress_bar.empty()

        # 2. Playback Phase
        for slide in slideshow_data:
            with main_display.container():
                st.info(f"▶️ Slideshow: {slide['name']}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📷 Original")
                    st.image(slide['original'], use_container_width=True)
                with c2:
                    st.subheader("🎯 Segmentation")
                    st.image(slide['overlay'], use_container_width=True)
                
                st.divider()
                display_legend_badges(slide['mask'], CLASS_COLORS_RGB, FINAL_CLASS_NAMES)
            
            # The pause duration (default 0.5s)
            time.sleep(slideshow_speed)
        
        st.success("Slideshow finished!")
        time.sleep(1)
        st.rerun()

    else:
        # --- MANUAL MODE ---
        if selected_file_name and selected_file_name in st.session_state.batch_results:
            with main_display.container():
                data = st.session_state.batch_results[selected_file_name]
                # In manual mode, we render on the fly to allow slider adjustments
                render_result_view(
                    data["image"], 
                    data["pred_mask"], 
                    data["confidence_scores"], 
                    confidence_threshold, 
                    overlay_alpha
                )

# Footer
st.divider()
st.markdown("""
### 📝 Notes
- **Batch Upload:** Select multiple files to analyze a sequence.
- **Slideshow:** Use the sidebar controls to adjust speed (default 0.5s).
- **Smoothness:** Transitions are animated to reduce eye strain.
""")