import streamlit as st
import time
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from transformers import AutoImageProcessor

# Ensure your local src folder structure exists
from src.model import load_model
from src.utils import (
    preprocess_image,
    run_inference_and_upscale,
    postprocess_mask,
    create_overlay,
    check_track_suitability,
)
from src.hazard_clip import HazardCLIP
from src.config import (
    PRETRAINED_MODEL_NAME, BEST_MODEL_PATH, DEVICE,
    NUM_COMBINED_CLASSES, CLASS_COLORS_RGB,
    CLASS_COLORS_NORMALIZED, CONFIDENCE_THRESHOLD_DEFAULT,
    OVERLAY_ALPHA_DEFAULT, VOID_LABEL, FINAL_CLASS_NAMES
)
import os
import csv

# Class ids for quick access
TRACK_CLASS_ID = 5
VEGETATION_CLASS_ID = 2
OBJECT_CLASS_ID = 3
CLIP_HAZARD_THRESHOLD = 0.4
TRACK_VICINITY_DILATION = 0   # No expansion; stay on the rails to avoid side bleed
MIN_INTRUSION_RATIO = 0.03    # Require >=3% of track vicinity overlap
MIN_INTRUSION_PIXELS = 800    # Require a minimum absolute overlap area
MIN_ONTRACK_PIXELS = 80       # Require more evidence directly on track

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
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []

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

@st.cache_resource
def load_clip_model():
    """Load and cache CLIP-based hazard detector"""
    return HazardCLIP(device=DEVICE)

processor = load_processor()
best_model = load_cached_model()
hazard_clip = load_clip_model()
WEATHER_SCENARIOS_PATH = "/Users/aprajita/pmda-proj/weather_scenarios.csv"
WEATHER_CLASSIFICATION_PATH = "/Users/aprajita/pmda-proj/weather_classification.csv"

@st.cache_data
def load_weather_lookup():
    """
    Load weather metrics and classification from CSVs on disk.
    Returns a dict keyed by basename with merged info.
    """
    lookup = {}

    def read_csv(path):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except FileNotFoundError:
            return []

    scenarios_rows = read_csv(WEATHER_SCENARIOS_PATH)
    for row in scenarios_rows:
        key = os.path.basename(row.get("image_rel_path", "") or row.get("filename", ""))
        if not key:
            continue
        lookup[key] = {
            "scenario": row.get("scenario"),
            "brightness": row.get("brightness"),
            "contrast": row.get("contrast"),
            "edge_density": row.get("edge_density"),
        }

    classification_rows = read_csv(WEATHER_CLASSIFICATION_PATH)
    for row in classification_rows:
        key = os.path.basename(row.get("image_rel_path", "") or row.get("filename", ""))
        if not key:
            continue
        entry = lookup.get(key, {})
        # Prefer scenario from scenarios.csv; otherwise use classification scenario
        if not entry.get("scenario"):
            entry["scenario"] = row.get("scenario")
        entry["classification_confidence"] = row.get("confidence")
        lookup[key] = entry

    return lookup

weather_lookup = load_weather_lookup()

# --- 4. UI Layout & Title ---
st.title("Image Segmentation Demo")
st.markdown("Upload images or a local video (QuickTime .mov supported) to see segmentation results or run a slideshow.")

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

    st.divider()
    st.header("Local Video Input (optional)")
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=False,
        help="QuickTime .mov and other common formats supported"
    )
    video_fps = st.slider("Sample FPS", min_value=0.5, max_value=5.0, value=1.0, step=0.5, help="Frames per second to sample")
    video_process = st.button("Process Video", type="primary")

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

def draw_bounding_boxes(img_array: np.ndarray, boxes):
    """
    Draw bounding boxes on a numpy image array.
    """
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    colors = {
        OBJECT_CLASS_ID: (255, 0, 0),       # red
        VEGETATION_CLASS_ID: (0, 200, 0),   # green
    }

    for b in boxes:
        x_min, y_min, x_max, y_max = b["box"]
        color = colors.get(b["class_id"], (255, 255, 0))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 3, y_min + 3), b["label"], fill=color)

    return np.array(img)

def render_result_view(img, mask, scores, conf_thresh, alpha, clip_result=None, weather_row=None):
    """
    Helper to render the two-column view inside a placeholder.
    Note: For manual view, we generate overlay on the fly.
    """
    # Apply filtering
    filt_mask = postprocess_mask(mask, scores, conf_thresh)
    track_present = np.any(filt_mask == TRACK_CLASS_ID)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("🎯 Segmentation Overlay")
        # Generate overlay (no boxes)
        ov = create_overlay(img, filt_mask, alpha, CLASS_COLORS_NORMALIZED)
        st.image(ov, use_container_width=True)
    
    # Alert if a hazard is present on/near the track
    hazard, hazard_labels = detect_track_intrusion(filt_mask, scores, conf_thresh)
    clip_vote, clip_text = evaluate_clip_vote(clip_result, track_present)
    if hazard or clip_vote:
        messages = []
        if hazard:
            classes_text = ", ".join(hazard_labels)
            messages.append(f"{classes_text} on/near the track")
        if clip_vote:
            messages.append(clip_text)
        st.error("⚠️ Alert: " + " | ".join(messages))

    def compute_simple_metrics(pil_img: Image.Image):
        arr = np.array(pil_img.convert("L"), dtype=np.float32)
        brightness = float(arr.mean())
        contrast = float(arr.std())
        gy, gx = np.gradient(arr)
        edge = np.hypot(gx, gy)
        edge_density = float((edge > edge.mean()).mean() * 100.0)
        return brightness, contrast, edge_density

    if weather_row:
        suitability = check_track_suitability(
            weather_row.get("scenario"),
            weather_row.get("brightness"),
            weather_row.get("contrast"),
            weather_row.get("edge_density")
        )
    else:
        b, c, e = compute_simple_metrics(img)
        suitability = check_track_suitability(
            scenario=None,
            brightness=b,
            contrast=c,
            edge_density=e
        )

    # If hazards are detected, force STOP regardless of visibility
    status = suitability["status"]
    reason = suitability["reason"]
    if hazard or clip_vote:
        status = "STOP"
        reason = "Track blocked/hazard detected; requires clearance"
    elif status == "CAUTION":
        reason = "Drive with caution (visibility marginal)"

    if status == "GO":
        st.success(f"🟢 GO: {reason}")
    elif status == "CAUTION":
        st.warning(f"🟡 CAUTION: {reason}")
    else:
        st.error(f"🔴 STOP: {reason}")

    st.divider()
    display_legend_badges(filt_mask, CLASS_COLORS_RGB, FINAL_CLASS_NAMES)

def sample_video_frames(video_path: str, target_fps: float = 1.0, max_frames: int = 30):
    """
    Sample frames from a video path. Tries cv2 first, then imageio.
    Returns list of PIL Images.
    """
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("cv2 could not open video")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(int(video_fps // target_fps), 1)
        idx = 0
        taken = 0
        while taken < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                taken += 1
            idx += 1
        cap.release()
        if frames:
            return frames
    except Exception:
        pass

    # Fallback to imageio
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        video_fps = meta.get("fps", 30.0)
        stride = max(int(video_fps // target_fps), 1)
        for i, frame in enumerate(reader):
            if len(frames) >= max_frames:
                break
            if i % stride == 0:
                frames.append(Image.fromarray(frame))
        reader.close()
    except Exception as e:
        st.error(f"Failed to sample frames: {e}")

    return frames

def binary_dilation(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Lightweight binary dilation to mark a small vicinity around the track.
    Avoids extra dependencies while giving a buffer around the rails.
    """
    dilated = mask
    for _ in range(iterations):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        neighborhoods = [
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
            padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
        ]
        dilated = np.logical_or.reduce(neighborhoods)
    return dilated

def detect_track_intrusion(mask: np.ndarray, confidence_scores: np.ndarray, conf_threshold: float):
    """
    Detect if vegetation or objects are on/near the track.
    We expand the track mask slightly to catch close-by intrusions.
    """
    track_mask = mask == TRACK_CLASS_ID
    if not np.any(track_mask):
        return False, []

    h, w = mask.shape
    track_vicinity = binary_dilation(track_mask, iterations=TRACK_VICINITY_DILATION)
    # Build a narrow band around the track centerline to focus on true blockers
    x_coords = np.where(track_mask)[1]
    center_x = x_coords.mean() if len(x_coords) else w / 2
    band_half_width = max(int(w * 0.06), 4)  # ~12% band or at least 4px
    x_grid = np.arange(w)
    core_cols = np.abs(x_grid - center_x) <= band_half_width
    track_core = track_mask & core_cols

    hazard_classes = []
    vicinity_pixels = track_vicinity.sum()
    conf_mask = confidence_scores >= conf_threshold

    for cls_id in (VEGETATION_CLASS_ID, OBJECT_CLASS_ID):
        class_mask = mask == cls_id
        overlap_vicinity = track_vicinity & class_mask & conf_mask
        overlap_track = track_core & class_mask & conf_mask
        overlap_pixels = overlap_vicinity.sum()
        overlap_ratio = (overlap_pixels / vicinity_pixels) if vicinity_pixels else 0.0
        overlap_on_track = overlap_track.sum()
        if (
            overlap_on_track >= MIN_ONTRACK_PIXELS
            or (overlap_pixels >= MIN_INTRUSION_PIXELS and overlap_ratio >= MIN_INTRUSION_RATIO)
        ):
            hazard_classes.append(FINAL_CLASS_NAMES.get(cls_id, f"Class {cls_id}"))

    return len(hazard_classes) > 0, hazard_classes

def evaluate_clip_vote(clip_result, track_present: bool, threshold: float = CLIP_HAZARD_THRESHOLD):
    """
    Determine if CLIP predicts a hazard with enough confidence to count as a vote.
    """
    if not clip_result or not track_present:
        return False, ""

    label = clip_result.get("label", "").lower()
    confidence = clip_result.get("confidence", 0.0)

    if label == "clear railway track":
        return False, ""

    if confidence < threshold:
        return False, ""

    return True, f"{clip_result['label']} ({confidence:.2f})"

# --- 7. Main Logic ---

uploaded_files = st.file_uploader(
    "Choose images...",
    type=["jpg", "jpeg", "png"],
    help="Upload images for segmentation",
    accept_multiple_files=True
)

if video_process:
    if video_file:
        suffix = Path(video_file.name).suffix or ".mp4"
        tmp_dir = tempfile.mkdtemp()
        local_video_path = Path(tmp_dir) / f"uploaded_video{suffix}"
        with open(local_video_path, "wb") as f:
            f.write(video_file.read())

        frames = sample_video_frames(str(local_video_path), target_fps=video_fps, max_frames=30)
        if frames:
            st.session_state.video_frames = [
                {"name": f"video_frame_{i}.jpg", "image": frame}
                for i, frame in enumerate(frames)
            ]
            st.success(f"Captured {len(frames)} frames from uploaded video.")
        else:
            st.session_state.video_frames = []
            st.error("Could not sample frames from the uploaded video. Please try a different file or FPS.")
    else:
        st.warning("Please upload a video file to process.")

# Combine uploaded files and video frames into a unified list
input_items = []
if uploaded_files:
    for f in uploaded_files:
        input_items.append({"name": f.name, "source": "upload", "file": f})
for vf in st.session_state.video_frames:
    input_items.append({"name": vf["name"], "source": "video", "image": vf["image"]})

if input_items:
    # --- PROCESSING PHASE ---
    current_file_names = [item["name"] for item in input_items]
    
    # Identify new files
    files_to_process = [item for item in input_items if item["name"] not in st.session_state.batch_results]
    
    if files_to_process:
        with st.spinner(f"🔄 Processing {len(files_to_process)} new items..."):
            for item in files_to_process:
                try:
                    if item["source"] == "upload":
                        image = Image.open(item["file"]).convert("RGB")
                    else:
                        image = item["image"].convert("RGB")
                    image_key = item["name"]
                    
                    # Inference
                    image_tensor = preprocess_image(image, processor)
                    pred_mask_upscaled, confidence_scores = run_inference_and_upscale(
                        image, image_tensor, best_model, DEVICE
                    )

                    # CLIP-based hazard prediction (secondary vote)
                    clip_result = hazard_clip.predict_image(image)

                    # Store results
                    st.session_state.batch_results[image_key] = {
                        "image": image,
                        "pred_mask": pred_mask_upscaled,
                        "confidence_scores": confidence_scores,
                        "clip_hazard": clip_result
                    }
                except Exception as e:
                    st.error(f"Error processing {image_key}: {e}")

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
                hazard, hazard_labels = detect_track_intrusion(filt_mask, scores, confidence_threshold)
                track_present = np.any(filt_mask == TRACK_CLASS_ID)
                
                # Generate the overlay image here
                final_overlay = create_overlay(img, filt_mask, overlay_alpha, CLASS_COLORS_NORMALIZED)
                clip_result = st.session_state.batch_results[fname].get("clip_hazard")
                clip_vote, clip_text = evaluate_clip_vote(clip_result, track_present)
                weather_row = weather_lookup.get(os.path.basename(fname))
                suitability = None
                if weather_row:
                    suitability = check_track_suitability(
                        weather_row.get("scenario"),
                        weather_row.get("brightness"),
                        weather_row.get("contrast"),
                        weather_row.get("edge_density")
                    )

                slideshow_data.append({
                    "name": fname,
                    "original": img,
                    "overlay": final_overlay,
                    "mask": filt_mask,
                    "hazard": hazard,
                    "hazard_labels": hazard_labels,
                    "clip_vote": clip_vote,
                    "clip_text": clip_text,
                    "suitability": suitability
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
                    st.image(slide['original'])
                with c2:
                    st.subheader("🎯 Segmentation")
                    st.image(slide['overlay'])
                
                if slide["hazard"] or slide["clip_vote"]:
                    messages = []
                    if slide["hazard"]:
                        classes_text = ", ".join(slide["hazard_labels"])
                        messages.append(f"{classes_text} on/near the track")
                    if slide["clip_vote"]:
                        messages.append(slide["clip_text"])
                    st.error("⚠️ Alert: " + " | ".join(messages))

                suitability = slide.get("suitability")
                # Enforce STOP if any hazard/clip vote is detected, regardless of visibility
                if slide["hazard"] or slide["clip_vote"]:
                    st.error("🔴 STOP: Track blocked/hazard detected; requires clearance")
                elif suitability:
                    status = suitability["status"]
                    reason = suitability["reason"]
                    if status == "GO":
                        st.success(f"🟢 GO: {reason}")
                    elif status == "CAUTION":
                        st.warning(f"🟡 CAUTION: {reason}")
                    else:
                        st.error(f"🔴 STOP: {reason}")

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
                weather_row = weather_lookup.get(os.path.basename(selected_file_name))
                # In manual mode, we render on the fly to allow slider adjustments
                render_result_view(
                    data["image"], 
                    data["pred_mask"], 
                    data["confidence_scores"], 
                    confidence_threshold, 
                    overlay_alpha,
                    data.get("clip_hazard"),
                    weather_row
                )

# Footer
st.divider()
st.markdown("""
### 📝 Notes
- **Batch Upload:** Select multiple files to analyze a sequence.
- **Local Video:** Upload a QuickTime .mov or other common format in the sidebar to auto-sample frames.
- **Slideshow:** Use the sidebar controls to adjust speed (default 0.5s).
- **Smoothness:** Transitions are animated to reduce eye strain.
""")
