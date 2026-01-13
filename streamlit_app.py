import streamlit as st
import time
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
# Street (DeepLab) import commented out; rail-only mode enabled
# import torch
# from torchvision.models.segmentation import (
#     deeplabv3_resnet50,
#     DeepLabV3_ResNet50_Weights
# )
import torch
from skimage.measure import label as cc_label

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
CLIP_HAZARD_THRESHOLD = 0.75  # Increased from 0.65 to reduce false positives
TRACK_VICINITY_DILATION = 1   # Minimal dilation - focus on actual track
MIN_INTRUSION_RATIO = 0.08    # Require >=8% of track core overlap (more strict)
MIN_INTRUSION_PIXELS = 2500   # Higher minimum overlap area
MIN_ONTRACK_PIXELS = 400      # Evidence directly on track (increased)
MIN_ONTRACK_PIXELS_OBJECT = 150  # Smaller objects (person/vehicle) on track
MIN_BLOB_PIXELS = 250         # Ignore tiny specks
MIN_MAJOR_OVERLAP_PIXELS = 800    # Core overlap pixels to force STOP (doubled)
MIN_MAJOR_OVERLAP_RATIO = 0.10    # Core overlap ratio to force STOP (10% - tripled)
MIN_VICINITY_OVERLAP_PIXELS = 1200  # Vicinity overlap pixels to force STOP
MIN_VICINITY_OVERLAP_RATIO = 0.05   # Vicinity overlap ratio to force STOP
TRACK_CORRIDOR_TOP_FRAC = 0.35

# Bias rail model toward track/object for cleaner detection
RAIL_CLASS_WEIGHTS_BIASED = [1.0, 1.0, 1.0, 6.0, 2.0, 3.0]

# KPI thresholds
HORIZON_GO = 0.8   # 80% visible track is healthy
HORIZON_CAUTION = 0.5
SNOW_BRIGHTNESS_THRESHOLD = 200  # Increased from 175 - much brighter to avoid white trains
SNOW_COLOR_RANGE = 20  # Reduced from 35 - more uniform color required
SNOW_CAUTION_COVERAGE = 0.25  # Increased from 0.12 - need more coverage to trigger
SNOW_HEAVY_COVERAGE = 0.45  # Increased from 0.35

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

            /* KPI cards */
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin: 12px 0 6px 0;
            }
            .kpi-card {
                background: #ffffff;
                border-radius: 12px;
                padding: 12px 14px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.08);
                border: 1px solid #e6e6e6;
                position: relative;
                overflow: hidden;
            }
            .kpi-card::before {
                content: "";
                position: absolute;
                top: 0; left: 0;
                width: 100%; height: 4px;
                background: linear-gradient(90deg, #7bd389, #4ca1af);
            }
            .kpi-title {
                font-size: 13px;
                font-weight: 600;
                color: #5b6570;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .kpi-value {
                font-size: 26px;
                font-weight: 700;
                color: #1f2933;
                margin-bottom: 2px;
            }
            .kpi-sub {
                font-size: 12px;
                color: #6b7280;
            }
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 600;
            }
            .pill.green { background: #e8f5e9; color: #1f7a3d; }
            .pill.amber { background: #fff7e6; color: #b37400; }
            .pill.red { background: #ffe8e6; color: #b83227; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 2. Session State Initialization ---
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = {}
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []
if 'kpi_history' not in st.session_state:
    st.session_state.kpi_history = []

SEGMENTATION_MODE = "Rail (DINOv2)"

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
st.title("Image / Video Segmentation Demo")
st.markdown("Upload images or a local video (QuickTime .mov supported) to see segmentation results or run a slideshow.")

# --- 5. Sidebar Controls ---
# Select Mode
with st.sidebar:
    st.title("🛠️ Dashboard")
    
    st.header("📊 Processing Mode")
    app_mode = st.segmented_control(
        "Select Analysis Target", 
        options=["Image", "Video"], 
        default="Image",
        label_visibility="collapsed"
    )
    
    st.header("⚙️ Controls")
    st.info("Segmentation Model: Rail (DINOv2)", icon="🚈")
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
    
    # Video-specific controls
    if app_mode == "Video":
        st.divider()
        st.header("🎬 Video Settings")
        video_fps = st.slider(
            "Sample FPS",
            0.5, 5.0, 1.0, step=0.5,
            help="Frames per second to sample"
        )
    
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

# --- KPI helpers ---
def compute_simple_metrics(pil_img: Image.Image):
    arr = np.array(pil_img.convert("L"), dtype=np.float32)
    brightness = float(arr.mean())
    contrast = float(arr.std())
    gy, gx = np.gradient(arr)
    edge = np.hypot(gx, gy)
    edge_density = float((edge > edge.mean()).mean() * 100.0)
    return brightness, contrast, edge_density

def compute_track_band(mask: np.ndarray) -> np.ndarray:
    """Compute a center corridor band for track-related checks."""
    h, w = mask.shape
    track_mask = mask == TRACK_CLASS_ID
    x_coords = np.where(track_mask)[1]
    center_x = x_coords.mean() if len(x_coords) else w / 2
    band_half_width = max(int(w * 0.06), 4)
    x_grid = np.arange(w)
    core_cols = np.abs(x_grid - center_x) <= band_half_width
    if np.any(track_mask):
        min_y = np.where(track_mask)[0].min()
    else:
        min_y = int(h * 0.35)
    band_rows = np.arange(h) >= min_y
    return np.tile(core_cols, (h, 1)) & band_rows[:, None]

def compute_snow_coverage(image: Image.Image, track_band: np.ndarray) -> float:
    """Estimate snow coverage ratio within the track corridor band."""
    if track_band is None or not np.any(track_band):
        return 0.0
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    if arr.shape[0] != track_band.shape[0] or arr.shape[1] != track_band.shape[1]:
        return 0.0
    arr_f = arr.astype(np.float32)
    r = arr_f[..., 0]
    g = arr_f[..., 1]
    b = arr_f[..., 2]
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    brightness = (r + g + b) / 3.0
    
    # Very strict snow detection: bright, uniform white pixels
    # Also check that it's actually white (all channels high), not just bright
    snow_pixels = (
        (brightness >= SNOW_BRIGHTNESS_THRESHOLD) & 
        ((max_c - min_c) <= SNOW_COLOR_RANGE) &
        (r >= 180) & (g >= 180) & (b >= 180)  # Must be white, not just bright
    )
    coverage = float(snow_pixels[track_band].mean())
    
    # Only report if coverage is significant (avoid false positives from reflections)
    return coverage if coverage >= 0.05 else 0.0

def compute_horizon(track_mask: np.ndarray) -> float:
    """How far the track extends vertically (0-1), ignoring tiny speckles."""
    if track_mask is None or track_mask.size == 0:
        return 0.0
    row_counts = track_mask.sum(axis=1)
    min_row_pixels = max(2, int(track_mask.shape[1] * 0.0015))
    valid_rows = np.where(row_counts >= min_row_pixels)[0]
    if valid_rows.size == 0:
        return 0.0
    min_y = valid_rows.min()
    h = track_mask.shape[0]
    return max(0.0, min(1.0, (h - min_y) / h))

def compute_active_risks(filt_mask: np.ndarray, track_band: np.ndarray) -> int:
    """Count object blobs near/over track."""
    if track_band is None or not np.any(track_band):
        h, w = filt_mask.shape
        x_grid = np.arange(w)
        center_x = w / 2
        band_half_width = max(int(w * 0.06), 4)
        core_cols = np.abs(x_grid - center_x) <= band_half_width
        track_band = np.tile(core_cols, (h, 1))
    vicinity = binary_dilation(track_band, iterations=max(TRACK_VICINITY_DILATION, 1))
    risk_mask = (filt_mask == OBJECT_CLASS_ID) & vicinity
    labels, num = cc_label(risk_mask, return_num=True, connectivity=1)
    count = 0
    for i in range(1, num + 1):
        if (labels == i).sum() >= MIN_BLOB_PIXELS:
            count += 1
    return count

def compute_rail_confidence(filt_mask: np.ndarray, scores: np.ndarray, conf_thresh: float) -> float:
    rail_pixels = (filt_mask == TRACK_CLASS_ID) & (scores >= conf_thresh)
    if not np.any(rail_pixels):
        return 0.0
    return float(scores[rail_pixels].mean())

def is_video_frame(filename):
    """Detect if this is a video frame vs static image."""
    if not filename:
        return False
    return "video_frame_" in str(filename).lower() or filename.startswith("video_frame")

def analyze_frame(image, mask, scores, conf_thresh, clip_result, weather_row, is_video=False):
    # Two thresholds: strict for display, relaxed for hazard detection
    filt_mask = postprocess_mask(mask, scores, conf_thresh)
    relaxed_thresh = max(conf_thresh * 0.5, 0.2)
    hazard_mask = postprocess_mask(mask, scores, relaxed_thresh)

    track_present = np.any(hazard_mask == TRACK_CLASS_ID)
    track_mask, track_core, track_corridor, _ = build_track_core(hazard_mask)
    hazard, hazard_labels = detect_track_intrusion(hazard_mask, scores, conf_thresh)
    
    # VIDEO-SPECIFIC: Apply stricter CLIP threshold for video frames
    clip_threshold = 0.85 if is_video else CLIP_HAZARD_THRESHOLD
    clip_vote, clip_text = evaluate_clip_vote(clip_result, track_present, threshold=clip_threshold)
    
    # CRITICAL: Validate CLIP claims against actual segmentation evidence
    if clip_vote and track_present:
        clip_conf = clip_result.get("confidence", 0) if clip_result else 0
        clip_label = clip_result.get("label", "").lower() if clip_result else ""
        
        # VIDEO: Even stricter validation for video frames (distant objects, perspective issues)
        min_object_threshold = MIN_ONTRACK_PIXELS_OBJECT * 2 if is_video else MIN_ONTRACK_PIXELS_OBJECT
        min_corridor_threshold = MIN_ONTRACK_PIXELS * 2 if is_video else MIN_ONTRACK_PIXELS
        
        # If CLIP claims object/hazard blocking track, verify with segmentation
        if any(word in clip_label for word in ["object", "blocking", "vehicle", "car", "people", "tree", "debris"]):
            # Check if there's actually object/vegetation on the track core
            object_on_core = ((hazard_mask == OBJECT_CLASS_ID) | (hazard_mask == VEGETATION_CLASS_ID)) & track_core
            object_pixels = object_on_core.sum()
            
            # Also check corridor for more lenient detection
            object_on_corridor = ((hazard_mask == OBJECT_CLASS_ID) | (hazard_mask == VEGETATION_CLASS_ID)) & track_corridor
            corridor_pixels = object_on_corridor.sum()
            
            # If no significant object presence detected by segmentation, CLIP is wrong
            if object_pixels < min_object_threshold and corridor_pixels < min_corridor_threshold:
                clip_vote = False
                clip_text = ""
        
        # Additional check: if track is very visible and CLIP confidence not extremely high, be skeptical
        track_pixels = (hazard_mask == TRACK_CLASS_ID).sum()
        confidence_threshold = 0.95 if is_video else 0.90  # Higher bar for video
        if track_pixels > 8000 and clip_conf < confidence_threshold:
            clip_vote = False
            clip_text = ""

    track_band = compute_track_band(mask)
    snow_coverage = compute_snow_coverage(image, track_band)
    snow_level = None
    if snow_coverage >= SNOW_HEAVY_COVERAGE:
        snow_level = "heavy"
    elif snow_coverage >= SNOW_CAUTION_COVERAGE:
        snow_level = "light"

    if clip_text and "snow" in clip_text.lower() and snow_level != "heavy":
        clip_vote = False
        clip_text = ""
    if snow_level == "heavy":
        clip_vote = True
        clip_text = f"heavy snow blocking the track ({snow_coverage*100:.0f}%)"

    scenario_str = None
    if weather_row:
        suitability = check_track_suitability(
            weather_row.get("scenario"),
            weather_row.get("brightness"),
            weather_row.get("contrast"),
            weather_row.get("edge_density")
        )
        scenario_str = (weather_row.get("scenario") or "").lower()
    else:
        b, c, e = compute_simple_metrics(image)
        suitability = check_track_suitability(
            scenario=None,
            brightness=b,
            contrast=c,
            edge_density=e
        )
        scenario_str = None

    # Optional: ignore CLIP hazard vote on snowy scenes (weather classifier)
    if scenario_str and "snow" in scenario_str and snow_level != "heavy":
        clip_vote, clip_text = False, ""

    # Use raw predicted track mask to avoid under-reporting visibility.
    horizon = compute_horizon(mask == TRACK_CLASS_ID)
    risks = compute_active_risks(hazard_mask, track_corridor)
    rail_conf = compute_rail_confidence(filt_mask, scores, conf_thresh)

    # Major overlap override: if large vegetation/object on core track band, force STOP
    core_area = track_core.sum()
    major_blocker = False
    if core_area > 0:
        blocker_mask = ((hazard_mask == VEGETATION_CLASS_ID) | (hazard_mask == OBJECT_CLASS_ID)) & track_core
        blocker_pixels = blocker_mask.sum()
        blocker_ratio = blocker_pixels / core_area if core_area else 0.0
        if blocker_pixels >= MIN_MAJOR_OVERLAP_PIXELS and blocker_ratio >= MIN_MAJOR_OVERLAP_RATIO:
            major_blocker = True

    status = suitability["status"]
    reason = suitability["reason"]
    if major_blocker or hazard or clip_vote:
        status = "STOP"
        if snow_level == "heavy":
            reason = "Heavy snow blocking the track; clearance required"
        else:
            reason = "Track blocked/hazard detected; requires clearance"
    elif snow_level == "light" and status != "STOP":
        if status == "GO":
            status = "CAUTION"
        reason = "Proceed with caution (light snow on track)"
    # Override visibility-based STOP if track is clear with high rail confidence
    elif status == "STOP" and not hazard and not major_blocker and rail_conf >= 0.85:
        # Track is clear and rail detection is confident - override visibility STOP
        is_night_scene = scenario_str and any(term in scenario_str.lower() for term in ["night", "evening", "dusk", "dim"])
        if is_night_scene:
            status = "GO"
            reason = "Track clear with good detection confidence (train lights enable safe operation)"
        else:
            status = "CAUTION"
            reason = "Drive with caution (low visibility but track clear)"
    elif status == "CAUTION":
        # Upgrade CAUTION→GO for night/evening scenes if track is clear and rail confidence is good
        if not hazard and not major_blocker and rail_conf >= 0.90:
            is_night_scene = scenario_str and any(term in scenario_str.lower() for term in ["night", "evening", "dusk", "dim"])
            if is_night_scene:
                status = "GO"
                reason = "Track clear with good detection confidence (train lights enable safe operation)"
            else:
                reason = "Drive with caution (visibility marginal)"
        else:
            reason = "Drive with caution (visibility marginal)"

    return {
        "filtered_mask": filt_mask,
        "hazard_mask": hazard_mask,
        "hazard": hazard,
        "hazard_labels": hazard_labels,
        "clip_vote": clip_vote,
        "clip_text": clip_text,
        "status": status,
        "reason": reason,
        "horizon": horizon,
        "risks": risks,
        "rail_conf": rail_conf,
        "major_blocker": major_blocker,
        "snow_coverage": snow_coverage
    }

def display_kpi_row(analysis, current_availability=None):
    """Display KPI row as simple cards."""
    status = analysis["status"]
    horizon = analysis["horizon"]
    risks = analysis["risks"]
    rail_conf = analysis["rail_conf"]
    clip_flag = analysis["clip_vote"]
    risk_total = risks + (1 if clip_flag else 0)

    def status_pill(text, tone):
        return f"<span class='pill {tone}'>{text}</span>"

    def tone_for(value, go_thr, caution_thr=None):
        if caution_thr is None:
            return "green" if value else "red"
        if value >= go_thr:
            return "green"
        if value >= caution_thr:
            return "amber"
        return "red"

    status_tone = "green" if status == "GO" else ("amber" if status == "CAUTION" else "red")
    horizon_tone = tone_for(horizon, HORIZON_GO, HORIZON_CAUTION)
    risk_tone = "red" if risk_total > 0 else "green"
    rail_tone = "green" if rail_conf >= 0.6 else ("amber" if rail_conf >= 0.4 else "red")
    availability_tone = "green" if (current_availability is not None and current_availability >= 0.8) else ("amber" if current_availability else "red")

    cards = [
        {"title": "🚦 Operational Status", "value": status, "sub": analysis["reason"], "tone": status_tone},
        {"title": "👁️ Visibility Horizon", "value": f"{horizon*100:.0f}%", "sub": "Clear view" if horizon >= HORIZON_GO else ("Low visibility" if horizon < HORIZON_CAUTION else "Marginal"), "tone": horizon_tone},
        {"title": "⚠️ Active Hazards", "value": risk_total, "sub": f"{risks} on/near track" if risks else ("Environmental hazard" if clip_flag else "Track clear"), "tone": risk_tone},
        {"title": "🛡️ Rail Confidence", "value": f"{rail_conf:.2f}", "sub": "Healthy" if rail_conf >= 0.6 else "Lower confidence", "tone": rail_tone},
        {"title": "📈 Track Availability", "value": f"{current_availability*100:.1f}%" if current_availability is not None else "--", "sub": "Session GO rate" if current_availability is not None else "Awaiting data", "tone": availability_tone if current_availability is not None else "amber"},
    ]

    blocks = ["<div class='kpi-grid'>"]
    for c in cards:
        blocks.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-title'>{c['title']}</div>"
            f"<div class='kpi-value'>{c['value']}</div>"
            f"<div class='kpi-sub'>{status_pill(c['sub'], c['tone'])}</div>"
            "</div>"
        )
    blocks.append("</div>")
    st.markdown("\n".join(blocks), unsafe_allow_html=True)

def compute_session_availability(current_file_names, conf_thresh):
    """Compute GO ratio across current items."""
    if not current_file_names:
        return None
    go = 0
    total = 0
    for fname in current_file_names:
        data = st.session_state.batch_results.get(fname)
        if not data:
            continue
        weather_row = weather_lookup.get(os.path.basename(fname))
        is_vid = is_video_frame(fname)
        analysis = analyze_frame(
            data["image"],
            data["pred_mask"],
            data["confidence_scores"],
            conf_thresh,
            data.get("clip_hazard"),
            weather_row,
            is_video=is_vid
        )
        total += 1
        if analysis["status"] == "GO":
            go += 1
    if total == 0:
        return None
    return go / total

def render_result_view(img, mask, scores, conf_thresh, alpha, clip_result=None, weather_row=None, availability=None, filename=None):
    """
    Helper to render the two-column view inside a placeholder.
    Note: For manual view, we generate overlay on the fly.
    """
    # Apply filtering and compute analysis
    is_vid = is_video_frame(filename) if filename else False
    analysis = analyze_frame(img, mask, scores, conf_thresh, clip_result, weather_row, is_video=is_vid)
    if availability is not None:
        analysis["availability"] = availability
    else:
        analysis["availability"] = None
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image / Video")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("🎯 Segmentation Overlay")
        ov = create_overlay(img, analysis["filtered_mask"], alpha, CLASS_COLORS_NORMALIZED)
        st.image(ov, use_container_width=True)
    
    # KPIs row
    display_kpi_row(analysis, current_availability=analysis.get("availability", None))

    # Alerts and Status
    status = analysis["status"]
    reason = analysis["reason"]
    snow_coverage = analysis.get("snow_coverage", 0)
    
    # Always show snow alert if heavy snow detected
    if snow_coverage >= SNOW_HEAVY_COVERAGE:
        st.error(f"⚠️ Alert: heavy snow blocking the track ({snow_coverage*100:.1f}%)")
    elif analysis["hazard"] or analysis["clip_vote"]:
        messages = []
        if analysis["hazard"]:
            classes_text = ", ".join(analysis["hazard_labels"])
            messages.append(f"{classes_text} on/near the track")
        if analysis["clip_vote"] and snow_coverage < SNOW_HEAVY_COVERAGE:
            messages.append(analysis["clip_text"])
        if messages:
            st.error("⚠️ Alert: " + " | ".join(messages))

    if status == "GO":
        st.success(f"🟢 GO: {reason}")
    elif status == "CAUTION":
        st.warning(f"🟡 CAUTION: {reason}")
    else:
        st.error(f"🔴 STOP: {reason}")

    st.divider()
    display_legend_badges(analysis["filtered_mask"], CLASS_COLORS_RGB, FINAL_CLASS_NAMES)

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

def build_track_core(mask: np.ndarray):
    """Return (track_mask, track_core, track_corridor, track_vicinity) with fallback if no track pixels are present."""
    h, w = mask.shape
    track_mask = mask == TRACK_CLASS_ID
    band_half_width = max(int(w * 0.04), 3)  # Tighter ~8% band (was 6%) for more precise core

    if np.any(track_mask):
        x_coords = np.where(track_mask)[1]
        center_x = x_coords.mean() if len(x_coords) else w / 2
        min_y = np.where(track_mask)[0].min()
        band_start = min(min_y, int(h * TRACK_CORRIDOR_TOP_FRAC))
    else:
        center_x = w / 2
        band_start = int(h * TRACK_CORRIDOR_TOP_FRAC)

    x_grid = np.arange(w)
    core_cols = np.abs(x_grid - center_x) <= band_half_width
    band_rows = np.arange(h) >= band_start
    track_corridor = np.tile(core_cols, (h, 1)) & band_rows[:, None]

    # Fallback: if no track predicted, assume center band is the track corridor
    if not np.any(track_mask):
        track_core = track_corridor
        track_vicinity = binary_dilation(track_core, iterations=max(TRACK_VICINITY_DILATION, 1))
        return track_mask, track_core, track_corridor, track_vicinity

    track_vicinity = binary_dilation(track_mask, iterations=TRACK_VICINITY_DILATION)
    track_core = track_mask & core_cols
    return track_mask, track_core, track_corridor, track_vicinity

def detect_track_intrusion(mask: np.ndarray, confidence_scores: np.ndarray, conf_threshold: float):
    """
    Detect if vegetation or objects are actually ON the track core (not just nearby).
    """
    _, track_core, track_corridor, _ = build_track_core(mask)

    hazard_classes = []
    relaxed_thresh = max(conf_threshold * 0.5, 0.2)
    conf_mask = confidence_scores >= relaxed_thresh

    for cls_id in (VEGETATION_CLASS_ID, OBJECT_CLASS_ID):
        class_mask = mask == cls_id
        # Only check core track overlap - not corridor (corridor is too permissive)
        overlap_track = track_core & class_mask & conf_mask
        overlap_track_pixels = overlap_track.sum()
        
        # Calculate ratio of track core that's blocked
        core_area = track_core.sum()
        core_ratio = overlap_track_pixels / core_area if core_area > 0 else 0.0
        
        min_ontrack = MIN_ONTRACK_PIXELS_OBJECT if cls_id == OBJECT_CLASS_ID else MIN_ONTRACK_PIXELS
        # Require BOTH pixel count AND ratio thresholds (stricter)
        if overlap_track_pixels >= min_ontrack and core_ratio >= MIN_INTRUSION_RATIO:
            hazard_classes.append(FINAL_CLASS_NAMES.get(cls_id, f"Class {cls_id}"))

    return len(hazard_classes) > 0, hazard_classes

def evaluate_clip_vote(clip_result, track_present: bool, threshold: float = CLIP_HAZARD_THRESHOLD):
    """
    Determine if CLIP predicts a hazard with enough confidence to count as a vote.
    Only flag actual BLOCKING hazards, not vegetation near the track.
    """
    if not clip_result or not track_present:
        return False, ""

    label = clip_result.get("label", "").lower()
    confidence = clip_result.get("confidence", 0.0)

    # Clear track scenarios are NOT hazards
    harmless_phrases = {
        "clear railway track", "clear track", "empty track",
        "no obstruction", "no obstacle", "vegetation on the sides",
        "clear railway track with vegetation"
    }
    if any(phrase in label for phrase in harmless_phrases):
        return False, ""

    # Must explicitly mention BLOCKING, not just presence
    hazard_like = {
        "blocking", "blocked", "obstruction", "completely blocked",
        "fallen across", "tree trunk blocking", "debris blocking",
        "object directly blocking", "water covering rails",
        "landslide debris", "heavy snow completely covering",
        "train directly ahead", "overgrown blocking",
        "people standing on", "vehicle stopped on"
    }
    hazard_match = any(h in label for h in hazard_like)

    if confidence < threshold or not hazard_match:
        return False, ""

    return True, f"{clip_result['label']} ({confidence:.2f})"

# --- 7. Main Logic ---

if app_mode == "Image":
    st.subheader("🖼️ Image Mode")
    st.markdown("Choose images...")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["jpg", "jpeg", "png"],
        help="Limit 200MB per file • JPG, JPEG, PNG",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Ensure when switching to image, the video variable is empty
    st.session_state.video_frames = []

elif app_mode == "Video":
    st.subheader("🎬 Video Mode")
    st.markdown("Choose video...")
    video_file = st.file_uploader(
        "Drag and drop file here",
        type=["mov", "mp4", "mkv", "avi", "mpeg4"],
        help="Limit 200MB per file • MP4, MOV, MKV, AVI, MPEG4",
        label_visibility="collapsed"
    )
    
    video_process = st.button("🎬 Process Video", type="primary", use_container_width=True)
    
    # Ensure when switching to video, the image variable is empty
    uploaded_files = None
    
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
                    
                    # Rail inference
                    image_tensor = preprocess_image(image, processor)
                    pred_mask_upscaled, confidence_scores = run_inference_and_upscale(
                        image, image_tensor, best_model, DEVICE, class_weights=RAIL_CLASS_WEIGHTS_BIASED
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
    with st.sidebar:
        st.divider()
        st.header("🎮 Playback Controls")
        
        # Selector for manual view
        selected_file_name = st.selectbox(
            "📂 Select Image to View",
            options=current_file_names,
            index=0
        )
        
        # Slideshow toggle button
        start_slideshow = st.button(
            "▶️ Play Slideshow", 
            disabled=len(current_file_names) < 2,
            use_container_width=True
        )

    # --- DISPLAY PHASE ---
    main_display = st.empty()

    if start_slideshow:
        # --- SLIDESHOW MODE ---
        # 1. Pre-calculation Phase (Crucial for smooth 0.5s transitions)
        slideshow_data = []
        progress_bar = st.progress(0, text="⏳ Pre-rendering frames for smooth playback...")
        
        availability = compute_session_availability(current_file_names, confidence_threshold)

        for i, fname in enumerate(current_file_names):
            if fname in st.session_state.batch_results:
                data = st.session_state.batch_results[fname]
                
                # Pre-calculate analysis so the loop is instant
                img = data["image"]
                mask = data["pred_mask"]
                scores = data["confidence_scores"]
                clip_result = st.session_state.batch_results[fname].get("clip_hazard")
                weather_row = weather_lookup.get(os.path.basename(fname))
                analysis = analyze_frame(img, mask, scores, confidence_threshold, clip_result, weather_row)
                analysis["availability"] = availability
                final_overlay = create_overlay(img, analysis["filtered_mask"], overlay_alpha, CLASS_COLORS_NORMALIZED)

                slideshow_data.append({
                    "name": fname,
                    "original": img,
                    "overlay": final_overlay,
                    "analysis": analysis
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

                analysis = slide["analysis"]
                display_kpi_row(analysis, current_availability=analysis.get("availability"))

                if analysis["hazard"] or analysis["clip_vote"]:
                    messages = []
                    if analysis["hazard"]:
                        classes_text = ", ".join(analysis["hazard_labels"])
                        messages.append(f"{classes_text} on/near the track")
                    if analysis["clip_vote"]:
                        messages.append(analysis["clip_text"])
                    st.error("⚠️ Alert: " + " | ".join(messages))

                # Enforce STOP if any hazard/clip vote is detected, regardless of visibility
                if analysis["hazard"] or analysis["clip_vote"]:
                    st.error("🔴 STOP: Track blocked/hazard detected; requires clearance")
                else:
                    status = analysis["status"]
                    reason = analysis["reason"]
                    if status == "GO":
                        st.success(f"🟢 GO: {reason}")
                    elif status == "CAUTION":
                        st.warning(f"🟡 CAUTION: {reason}")
                    else:
                        st.error(f"🔴 STOP: {reason}")

                st.divider()
                display_legend_badges(analysis['filtered_mask'], CLASS_COLORS_RGB, FINAL_CLASS_NAMES)
            
            # The pause duration (default 0.5s)
            time.sleep(slideshow_speed)
        
        st.success("Slideshow finished!")
        time.sleep(1)
        st.rerun()

    else:
        # --- MANUAL MODE ---
        if selected_file_name and selected_file_name in st.session_state.batch_results:
            availability = compute_session_availability(current_file_names, confidence_threshold)
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
                    weather_row,
                    availability=availability,
                    filename=selected_file_name
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
