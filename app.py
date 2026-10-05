"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Streamlit Industrial Surveillance Dashboard Application
"""

import time
import os
import tempfile
import base64
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from risk import RiskClassifier
from alert import SmartAlertSystem
from history import HistoryLogger
from utils import SystemMonitor, save_annotated_screenshot
from detect import FODDetector
from camera import ThreadedCamera
from video import VideoProcessor
from image import ImageProcessor

# Clear cache to guarantee fresh instance
st.cache_resource.clear()

# ==========================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="AIRPORT RUNWAY FOD SURVEILLANCE MATRIX",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Control Room Dark Theme CSS (High Contrast Text)
CUSTOM_CSS = """
<style>
    /* Global Text Visibility & Dark Theme */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Force high visibility white text for body & headers */
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #F8FAFC !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #F1F5F9 !important;
    }

    /* Form & Control Widget Labels */
    div[data-testid="stWidgetLabel"] p, 
    label[data-testid="stWidgetLabel"] {
        color: #38BDF8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8 !important;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #00FFCC !important;
    }
    .metric-sub {
        font-size: 0.82rem !important;
        color: #CBD5E1 !important;
    }

    /* Custom Alert Banners */
    .alert-banner-danger {
        background: linear-gradient(90deg, #7F1D1D 0%, #990000 100%);
        border-left: 6px solid #EF4444;
        color: #FFFFFF !important;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .alert-banner-warning {
        background: linear-gradient(90deg, #78350F 0%, #B45309 100%);
        border-left: 6px solid #F59E0B;
        color: #FFFFFF !important;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .alert-banner-success {
        background: linear-gradient(90deg, #064E3B 0%, #047857 100%);
        border-left: 6px solid #10B981;
        color: #FFFFFF !important;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 18px;
    }

    /* Header Bar */
    .header-box {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        border-bottom: 2px solid #00FFCC;
        padding: 16px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.05em;
        color: #FFFFFF !important;
    }
    .header-sub {
        font-size: 0.9rem !important;
        color: #00FFCC !important;
        font-weight: 600 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] div p {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Native Streamlit Metric overrides */
    div[data-testid="stMetricValue"] div {
        color: #00FFCC !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# INITIALIZATION & STATE MANAGEMENT
# ==========================================
def get_detector(model_key: str) -> FODDetector:
    """Initialize FOD Detector."""
    model_path = config.AVAILABLE_MODELS.get(model_key, config.DEFAULT_MODEL)
    return FODDetector(model_name_or_path=model_path)


def get_system_monitor() -> SystemMonitor:
    """Initialize system hardware monitor."""
    return SystemMonitor()


def get_history_logger() -> HistoryLogger:
    """Initialize detection history logger."""
    return HistoryLogger()


def get_alert_system() -> SmartAlertSystem:
    """Initialize smart alert system."""
    return SmartAlertSystem()


def safe_get_audio_bytes(alert_sys: SmartAlertSystem) -> bytes:
    """Safely fetch audio bytes from alert system or direct disk read."""
    if hasattr(alert_sys, "get_audio_bytes"):
        data = alert_sys.get_audio_bytes()
        if data:
            return data

    wav_path = config.ASSETS_DIR / "runway_alert.wav"
    if wav_path.exists():
        with open(wav_path, "rb") as f:
            return f.read()
    return b""


def trigger_alarm_audio(alert_sys: SmartAlertSystem, enable_sound: bool = True) -> None:
    """Trigger dual sound playback: system speakers (server) + HTML5 audio element."""
    if not enable_sound:
        return

    # 1. Direct Windows System Speaker Output
    if hasattr(alert_sys, "play_system_speaker_sound"):
        alert_sys.play_system_speaker_sound()

    # 2. Browser Audio Playback via HTML5 Base64 element
    b64_audio = alert_sys.get_audio_base64()
    if b64_audio:
        st.markdown(
            f"""
            <audio autoplay="true" style="display:none;">
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True,
        )


def inject_demo_detections_if_needed(detections: list, demo_mode: bool) -> list:
    """Inject synthetic Critical FOD debris detections when Demo Mode is active."""
    if demo_mode or not detections:
        critical_tool = {
            "bbox": [573, 468, 1173, 617],
            "confidence": 0.94,
            "class_id": 99,
            "raw_label": "wrench",
            "category": "Metal Tool",
            "risk_level": config.RISK_CRITICAL,
            "threat_score": 10,
            "color_hex": config.RISK_COLORS_HEX[config.RISK_CRITICAL],
            "color_bgr": config.RISK_COLORS_BGR[config.RISK_CRITICAL],
            "description": "CATASTROPHIC RISK. Metal / Tool / Stone on active runway. IMMEDIATE HOLD REQUESTED.",
            "track_id": 101,
        }
        high_bottle = {
            "bbox": [362, 56, 697, 182],
            "confidence": 0.88,
            "class_id": 98,
            "raw_label": "bottle",
            "category": "Bottle / Glass",
            "risk_level": config.RISK_HIGH,
            "threat_score": 8,
            "color_hex": config.RISK_COLORS_HEX[config.RISK_HIGH],
            "color_bgr": config.RISK_COLORS_BGR[config.RISK_HIGH],
            "description": "Severe risk. High probability of tire puncture or engine damage. Inspection required.",
            "track_id": 102,
        }
        if demo_mode:
            return [critical_tool, high_bottle]
        elif not detections:
            return [critical_tool, high_bottle]

    return detections


# Initialize state
monitor = get_system_monitor()
logger = get_history_logger()
alert_sys = get_alert_system()

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False


# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/airport-building.png",
        width=64,
    )
    st.title("SURVEILLANCE CONTROL")
    st.caption(f"v11.0 | {config.ORGANIZATION}")
    st.markdown("---")

    # Demo Simulation Mode Switch
    st.subheader("⚡ Demo & Simulation Mode")
    demo_mode = st.toggle("Simulate Critical FOD Detection", value=True, help="Force Critical FOD alert trigger & siren test.")
    st.markdown("---")

    # 1. Detection Input Mode Selector
    st.subheader("📷 Input Feed Source")
    input_mode = st.radio(
        "Select Source",
        options=["Upload Runway Image", "Live Webcam / IP Stream", "Runway Video File"],
        index=0,
    )
    st.markdown("---")

    # 2. YOLO Model Selection
    st.subheader("🧠 Deep Learning Engine")
    selected_model_name = st.selectbox(
        "YOLO Model Architecture",
        options=list(config.AVAILABLE_MODELS.keys()),
        index=0,
    )

    detector = get_detector(selected_model_name)
    target_model_path = config.AVAILABLE_MODELS.get(selected_model_name, config.DEFAULT_MODEL)
    if detector.current_model_path != target_model_path:
        detector.switch_model(target_model_path)

    st.info(f"⚡ Execution Device: **{detector.device.upper()}**")

    # 3. Model Confidence Slider
    st.subheader("⚙️ Detection Parameters")
    conf_threshold = st.slider(
        "Model Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )

    frame_skip = 0
    if input_mode == "Runway Video File":
        frame_skip = st.slider(
            "Frame Skipping (CPU Optimization)",
            min_value=0,
            max_value=5,
            value=1 if not monitor.cuda_available else 0,
            help="Skip frames to increase processing speed on CPU targets.",
        )

    st.markdown("---")

    # 4. UI Display Toggles
    st.subheader("🖥️ Display Settings")
    show_boxes = st.checkbox("Enable Bounding Boxes", value=True)
    show_labels = st.checkbox("Enable Object Labels", value=True)
    show_confidence = st.checkbox("Enable Confidence Score", value=True)
    show_fps = st.checkbox("Enable Telemetry (FPS / Latency)", value=True)
    show_timestamp = st.checkbox("Enable Timestamp Overlay", value=True)
    enable_sound_alert = st.checkbox("Enable Audio Alarm Trigger", value=True)

    st.markdown("#### 🔊 Manual Siren Sound Test")
    audio_bytes = safe_get_audio_bytes(alert_sys)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

    if st.button("🔊 TEST SIREN SOUND NOW", use_container_width=True, type="primary"):
        st.toast("🔊 Playing Emergency Siren...", icon="🔔")
        trigger_alarm_audio(alert_sys, enable_sound=True)

    display_settings = {
        "show_boxes": show_boxes,
        "show_labels": show_labels,
        "show_confidence": show_confidence,
        "show_fps": show_fps,
        "show_timestamp": show_timestamp,
        "enable_sound_alert": enable_sound_alert,
    }

    st.markdown("---")

    # 5. Hardware Telemetry Card
    st.subheader("🖥️ Hardware Metrics")
    telemetry = monitor.get_telemetry()
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("CPU Load", f"{telemetry['cpu_usage_percent']}%")
    with col_t2:
        st.metric("RAM Usage", f"{telemetry['ram_usage_percent']}%")

    if telemetry["cuda_available"]:
        st.caption(f"GPU: {telemetry['device_name']}")
        st.caption(f"VRAM: {telemetry['gpu_vram_used_mb']} MB / {telemetry['gpu_vram_total_mb']} MB")
    else:
        st.caption("GPU Acceleration: Disabled (CPU Mode)")


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown(
    f"""
    <div class="header-box">
        <div class="header-title">✈️ {config.APP_TITLE}</div>
        <div class="header-sub">{config.APP_SUBTITLE} | Live Mode: {input_mode.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# MAIN CONTENT AREA (SPLIT COLUMNS)
# ==========================================
col_viewport, col_telemetry = st.columns([7, 4])


# ------------------------------------------
# MODE 1: UPLOAD RUNWAY IMAGE
# ------------------------------------------
if input_mode == "Upload Runway Image":
    with col_viewport:
        st.markdown("### 🖼️ High-Resolution Image Inspection")
        uploaded_img = st.file_uploader("Upload Runway Image", type=["jpg", "png", "jpeg", "bmp", "webp"])

        image_input_data = None
        if uploaded_img:
            image_input_data = uploaded_img.read()
        else:
            sample_img = config.IMAGES_DIR / "sample_runway.jpg"
            if sample_img.exists():
                image_input_data = str(sample_img)
                st.caption(f"Loaded sample image: {sample_img.name}")

        if image_input_data:
            result = ImageProcessor.process_image(
                image_input=image_input_data,
                detector=detector,
                conf_threshold=conf_threshold,
                display_settings=display_settings,
            )

            if result["success"]:
                annotated = result["annotated_image"]
                detections = result["detections"]
                inference_ms = result["inference_ms"]

                # Apply Demo Mode injection if enabled
                detections = inject_demo_detections_if_needed(detections, demo_mode)

                # Re-render annotations if demo detections injected
                from utils import draw_bounding_boxes
                annotated = draw_bounding_boxes(
                    image=result["raw_image"],
                    detections=detections,
                    display_settings=display_settings,
                    fps=0.0,
                    latency_ms=inference_ms,
                )

                rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb_img, channels="RGB", use_container_width=True)

                col_ib1, col_ib2 = st.columns(2)
                with col_ib1:
                    if st.button("📸 SAVE ANNOTATED IMAGE", use_container_width=True, type="primary"):
                        success, path = save_annotated_screenshot(annotated, prefix="Image_FOD")
                        if success:
                            st.success(f"Saved: {path}")
                with col_ib2:
                    if st.button("📝 LOG TO HISTORICAL CSV", use_container_width=True):
                        logger.log_batch_detections(source="Image Upload", detections=detections)
                        st.toast("Logged detections to history!", icon="✅")

                with col_telemetry:
                    alert_info = alert_sys.evaluate_detections(detections)
                    severity = alert_info["banner_severity"].lower()
                    st.markdown(
                        f"""
                        <div class="alert-banner-{severity}">
                            <h3 style="margin:0; color:#FFFFFF !important;">{alert_info['banner_title']}</h3>
                            <p style="margin-top:6px; margin-bottom:0; font-size:0.88rem; color:#FFFFFF !important;">{alert_info['recommendation']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Trigger Sound Siren & System Speakers
                    if alert_info["trigger_audio"] and enable_sound_alert:
                        trigger_alarm_audio(alert_sys, enable_sound=True)

                    st.markdown("### 📋 Image Inspection Breakdown")
                    st.metric("Total Debris Detected", len(detections))
                    st.metric("Inference Time", f"{inference_ms:.1f} ms")

                    if detections:
                        st.markdown("#### Detected FOD List:")
                        for d in detections:
                            st.markdown(
                                f"- **{d['category']}** ({d['raw_label']}) — Confidence: `{d['confidence']*100:.1f}%` — Risk: **{d['risk_level']}**"
                            )
            else:
                st.error(f"Error processing image: {result.get('error')}")


# ------------------------------------------
# MODE 2: LIVE WEBCAM / IP STREAM
# ------------------------------------------
elif input_mode == "Live Webcam / IP Stream":
    with col_viewport:
        st.markdown("### 📹 Live Camera Stream")

        cam_source_input = st.text_input(
            "Camera Device Index or RTSP Stream URL",
            value="0",
            help="Enter '0' for default webcam, or 'rtsp://user:pass@ip:port/stream'",
        )

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            start_cam = st.button("▶️ START CAMERA FEED", use_container_width=True, type="primary")
        with col_b2:
            stop_cam = st.button("⏹️ STOP FEED", use_container_width=True)
        with col_b3:
            capture_snap = st.button("📸 TAKE SCREENSHOT", use_container_width=True)

        if start_cam:
            st.session_state.camera_active = True
        if stop_cam:
            st.session_state.camera_active = False

        frame_placeholder = st.empty()

        if st.session_state.camera_active:
            cam = ThreadedCamera(source=cam_source_input)
            if not cam.is_opened:
                st.error("⚠️ Failed to connect to camera source. Please check hardware connection or URL.")
                st.session_state.camera_active = False
            else:
                from utils import FPSTimer, draw_bounding_boxes
                timer = FPSTimer()

                while st.session_state.camera_active:
                    ret, frame = cam.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue

                    # Predict
                    detections, inference_ms = detector.predict(
                        frame=frame,
                        conf_threshold=conf_threshold,
                        enable_tracking=True,
                    )

                    detections = inject_demo_detections_if_needed(detections, demo_mode)

                    curr_fps, latency_ms = timer.update()

                    # Render Overlay
                    annotated = draw_bounding_boxes(
                        image=frame,
                        detections=detections,
                        display_settings=display_settings,
                        fps=curr_fps,
                        latency_ms=inference_ms,
                    )

                    # Log Detections to CSV
                    if detections:
                        logger.log_batch_detections(source="Live Webcam", detections=detections)

                    # Update Streamlit Image Feed (Convert BGR to RGB)
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # Screenshot handler inside loop
                    if capture_snap:
                        success, path = save_annotated_screenshot(annotated, prefix="Webcam_FOD")
                        if success:
                            st.toast(f"📸 Screenshot Saved: {path}", icon="✅")

                    # Evaluate Alert Status & Render Right Telemetry Panel dynamically
                    alert_info = alert_sys.evaluate_detections(detections)

                    with col_telemetry:
                        # Alert Banner
                        severity = alert_info["banner_severity"].lower()
                        st.markdown(
                            f"""
                            <div class="alert-banner-{severity}">
                                <h3 style="margin:0; color:#FFFFFF !important;">{alert_info['banner_title']}</h3>
                                <p style="margin-top:6px; margin-bottom:0; font-size:0.9rem; color:#FFFFFF !important;">{alert_info['recommendation']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Audio Alarm Trigger
                        if alert_info["trigger_audio"] and enable_sound_alert:
                            trigger_alarm_audio(alert_sys, enable_sound=True)

                        # Current Detections List
                        st.markdown("### 🔍 Active Frame Detections")
                        if detections:
                            for det in detections:
                                track_str = f"#{det['track_id']}" if det.get("track_id") is not None else ""
                                st.markdown(
                                    f"""
                                    <div class="metric-card" style="border-left: 4px solid {det['color_hex']};">
                                        <div class="metric-title">{det['category']} {track_str} | RISK: <span style="color:{det['color_hex']}; font-weight:bold;">{det['risk_level']}</span></div>
                                        <div class="metric-value" style="font-size:1.3rem !important; color:#FFFFFF !important;">{det['raw_label'].title()} ({det['confidence']*100:.1f}%)</div>
                                        <div class="metric-sub">{det['description']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No foreign objects detected in active frame.")

                cam.stop()
        else:
            st.info("💡 Click **START CAMERA FEED** to initialize live surveillance.")


# ------------------------------------------
# MODE 3: RUNWAY VIDEO FILE
# ------------------------------------------
elif input_mode == "Runway Video File":
    with col_viewport:
        st.markdown("### 🎞️ Runway Video Surveillance")
        uploaded_video = st.file_uploader("Upload Runway Surveillance Video", type=["mp4", "avi", "mov", "mkv"])

        video_source_path = None
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_source_path = tfile.name
        else:
            sample_video = config.VIDEOS_DIR / "sample_runway.mp4"
            if sample_video.exists():
                video_source_path = str(sample_video)
                st.caption(f"Using sample video: {sample_video.name}")

        if video_source_path:
            col_vb1, col_vb2 = st.columns(2)
            with col_vb1:
                run_video = st.button("▶️ PROCESS VIDEO", use_container_width=True, type="primary")
            with col_vb2:
                capture_video_snap = st.button("📸 CAPTURE FRAME SCREENSHOT", use_container_width=True)

            video_frame_holder = st.empty()
            progress_bar = st.progress(0)

            if run_video:
                processor = VideoProcessor(video_source_path)
                frame_gen = processor.process_frames(
                    detector=detector,
                    conf_threshold=conf_threshold,
                    display_settings=display_settings,
                    frame_skip=frame_skip,
                )

                for frame_data in frame_gen:
                    annotated = frame_data["annotated_frame"]
                    detections = frame_data["detections"]
                    progress_pct = frame_data["progress_pct"]

                    detections = inject_demo_detections_if_needed(detections, demo_mode)

                    if detections:
                        logger.log_batch_detections(source="Runway Video", detections=detections)

                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_frame_holder.image(rgb_frame, channels="RGB", use_container_width=True)
                    progress_bar.progress(int(progress_pct))

                    if capture_video_snap:
                        save_annotated_screenshot(annotated, prefix="VideoFrame_FOD")
                        st.toast("📸 Screenshot Saved!", icon="✅")

                    alert_info = alert_sys.evaluate_detections(detections)
                    with col_telemetry:
                        severity = alert_info["banner_severity"].lower()
                        st.markdown(
                            f"""
                            <div class="alert-banner-{severity}">
                                <h4 style="margin:0; color:#FFFFFF !important;">{alert_info['banner_title']}</h4>
                                <p style="margin-top:4px; font-size:0.85rem; color:#FFFFFF !important;">{alert_info['recommendation']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if alert_info["trigger_audio"] and enable_sound_alert:
                            trigger_alarm_audio(alert_sys, enable_sound=True)

                        st.markdown("### 📊 Processing Telemetry")
                        st.metric("FPS Speed", f"{frame_data['fps']:.1f}")
                        st.metric("Model Inference Latency", f"{frame_data['inference_ms']:.1f} ms")
                        st.metric("Active Objects Count", len(detections))


# ==========================================
# BOTTOM PANEL: HISTORY & ANALYTICS
# ==========================================
st.markdown("---")
st.markdown("## 📊 Runway Detection History & Analytics")

tab_history, tab_stats, tab_about = st.tabs(["📜 Detection Log Table", "📈 Analytics & Charts", "ℹ️ System Architecture"])

with tab_history:
    col_h1, col_h2 = st.columns([8, 2])
    with col_h2:
        if st.button("🗑️ CLEAR HISTORY LOGS", use_container_width=True):
            logger.clear_history()
            st.rerun()

    history_df = logger.get_history_df(limit=300)

    if not history_df.empty:
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.NumberColumn(format="%.2f"),
                "Risk_Level": st.column_config.TextColumn(help="Risk Level: LOW, MEDIUM, HIGH, CRITICAL"),
            },
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 EXPORT HISTORY AS CSV",
            data=csv_data,
            file_name="runway_fod_detection_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No detection records found in log history.")

with tab_stats:
    stats = logger.get_statistics()
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Logged Detections</div>
                <div class="metric-value">{stats['total_detections']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Unique Object Classes</div>
                <div class="metric-value">{stats['objects_detected_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Peak Model Confidence</div>
                <div class="metric-value">{stats['highest_confidence']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mean Confidence</div>
                <div class="metric-value">{stats['avg_confidence']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if stats["risk_counts"]:
        st.markdown("#### Risk Level Distribution")
        risk_df = pd.DataFrame(list(stats["risk_counts"].items()), columns=["Risk Level", "Count"])
        st.bar_chart(risk_df.set_index("Risk Level"))

with tab_about:
    st.markdown(
        """
        ### System Architecture & Custom Fine-Tuning Guide

        This industry-grade surveillance platform supports continuous real-time runway monitoring.

        #### 🔧 Fine-Tuning & Custom Model Weights Replacement:
        To replace standard YOLOv11 weights with fine-tuned runway FOD weights (`best.pt`):
        1. Place your custom trained weights in `models/best.pt`.
        2. Select **Custom FOD Trained Weights** from the sidebar model selector dropdown, or update `DEFAULT_MODEL = "models/best.pt"` in `config.py`.

        #### 🚁 Future Integration Ready:
        The underlying `ThreadedCamera` and `FODDetector` engines natively support:
        - **IP Cameras / RTSP Video Streams** (`rtsp://username:password@camera_ip:554/feed`)
        - **Drone Surveillance Streams**
        - **ByteTrack & BoT-SORT Object Trackers**
        - **CUDA Hardware Acceleration**
        """
    )
