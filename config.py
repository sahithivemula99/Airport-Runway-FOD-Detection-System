"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Configuration Module
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Directory
BASE_DIR: Path = Path(__file__).resolve().parent

# Mandatory Directories
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
IMAGES_DIR: Path = BASE_DIR / "images"
VIDEOS_DIR: Path = BASE_DIR / "videos"
DATASET_DIR: Path = BASE_DIR / "dataset"
ASSETS_DIR: Path = BASE_DIR / "assets"

# Detection History Log File
LOG_FILE_PATH: Path = OUTPUTS_DIR / "detection_history.csv"


def ensure_directories_exist() -> None:
    """Create all required project directories if they do not exist."""
    directories = [
        MODELS_DIR,
        OUTPUTS_DIR,
        IMAGES_DIR,
        VIDEOS_DIR,
        DATASET_DIR,
        ASSETS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories are created on module import
ensure_directories_exist()

# Model Definitions
AVAILABLE_MODELS: Dict[str, str] = {
    "YOLOv11 Nano (Fastest)": "yolo11n.pt",
    "YOLOv11 Small (Balanced)": "yolo11s.pt",
    "YOLOv11 Medium (High Accuracy)": "yolo11m.pt",
    "Custom FOD Trained Weights": "models/best.pt",
}
DEFAULT_MODEL: str = "yolo11n.pt"

# Object Detection & Tracker Config
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.20  # Sensitive FOD detection
DEFAULT_IOU_THRESHOLD: float = 0.45
TRACKER_TYPE: str = "bytetrack.yaml"

# Default Target FPS Benchmarks
TARGET_CPU_FPS: float = 20.0
TARGET_GPU_FPS: float = 40.0

# Risk Levels Definition
RISK_LOW: str = "LOW"
RISK_MEDIUM: str = "MEDIUM"
RISK_HIGH: str = "HIGH"
RISK_CRITICAL: str = "CRITICAL"

RISK_COLORS_HEX: Dict[str, str] = {
    RISK_LOW: "#10B981",       # Emerald Green
    RISK_MEDIUM: "#F59E0B",    # Amber Yellow
    RISK_HIGH: "#EF4444",      # Bright Red
    RISK_CRITICAL: "#990000",  # Dark Crimson Red
}

RISK_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    RISK_LOW: (129, 185, 16),       # Emerald
    RISK_MEDIUM: (11, 158, 245),     # Amber
    RISK_HIGH: (68, 68, 239),       # Bright Red
    RISK_CRITICAL: (0, 0, 153),     # Crimson Red
}

# FOD Classes & COCO Map Fallback for Pretrained YOLO Models
FOD_CATEGORY_MAP: Dict[str, Dict[str, str]] = {
    # Custom FOD Specific Categories
    "stone": {"category": "Stone", "risk": RISK_CRITICAL},
    "metal": {"category": "Metal", "risk": RISK_CRITICAL},
    "tool": {"category": "Tool", "risk": RISK_CRITICAL},
    "bottle": {"category": "Bottle", "risk": RISK_HIGH},
    "plastic": {"category": "Plastic", "risk": RISK_HIGH},
    "bird": {"category": "Bird", "risk": RISK_MEDIUM},
    "tyre debris": {"category": "Tyre Debris", "risk": RISK_HIGH},
    "unknown object": {"category": "Unknown Object", "risk": RISK_HIGH},
    
    # Pretrained COCO Mappings (High-Risk Fallbacks for Pretrained YOLO)
    "rock": {"category": "Stone", "risk": RISK_CRITICAL},
    "wrench": {"category": "Metal Tool", "risk": RISK_CRITICAL},
    "scissors": {"category": "Metal Tool", "risk": RISK_CRITICAL},
    "knife": {"category": "Metal Tool", "risk": RISK_CRITICAL},
    "skateboard": {"category": "Metal / Heavy FOD Debris", "risk": RISK_CRITICAL},
    "airplane": {"category": "Aircraft Component / Debris", "risk": RISK_CRITICAL},
    "cup": {"category": "Plastic", "risk": RISK_HIGH},
    "wine glass": {"category": "Bottle / Glass", "risk": RISK_HIGH},
    "suitcase": {"category": "Luggage Debris", "risk": RISK_CRITICAL},
    "backpack": {"category": "Luggage Debris", "risk": RISK_CRITICAL},
    "handbag": {"category": "Luggage Debris", "risk": RISK_HIGH},
    "umbrella": {"category": "Foreign Object Debris", "risk": RISK_HIGH},
    "cell phone": {"category": "Metal Electronics", "risk": RISK_CRITICAL},
    "laptop": {"category": "Metal Electronics", "risk": RISK_CRITICAL},
    "mouse": {"category": "Plastic", "risk": RISK_MEDIUM},
    "remote": {"category": "Plastic", "risk": RISK_HIGH},
    "keyboard": {"category": "Plastic / Electronics", "risk": RISK_HIGH},
    "book": {"category": "Paper Debris", "risk": RISK_LOW},
    "sports ball": {"category": "Tyre Debris", "risk": RISK_HIGH},
    "frisbee": {"category": "Plastic", "risk": RISK_HIGH},
}

# UI Display Default Settings
DEFAULT_DISPLAY_SETTINGS: Dict[str, bool] = {
    "show_boxes": True,
    "show_labels": True,
    "show_confidence": True,
    "show_fps": True,
    "show_timestamp": True,
    "dark_mode": True,
    "enable_sound_alert": True,
    "frame_skipping": False,
}

# System Information
APP_TITLE: str = "AIRPORT SURVEILLANCE MATRIX — FOD DETECTION & RISK ASSESSMENT"
APP_SUBTITLE: str = "Real-Time Runway Safety Computer Vision Platform"
ORGANIZATION: str = "Airport Safety & Operations Directorate"
