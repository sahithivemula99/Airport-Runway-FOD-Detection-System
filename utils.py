"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Utilities & System Monitor Module
"""

import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np
import psutil
import torch
import config


class SystemMonitor:
    """
    Monitors hardware telemetry including CPU, GPU, System Memory (RAM),
    CUDA status, inference latency, and processing FPS.
    """

    def __init__(self) -> None:
        """Initialize CUDA detection status."""
        self.cuda_available: bool = torch.cuda.is_available()
        self.device_name: str = (
            torch.cuda.get_device_name(0) if self.cuda_available else "CPU (Host System)"
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Fetch real-time CPU, RAM, and GPU hardware metrics.

        :return: Telemetry metrics dictionary.
        """
        cpu_usage: float = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        ram_percent: float = ram_info.percent
        ram_used_gb: float = round(ram_info.used / (1024 ** 3), 2)
        ram_total_gb: float = round(ram_info.total / (1024 ** 3), 2)

        gpu_vram_used_mb: float = 0.0
        gpu_vram_total_mb: float = 0.0
        gpu_utilization: float = 0.0

        if self.cuda_available:
            try:
                gpu_vram_used_mb = round(
                    torch.cuda.memory_allocated(0) / (1024 ** 2), 2
                )
                gpu_vram_total_mb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2
                )
            except Exception:
                pass

        return {
            "cpu_usage_percent": cpu_usage,
            "ram_usage_percent": ram_percent,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "gpu_vram_used_mb": gpu_vram_used_mb,
            "gpu_vram_total_mb": gpu_vram_total_mb,
        }


class FPSTimer:
    """Calculates smoothed frames-per-second (FPS) and frame latency."""

    def __init__(self, alpha: float = 0.1) -> None:
        """
        :param alpha: Smoothing factor for exponential moving average (0 < alpha <= 1).
        """
        self.alpha: float = alpha
        self.prev_time: float = time.time()
        self.fps: float = 0.0
        self.latency_ms: float = 0.0

    def update(self) -> Tuple[float, float]:
        """
        Update timer for current frame.

        :return: Tuple of (smoothed_fps, latency_in_ms).
        """
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time

        if delta > 0:
            instant_fps = 1.0 / delta
            self.latency_ms = delta * 1000.0
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = (self.alpha * instant_fps) + ((1.0 - self.alpha) * self.fps)

        return round(self.fps, 1), round(self.latency_ms, 1)


def resize_maintain_aspect(
    image: np.ndarray, target_width: int = 1280, target_height: int = 720
) -> np.ndarray:
    """
    Resize image to fit within target dimensions while strictly maintaining aspect ratio.

    :param image: Input OpenCV BGR image array.
    :param target_width: Max width bound.
    :param target_height: Max height bound.
    :return: Resized image array.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)

    if scale >= 1.0:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_annotated_screenshot(
    image: np.ndarray, prefix: str = "FOD_Detection"
) -> Tuple[bool, str]:
    """
    Save annotated image array to output directory with timestamp.

    :param image: OpenCV image array.
    :param prefix: Output filename prefix.
    :return: Tuple of (success_status, saved_filepath_str).
    """
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{prefix}_{timestamp_str}.jpg"
        filepath = config.OUTPUTS_DIR / filename
        cv2.imwrite(str(filepath), image)
        return True, str(filepath)
    except Exception as e:
        print(f"[Utils] Screenshot Save Error: {e}")
        return False, ""


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    display_settings: Dict[str, bool],
    fps: float = 0.0,
    latency_ms: float = 0.0,
) -> np.ndarray:
    """
    Draw professional high-visibility bounding boxes, risk badges, telemetry, and timestamp overlay.

    :param image: Source frame image array.
    :param detections: List of detection result dicts.
    :param display_settings: UI display toggles dict.
    :param fps: Frame processing rate.
    :param latency_ms: Inference latency in milliseconds.
    :return: Annotated OpenCV image frame.
    """
    if image is None or image.size == 0:
        return image

    annotated = image.copy()
    h, w = annotated.shape[:2]

    # 1. Draw Detections
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        bgr_color = det.get("color_bgr", (0, 0, 255))
        category = det.get("category", "FOD")
        risk_level = det.get("risk_level", config.RISK_LOW)
        confidence = det.get("confidence", 0.0)
        track_id = det.get("track_id")

        # Bounding Box
        if display_settings.get("show_boxes", True):
            thickness = 3 if risk_level in [config.RISK_HIGH, config.RISK_CRITICAL] else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, thickness)

            # Draw corner crosshairs for tactical industrial look
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), bgr_color, thickness + 1)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), bgr_color, thickness + 1)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), bgr_color, thickness + 1)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), bgr_color, thickness + 1)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), bgr_color, thickness + 1)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), bgr_color, thickness + 1)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), bgr_color, thickness + 1)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), bgr_color, thickness + 1)

        # Label Construction
        if display_settings.get("show_labels", True):
            label_parts = []
            if track_id is not None:
                label_parts.append(f"#{track_id}")
            label_parts.append(category.upper())
            label_parts.append(f"[{risk_level}]")

            if display_settings.get("show_confidence", True):
                label_parts.append(f"{confidence*100:.0f}%")

            label_text = " ".join(label_parts)

            # Label Background Box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            box_y1 = max(0, y1 - text_h - 10)
            cv2.rectangle(
                annotated,
                (x1, box_y1),
                (x1 + text_w + 12, box_y1 + text_h + 8),
                bgr_color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 6, box_y1 + text_h + 3),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    # 2. Top Telemetry Bar Overlay (FPS, Latency, Timestamp)
    bar_height = 36
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    status_items = []
    if display_settings.get("show_fps", True):
        status_items.append(f"FPS: {fps:.1f}")
        status_items.append(f"LATENCY: {latency_ms:.1f}ms")

    status_items.append(f"OBJECTS: {len(detections)}")

    if display_settings.get("show_timestamp", True):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_items.append(f"UTC/LOCAL: {now_str}")

    telemetry_str = "  |  ".join(status_items)
    cv2.putText(
        annotated,
        telemetry_str,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 204),
        1,
        cv2.LINE_AA,
    )

    return annotated
