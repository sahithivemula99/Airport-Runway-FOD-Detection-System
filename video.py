"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Video File Processor Module
"""

import time
from pathlib import Path
from typing import Generator, Dict, Any, Tuple, Optional
import cv2
import numpy as np

import config
from detect import FODDetector
from utils import FPSTimer, resize_maintain_aspect, draw_bounding_boxes


class VideoProcessor:
    """
    Handles video file inference processing with frame-skipping optimizations,
    telemetry generation, and high-performance video streaming.
    """

    def __init__(self, video_path: str) -> None:
        """
        Initialize Video Processor.

        :param video_path: Path to input video file.
        """
        self.video_path: str = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps_original: float = 30.0
        self.width: int = 1280
        self.height: int = 720
        self._init_video()

    def _init_video(self) -> None:
        """Open video file and read video metadata properties."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Unable to open video file: {self.video_path}")

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps_original = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        except Exception as e:
            print(f"[VideoProcessor] Video opening error: {e}")

    def process_frames(
        self,
        detector: FODDetector,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        display_settings: Optional[Dict[str, bool]] = None,
        frame_skip: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Frame generator yielding real-time detection results, annotated images, and telemetry.

        :param detector: FODDetector instance.
        :param conf_threshold: Confidence slider value.
        :param display_settings: UI options dict.
        :param frame_skip: Number of frames to skip per detection pass for CPU speedup.
        :yield: Frame detection dictionary payload.
        """
        if not self.cap or not self.cap.isOpened():
            return

        if display_settings is None:
            display_settings = config.DEFAULT_DISPLAY_SETTINGS

        timer = FPSTimer()
        frame_count = 0
        last_detections = []
        last_inference_ms = 0.0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            frame_count += 1

            # Resize for optimal display & processing
            frame_resized = resize_maintain_aspect(frame, target_width=1280, target_height=720)

            # Frame skipping optimization
            if frame_skip > 0 and (frame_count % (frame_skip + 1) != 0) and last_detections:
                detections = last_detections
                inference_ms = last_inference_ms
            else:
                detections, inference_ms = detector.predict(
                    frame=frame_resized,
                    conf_threshold=conf_threshold,
                    enable_tracking=True,
                )
                last_detections = detections
                last_inference_ms = inference_ms

            curr_fps, latency_ms = timer.update()

            # Render OpenCV annotations
            annotated_frame = draw_bounding_boxes(
                image=frame_resized,
                detections=detections,
                display_settings=display_settings,
                fps=curr_fps,
                latency_ms=inference_ms,
            )

            progress_pct = (
                (frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
            )

            yield {
                "frame_index": frame_count,
                "total_frames": self.total_frames,
                "progress_pct": progress_pct,
                "raw_frame": frame_resized,
                "annotated_frame": annotated_frame,
                "detections": detections,
                "fps": curr_fps,
                "inference_ms": inference_ms,
                "latency_ms": latency_ms,
            }

        self.release()

    def release(self) -> None:
        """Release video resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
