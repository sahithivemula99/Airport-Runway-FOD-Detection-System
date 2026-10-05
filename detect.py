"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
YOLOv11 Object Detection & Tracking Engine Module
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from risk import RiskClassifier


class FODDetector:
    """
    Object detection and tracking engine utilizing Ultralytics YOLOv11 and ByteTrack.
    Supports CUDA GPU auto-acceleration, single model initialization, and runtime model switching.
    """

    def __init__(self, model_name_or_path: str = config.DEFAULT_MODEL) -> None:
        """
        Initialize FOD Detector with specified model weights.

        :param model_name_or_path: Model filename ('yolo11n.pt', 'yolo11s.pt') or path.
        """
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.current_model_path: str = model_name_or_path
        self.model: Optional[YOLO] = None
        self._load_model(self.current_model_path)

    def _load_model(self, model_path: str) -> None:
        """
        Load YOLO model weights safely.

        :param model_path: Model name or absolute filepath.
        """
        print(f"[FODDetector] Loading model '{model_path}' on device '{self.device}'...")
        try:
            # Check if custom model path exists or standard pretrained weight
            if model_path.startswith("models/") or model_path.endswith(".pt"):
                target_path = Path(model_path)
                if target_path.exists():
                    self.model = YOLO(str(target_path))
                else:
                    # Fallback to standard pretrained yolo11 weights
                    print(f"[FODDetector] Custom model path '{model_path}' not found. Loading '{config.DEFAULT_MODEL}' fallback.")
                    self.model = YOLO(config.DEFAULT_MODEL)
                    self.current_model_path = config.DEFAULT_MODEL
            else:
                self.model = YOLO(model_path)

            # Move model to CUDA if available
            if self.device.startswith("cuda") and hasattr(self.model, "to"):
                self.model.to(self.device)

            print(f"[FODDetector] Model loaded successfully.")
        except Exception as e:
            print(f"[FODDetector] Critical Error loading model '{model_path}': {e}")
            # Emergency fallback initialization
            try:
                self.model = YOLO("yolo11n.pt")
                self.current_model_path = "yolo11n.pt"
            except Exception as ex:
                raise RuntimeError(f"Failed to initialize any YOLO model: {ex}")

    def switch_model(self, model_name_or_path: str) -> bool:
        """
        Switch active model weights at runtime.

        :param model_name_or_path: Model key or file path.
        :return: True if switch succeeded, False otherwise.
        """
        if model_name_or_path == self.current_model_path:
            return True

        try:
            self._load_model(model_name_or_path)
            self.current_model_path = model_name_or_path
            return True
        except Exception as e:
            print(f"[FODDetector] Model switch failed: {e}")
            return False

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
        enable_tracking: bool = True,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Perform detection and ByteTrack object tracking on an input frame.

        :param frame: OpenCV BGR image frame.
        :param conf_threshold: Detection confidence filter threshold.
        :param iou_threshold: Non-Maximum Suppression (NMS) IOU threshold.
        :param enable_tracking: If True, uses ByteTrack multi-object tracking.
        :return: Tuple of (List of detection result dictionaries, inference_time_ms).
        """
        if frame is None or frame.size == 0 or self.model is None:
            return [], 0.0

        try:
            # Execute Inference / Tracking
            if enable_tracking:
                results = self.model.track(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    persist=True,
                    tracker=config.TRACKER_TYPE,
                    verbose=False,
                )
            else:
                results = self.model(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False,
                )

            if not results or len(results) == 0:
                return [], 0.0

            result = results[0]
            inference_ms = (
                result.speed.get("inference", 0.0) if hasattr(result, "speed") else 0.0
            )

            detections: List[Dict[str, Any]] = []
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return [], inference_ms

            for i in range(len(boxes)):
                box = boxes[i]
                # Bounding box coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(int, xyxy)

                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_label = (
                    result.names[cls_id]
                    if hasattr(result, "names") and cls_id in result.names
                    else "object"
                )

                # Tracking ID if available
                track_id: Optional[int] = None
                if box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())

                # Risk Assessment via RiskClassifier
                risk_info = RiskClassifier.assess(raw_label)

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": cls_id,
                        "raw_label": raw_label,
                        "category": risk_info["category"],
                        "risk_level": risk_info["risk_level"],
                        "threat_score": risk_info["threat_score"],
                        "color_hex": risk_info["color_hex"],
                        "color_bgr": risk_info["color_bgr"],
                        "description": risk_info["description"],
                        "track_id": track_id,
                    }
                )

            return detections, inference_ms

        except Exception as e:
            print(f"[FODDetector] Inference Exception: {e}")
            return [], 0.0
