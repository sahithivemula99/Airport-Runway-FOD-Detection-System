"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Static Image Processor Module
"""

import io
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image

import config
from detect import FODDetector
from utils import resize_maintain_aspect, draw_bounding_boxes


class ImageProcessor:
    """
    Handles static runway image processing, high-res scaling, object detection,
    and visual annotation.
    """

    @staticmethod
    def load_image(image_input: Union[str, Path, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """
        Load an image from filepath, byte buffer, or numpy array into OpenCV BGR format.

        :param image_input: Image file path, raw bytes, or array.
        :return: OpenCV BGR image array or None if invalid.
        """
        try:
            if isinstance(image_input, np.ndarray):
                return image_input

            if isinstance(image_input, (str, Path)):
                img_path = str(image_input)
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to read image from path: {img_path}")
                return img

            if isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                rgb_arr = np.array(pil_img)
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            return None
        except Exception as e:
            print(f"[ImageProcessor] Image Loading Error: {e}")
            return None

    @classmethod
    def process_image(
        cls,
        image_input: Union[str, Path, bytes, np.ndarray],
        detector: FODDetector,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        display_settings: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Process static image: load, resize, detect FOD, and annotate.

        :param image_input: Source image input.
        :param detector: FODDetector instance.
        :param conf_threshold: Detection confidence slider threshold.
        :param display_settings: Visualization options dict.
        :return: Dictionary containing annotated image, raw image, detections list, inference latency.
        """
        raw_img = cls.load_image(image_input)
        if raw_img is None or raw_img.size == 0:
            return {
                "success": False,
                "error": "Invalid or unsupported image file.",
                "raw_image": None,
                "annotated_image": None,
                "detections": [],
                "inference_ms": 0.0,
            }

        if display_settings is None:
            display_settings = config.DEFAULT_DISPLAY_SETTINGS

        # Maintain aspect ratio scaling for large runway aerial images
        resized_img = resize_maintain_aspect(raw_img, target_width=1600, target_height=1000)

        # Run Detection (tracking disabled for static images)
        detections, inference_ms = detector.predict(
            frame=resized_img,
            conf_threshold=conf_threshold,
            enable_tracking=False,
        )

        # Render OpenCV annotations
        annotated_img = draw_bounding_boxes(
            image=resized_img,
            detections=detections,
            display_settings=display_settings,
            fps=0.0,
            latency_ms=inference_ms,
        )

        return {
            "success": True,
            "error": "",
            "raw_image": resized_img,
            "annotated_image": annotated_img,
            "detections": detections,
            "inference_ms": inference_ms,
        }
