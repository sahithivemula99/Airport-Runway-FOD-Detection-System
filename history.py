"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Detection History Logger & Analytics Module
"""

import os
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import config


class HistoryLogger:
    """
    Manages persistent logging of runway FOD detections to CSV file
    and provides real-time analytics for the surveillance dashboard.
    """

    HEADERS: List[str] = [
        "Timestamp",
        "Source",
        "Track_ID",
        "Object_Name",
        "Category",
        "Confidence",
        "Risk_Level",
        "BBox",
    ]

    def __init__(self, log_path: Path = config.LOG_FILE_PATH) -> None:
        """Initialize logger and create log CSV file with headers if missing."""
        self.log_path: Path = log_path
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self) -> None:
        """Ensure destination directory and CSV file exist with appropriate headers."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log_detection(
        self,
        source: str,
        object_name: str,
        category: str,
        confidence: float,
        risk_level: str,
        track_id: Optional[int] = None,
        bbox: Optional[List[int]] = None,
    ) -> None:
        """
        Log a single detection record into CSV.

        :param source: Feeder name ('Live Webcam', 'Runway Video', 'Image Upload').
        :param object_name: Raw object name.
        :param category: Classified FOD category.
        :param confidence: Detection confidence score (0.0 to 1.0).
        :param risk_level: Assessed risk level.
        :param track_id: Unique object tracking ID.
        :param bbox: Bounding box [x1, y1, x2, y2].
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            bbox_str = str(bbox) if bbox else "N/A"
            track_str = str(track_id) if track_id is not None else "N/A"

            row = [
                timestamp,
                source,
                track_str,
                object_name,
                category,
                f"{confidence:.4f}",
                risk_level,
                bbox_str,
            ]

            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[HistoryLogger] Logging Error: {e}")

    def log_batch_detections(self, source: str, detections: List[Dict[str, Any]]) -> None:
        """Log multiple detections from a single frame."""
        for det in detections:
            self.log_detection(
                source=source,
                object_name=det.get("object_name", "Unknown"),
                category=det.get("category", "Unknown"),
                confidence=det.get("confidence", 0.0),
                risk_level=det.get("risk_level", config.RISK_LOW),
                track_id=det.get("track_id"),
                bbox=det.get("bbox"),
            )

    def get_history_df(self, limit: int = 500) -> pd.DataFrame:
        """
        Retrieve history logs as pandas DataFrame.

        :param limit: Maximum rows to retrieve (most recent first).
        :return: Pandas DataFrame.
        """
        try:
            if not self.log_path.exists() or self.log_path.stat().st_size == 0:
                return pd.DataFrame(columns=self.HEADERS)

            df = pd.read_csv(self.log_path)
            if df.empty:
                return pd.DataFrame(columns=self.HEADERS)

            # Return most recent rows first
            return df.tail(limit).iloc[::-1].reset_index(drop=True)
        except Exception as e:
            print(f"[HistoryLogger] Read Error: {e}")
            return pd.DataFrame(columns=self.HEADERS)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate analytics from log history.

        :return: Summary metrics dictionary.
        """
        df = self.get_history_df(limit=2000)
        if df.empty:
            return {
                "total_detections": 0,
                "objects_detected_count": 0,
                "highest_confidence": 0.0,
                "avg_confidence": 0.0,
                "risk_counts": {config.RISK_LOW: 0, config.RISK_MEDIUM: 0, config.RISK_HIGH: 0, config.RISK_CRITICAL: 0},
                "category_breakdown": {},
            }

        total = len(df)
        unique_objects = df["Object_Name"].nunique() if "Object_Name" in df else 0

        # Confidence stats
        conf_series = pd.to_numeric(df["Confidence"], errors="coerce").fillna(0.0)
        highest_conf = float(conf_series.max()) if not conf_series.empty else 0.0
        avg_conf = float(conf_series.mean()) if not conf_series.empty else 0.0

        # Risk Breakdown
        risk_counts = df["Risk_Level"].value_counts().to_dict() if "Risk_Level" in df else {}

        # Category Breakdown
        cat_counts = df["Category"].value_counts().to_dict() if "Category" in df else {}

        return {
            "total_detections": total,
            "objects_detected_count": unique_objects,
            "highest_confidence": highest_conf,
            "avg_confidence": avg_conf,
            "risk_counts": risk_counts,
            "category_breakdown": cat_counts,
        }

    def clear_history(self) -> None:
        """Clear all historical logs."""
        try:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
        except Exception as e:
            print(f"[HistoryLogger] Clear Error: {e}")
