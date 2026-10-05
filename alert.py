"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Smart Alert System Module
"""

import sys
import math
import wave
import struct
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import config

# Windows system audio module
if sys.platform == "win32":
    import winsound


class SmartAlertSystem:
    """
    Manages runway safety alert statuses, emergency operational recommendations,
    visual warning banners, browser HTML5 audio, and Windows system speaker alarms.
    """

    ALARM_AUDIO_PATH: Path = config.ASSETS_DIR / "runway_alert.wav"

    def __init__(self) -> None:
        """Initialize Smart Alert System and ensure alarm audio asset exists."""
        self._ensure_alarm_sound_exists()

    def _ensure_alarm_sound_exists(self) -> None:
        """Generates a high-intensity dual-frequency emergency runway siren audio file if missing."""
        try:
            if not self.ALARM_AUDIO_PATH.exists() or self.ALARM_AUDIO_PATH.stat().st_size == 0:
                sample_rate = 22050
                duration_sec = 1.5  # 1.5 seconds siren
                num_samples = int(sample_rate * duration_sec)

                with wave.open(str(self.ALARM_AUDIO_PATH), 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit PCM
                    wav_file.setframerate(sample_rate)

                    frames = bytearray()
                    for i in range(num_samples):
                        # Alternating emergency siren tone (880Hz / 587Hz)
                        cycle = (i // (sample_rate // 6)) % 2
                        freq = 880.0 if cycle == 0 else 587.33
                        t = float(i) / sample_rate

                        # Loud sine wave sample
                        sample_val = int(30000.0 * math.sin(2.0 * math.pi * freq * t))
                        frames.extend(struct.pack('<h', sample_val))

                    wav_file.writeframes(frames)
                print(f"[SmartAlertSystem] Generated alarm siren audio at: {self.ALARM_AUDIO_PATH}")
        except Exception as e:
            print(f"[SmartAlertSystem] Sound creation error: {e}")

    def play_system_speaker_sound(self) -> None:
        """Play alarm siren directly through system speakers (bypasses browser autoplay restrictions)."""
        try:
            self._ensure_alarm_sound_exists()
            if sys.platform == "win32" and self.ALARM_AUDIO_PATH.exists():
                winsound.PlaySound(
                    str(self.ALARM_AUDIO_PATH), winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            elif sys.platform == "win32":
                # Fallback to system warning beep if file issue
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception as e:
            print(f"[SmartAlertSystem] System speaker sound error: {e}")

    def get_audio_bytes(self) -> Optional[bytes]:
        """Return raw audio file bytes."""
        try:
            self._ensure_alarm_sound_exists()
            if self.ALARM_AUDIO_PATH.exists():
                with open(self.ALARM_AUDIO_PATH, "rb") as f:
                    return f.read()
        except Exception as e:
            print(f"[SmartAlertSystem] Read Audio Bytes Error: {e}")
        return None

    def get_audio_base64(self) -> Optional[str]:
        """Return base64 encoded audio string for HTML5 playback."""
        data = self.get_audio_bytes()
        if data:
            return base64.b64encode(data).decode("utf-8")
        return None

    def evaluate_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate current frame detections and return active alert status and recommendations.

        :param detections: List of detection dictionaries containing risk_level, category, confidence.
        :return: Alert state dictionary.
        """
        if not detections:
            return {
                "status": "CLEAR",
                "is_alert_active": False,
                "banner_title": "RUNWAY SAFE — NO DEBRIS DETECTED",
                "banner_severity": "SUCCESS",
                "recommendation": "Maintain standard automated continuous camera surveillance.",
                "highest_risk": config.RISK_LOW,
                "critical_objects": [],
                "trigger_audio": False,
            }

        risk_levels = [d.get("risk_level", config.RISK_LOW) for d in detections]
        highest_risk = self._get_highest_risk(risk_levels)
        critical_objects = [
            d["category"] for d in detections if d.get("risk_level") in [config.RISK_HIGH, config.RISK_CRITICAL]
        ]

        if highest_risk in [config.RISK_HIGH, config.RISK_CRITICAL]:
            return {
                "status": "CRITICAL_ALERT",
                "is_alert_active": True,
                "banner_title": f"⚠️ WARNING: RUNWAY UNSAFE — {highest_risk} RISK FOD DETECTED",
                "banner_severity": "DANGER",
                "recommendation": (
                    f"IMMEDIATE ACTION: Suspend runway takeoffs/landings. "
                    f"Dispatch FOD Inspection Team for: {', '.join(set(critical_objects))}."
                ),
                "highest_risk": highest_risk,
                "critical_objects": list(set(critical_objects)),
                "trigger_audio": True,
            }

        elif highest_risk == config.RISK_MEDIUM:
            return {
                "status": "WARNING",
                "is_alert_active": True,
                "banner_title": "⚠️ CAUTION: POTENTIAL RUNWAY FOD DETECTED",
                "banner_severity": "WARNING",
                "recommendation": "Monitor object closely. Prepare inspection sweeper vehicle if stationary.",
                "highest_risk": highest_risk,
                "critical_objects": [],
                "trigger_audio": False,
            }

        else:
            return {
                "status": "MONITORING",
                "is_alert_active": False,
                "banner_title": "RUNWAY CLEAR — MINOR FOD OBSERVED",
                "banner_severity": "INFO",
                "recommendation": "Low-level item detected. Schedule next routine runway sweep.",
                "highest_risk": highest_risk,
                "critical_objects": [],
                "trigger_audio": False,
            }

    @staticmethod
    def _get_highest_risk(risk_levels: List[str]) -> str:
        """Helper to determine peak risk."""
        hierarchy = [config.RISK_CRITICAL, config.RISK_HIGH, config.RISK_MEDIUM, config.RISK_LOW]
        for level in hierarchy:
            if level in risk_levels:
                return level
        return config.RISK_LOW
