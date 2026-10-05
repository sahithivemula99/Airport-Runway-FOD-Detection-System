"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Risk Classification Module
"""

from typing import Dict, Any, List, Optional
import config


class RiskClassifier:
    """
    Classifies detected foreign objects into operational risk levels on runway surfaces.
    Risk Levels: LOW, MEDIUM, HIGH, CRITICAL.
    """

    # Numeric risk weights for threat scoring (1 to 10 scale)
    RISK_SCORES: Dict[str, int] = {
        config.RISK_LOW: 2,
        config.RISK_MEDIUM: 5,
        config.RISK_HIGH: 8,
        config.RISK_CRITICAL: 10,
    }

    # Operational danger summaries
    RISK_DESCRIPTIONS: Dict[str, str] = {
        config.RISK_LOW: "Minor debris. Minimal risk to aircraft tires/engines. Monitor routine sweep.",
        config.RISK_MEDIUM: "Moderate hazard. Potential wildlife or engine ingestion threat. Caution advised.",
        config.RISK_HIGH: "Severe risk. High probability of tire puncture or engine damage. Inspection required.",
        config.RISK_CRITICAL: "CATASTROPHIC RISK. Metal / Tool / Stone on active runway. IMMEDIATE HOLD REQUESTED.",
    }

    @classmethod
    def assess(cls, object_name: str) -> Dict[str, Any]:
        """
        Assess risk rating, category, threat score, and danger recommendation for a detected object.

        :param object_name: Raw label string from object detector.
        :return: Dictionary containing category, risk_level, threat_score, color_hex, color_bgr, description.
        """
        clean_name: str = object_name.strip().lower()

        # Direct map check from config
        mapped_info: Optional[Dict[str, str]] = config.FOD_CATEGORY_MAP.get(clean_name)

        if mapped_info:
            category: str = mapped_info["category"]
            risk_level: str = mapped_info["risk"]
        else:
            # Fuzzy fallback matching
            category, risk_level = cls._fuzzy_classify(clean_name)

        threat_score: int = cls.RISK_SCORES.get(risk_level, 5)
        description: str = cls.RISK_DESCRIPTIONS.get(risk_level, "Unknown hazard rating.")
        hex_color: str = config.RISK_COLORS_HEX.get(risk_level, "#F59E0B")
        bgr_color: tuple = config.RISK_COLORS_BGR.get(risk_level, (11, 158, 245))

        return {
            "object_name": object_name,
            "category": category,
            "risk_level": risk_level,
            "threat_score": threat_score,
            "color_hex": hex_color,
            "color_bgr": bgr_color,
            "description": description,
        }

    @classmethod
    def _fuzzy_classify(cls, clean_name: str) -> tuple[str, str]:
        """Fuzzy fallback classification based on string keywords."""
        if any(w in clean_name for w in ["metal", "tool", "stone", "rock", "iron", "steel", "bolt", "nut", "wrench"]):
            return "Metal/Tool Debris", config.RISK_CRITICAL
        elif any(w in clean_name for w in ["bottle", "can", "glass", "tire", "tyre", "rubber"]):
            return "Hard Container/Tire Debris", config.RISK_HIGH
        elif any(w in clean_name for w in ["bird", "animal", "dog", "cat", "goose"]):
            return "Wildlife Hazard", config.RISK_MEDIUM
        elif any(w in clean_name for w in ["plastic", "bag", "paper", "cardboard"]):
            return "FOD Material", config.RISK_HIGH
        else:
            return "Unknown Foreign Object", config.RISK_MEDIUM

    @classmethod
    def get_highest_risk(cls, risk_levels: List[str]) -> str:
        """Return the highest risk level from a list of detected risk levels."""
        if not risk_levels:
            return config.RISK_LOW

        hierarchy = [config.RISK_CRITICAL, config.RISK_HIGH, config.RISK_MEDIUM, config.RISK_LOW]
        for level in hierarchy:
            if level in risk_levels:
                return level
        return config.RISK_LOW
