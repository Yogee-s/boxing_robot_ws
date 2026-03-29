"""Population benchmark engine for BoxBunny.

Computes percentile rankings by comparing user performance metrics
against population norms segmented by age, gender, and skill level.
Data sourced from sports science literature.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("boxbunny.benchmarks")


def _age_bracket(age: int) -> str:
    """Map age to bracket string."""
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


class BenchmarkEngine:
    """Computes percentile rankings against population norms.

    Usage::

        engine = BenchmarkEngine()
        result = engine.percentile("reaction_time_ms", 245, age=28, gender="male")
        # result = {"percentile": 48, "tier": "Average", "bracket": "25-34"}
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        if data_path is None:
            ws_root = Path(__file__).resolve().parents[3]
            data_path = str(ws_root / "data" / "benchmarks" / "population_norms.json")
        try:
            with open(data_path, "r") as f:
                self._data = json.load(f)
            logger.info("Loaded population benchmarks from %s", data_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load benchmarks: %s", e)
            self._data = {}
        self._tiers = self._data.get("tier_labels", {})

    def percentile(
        self,
        metric: str,
        value: float,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute percentile ranking for a metric value.

        Args:
            metric: Metric name (e.g., "reaction_time_ms", "punches_per_minute")
            value: The user's value
            age: User's age (used for age-bracketed metrics)
            gender: "male" or "female"
            level: "beginner"/"intermediate"/"advanced" (for level-bracketed metrics)

        Returns:
            Dict with keys: percentile (0-100), tier (str), bracket (str),
            comparison (str, human-readable), and norms (dict of p10-p90).
        """
        metric_data = self._data.get(metric)
        if metric_data is None:
            return {"percentile": 50, "tier": "Unknown", "comparison": "No benchmark data available"}

        # Resolve gender
        gender_key = (gender or "male").lower()
        if gender_key not in ("male", "female"):
            gender_key = "male"
        gender_data = metric_data.get(gender_key)
        if gender_data is None:
            return {"percentile": 50, "tier": "Unknown", "comparison": "No data for gender"}

        # Resolve bracket (age-based or level-based)
        bracket_key = None
        if level and level in gender_data:
            bracket_key = level
        elif age:
            bracket_key = _age_bracket(age)
        else:
            # Default to first available bracket
            bracket_key = next(
                (k for k in gender_data if not k.startswith("_")), None
            )
        if bracket_key is None or bracket_key not in gender_data:
            return {"percentile": 50, "tier": "Unknown", "comparison": "No data for bracket"}

        norms = gender_data[bracket_key]
        percentile = self._interpolate_percentile(value, norms, lower_is_better=(metric == "reaction_time_ms"))
        tier = self._tier_from_percentile(percentile)

        return {
            "percentile": percentile,
            "tier": tier,
            "bracket": bracket_key,
            "gender": gender_key,
            "norms": norms,
            "comparison": self._comparison_text(metric, percentile, tier, gender_key, bracket_key),
        }

    def get_all_percentiles(
        self, user_stats: Dict, age: Optional[int] = None,
        gender: Optional[str] = None, level: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Compute percentiles for all available metrics at once.

        Args:
            user_stats: Dict mapping metric names to user values
            age, gender, level: User demographics

        Returns:
            Dict mapping metric name to percentile result dict.
        """
        results = {}
        metric_map = {
            "avg_reaction_ms": "reaction_time_ms",
            "punches_per_minute": "punches_per_minute",
            "avg_force": "punch_force_normalized",
            "fatigue_index": "fatigue_index",
            "defense_rate": "defense_rate",
            "total_punches": "session_punch_count",
        }
        for stat_key, metric_key in metric_map.items():
            if stat_key in user_stats:
                results[stat_key] = self.percentile(
                    metric_key, user_stats[stat_key],
                    age=age, gender=gender, level=level,
                )
        return results

    def _interpolate_percentile(
        self, value: float, norms: Dict[str, float], lower_is_better: bool = False,
    ) -> int:
        """Interpolate percentile from norm distribution."""
        points = [
            (norms.get("p10", 0), 10),
            (norms.get("p25", 0), 25),
            (norms.get("p50", 0), 50),
            (norms.get("p75", 0), 75),
            (norms.get("p90", 0), 90),
        ]
        if lower_is_better:
            # Flip: lower value = higher percentile
            # e.g., 190ms reaction time for a bracket where p10=190 means top 10%
            points = [(v, 100 - p) for v, p in points]
            points.sort(key=lambda x: x[0])

        # Below lowest
        if value <= points[0][0]:
            return points[0][1] if not lower_is_better else 95
        # Above highest
        if value >= points[-1][0]:
            return points[-1][1] if not lower_is_better else 5

        # Linear interpolation between adjacent points
        for i in range(len(points) - 1):
            v0, p0 = points[i]
            v1, p1 = points[i + 1]
            if v0 <= value <= v1:
                if v1 == v0:
                    return int(p0)
                ratio = (value - v0) / (v1 - v0)
                return int(p0 + ratio * (p1 - p0))

        return 50  # Fallback

    def _tier_from_percentile(self, percentile: int) -> str:
        """Map percentile to tier label."""
        if percentile >= 90:
            return self._tiers.get("p90_plus", "Elite")
        if percentile >= 75:
            return self._tiers.get("p75_to_p90", "Advanced")
        if percentile >= 50:
            return self._tiers.get("p50_to_p75", "Above Average")
        if percentile >= 25:
            return self._tiers.get("p25_to_p50", "Average")
        if percentile >= 10:
            return self._tiers.get("p10_to_p25", "Developing")
        return self._tiers.get("below_p10", "Getting Started")

    def _comparison_text(
        self, metric: str, percentile: int, tier: str,
        gender: str, bracket: str,
    ) -> str:
        """Generate human-readable comparison text."""
        group = f"{gender}s aged {bracket}" if bracket[0].isdigit() else f"{bracket} {gender}s"
        metric_names = {
            "reaction_time_ms": "reaction time",
            "punches_per_minute": "punch rate",
            "punch_force_normalized": "punch power",
            "fatigue_index": "endurance",
            "defense_rate": "defense",
            "session_punch_count": "punch volume",
        }
        name = metric_names.get(metric, metric)
        return f"Your {name} is better than {percentile}% of {group} ({tier})"
