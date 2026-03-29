"""Fusion helpers for punch_processor node.

Ring buffers, pending-event dataclasses, and the pad-constraint
reclassification logic live here to keep punch_processor.py lean.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from boxbunny_core.constants import DefenseType, PadLocation, PunchType


# ── Dataclasses for in-flight events ───────────────────────────────────────

@dataclass
class PendingCV:
    """A CV detection waiting for an IMU match."""
    timestamp: float
    punch_type: str
    confidence: float
    raw_class: str = ""


@dataclass
class PendingIMU:
    """A pad IMU impact waiting for a CV match."""
    timestamp: float
    pad: str
    level: str
    force_normalized: float


@dataclass
class DefenseWindow:
    """Accumulator open while a robot-arm command is being evaluated."""
    open_time: float
    arm: str
    punch_code: str
    arm_events: List[dict] = field(default_factory=list)
    cv_blocks: List[dict] = field(default_factory=list)
    tracking_snapshots: List[dict] = field(default_factory=list)


@dataclass
class SessionStats:
    """Running tallies for the session punch summary."""
    punch_counts: Dict[str, int] = field(default_factory=lambda: {p: 0 for p in PunchType.OFFENSIVE})
    force_sums: Dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PunchType.OFFENSIVE})
    pad_counts: Dict[str, int] = field(default_factory=lambda: {p: 0 for p in PadLocation.ALL})
    total_punches: int = 0
    confidence_sum: float = 0.0
    imu_confirmed_count: int = 0
    peak_force_level: str = ""
    peak_force_value: float = 0.0
    robot_punches_thrown: int = 0
    robot_punches_landed: int = 0
    defense_types: Dict[str, int] = field(default_factory=lambda: {
        DefenseType.BLOCK: 0, DefenseType.SLIP: 0,
        DefenseType.DODGE: 0, DefenseType.HIT: 0,
        DefenseType.UNKNOWN: 0,
    })
    depth_values: List[float] = field(default_factory=list)
    lateral_values: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    rounds_completed: int = 0

    # ── helpers ─────────────────────────────────────────────────────────

    def record_punch(
        self,
        punch_type: str,
        pad: str,
        force: float,
        level: str,
        confidence: float,
        imu_confirmed: bool,
    ) -> None:
        self.total_punches += 1
        self.punch_counts[punch_type] = self.punch_counts.get(punch_type, 0) + 1
        self.force_sums[punch_type] = self.force_sums.get(punch_type, 0.0) + force
        if pad:
            self.pad_counts[pad] = self.pad_counts.get(pad, 0) + 1
        self.confidence_sum += confidence
        if imu_confirmed:
            self.imu_confirmed_count += 1
        if force > self.peak_force_value:
            self.peak_force_value = force
            self.peak_force_level = level

    def record_defense(self, defense_type: str) -> None:
        self.robot_punches_thrown += 1
        if defense_type == DefenseType.HIT:
            self.robot_punches_landed += 1
        self.defense_types[defense_type] = self.defense_types.get(defense_type, 0) + 1

    def record_tracking(self, depth: float, lateral: float) -> None:
        if depth > 0.0:
            self.depth_values.append(depth)
        if lateral != 0.0:
            self.lateral_values.append(abs(lateral))

    def to_summary_fields(self) -> dict:
        """Return a dict matching SessionPunchSummary message fields."""
        avg_conf = (self.confidence_sum / self.total_punches) if self.total_punches else 0.0
        imu_rate = (self.imu_confirmed_count / self.total_punches) if self.total_punches else 0.0
        defense_rate = (
            1.0 - (self.robot_punches_landed / self.robot_punches_thrown)
            if self.robot_punches_thrown else 0.0
        )
        avg_force = {}
        for p in PunchType.OFFENSIVE:
            cnt = self.punch_counts.get(p, 0)
            avg_force[p] = (self.force_sums.get(p, 0.0) / cnt) if cnt else 0.0

        avg_depth = (sum(self.depth_values) / len(self.depth_values)) if self.depth_values else 0.0
        depth_range = (max(self.depth_values) - min(self.depth_values)) if self.depth_values else 0.0
        lat_move = (sum(self.lateral_values) / len(self.lateral_values)) if self.lateral_values else 0.0

        return {
            "total_punches": self.total_punches,
            "punch_distribution_json": json.dumps(self.punch_counts),
            "force_distribution_json": json.dumps(avg_force),
            "pad_distribution_json": json.dumps(self.pad_counts),
            "average_confidence": avg_conf,
            "peak_force_level": self.peak_force_level,
            "imu_confirmation_rate": imu_rate,
            "robot_punches_thrown": self.robot_punches_thrown,
            "robot_punches_landed": self.robot_punches_landed,
            "defense_rate": defense_rate,
            "defense_type_breakdown_json": json.dumps(self.defense_types),
            "avg_depth": avg_depth,
            "depth_range": depth_range,
            "lateral_movement": lat_move,
            "session_duration_sec": time.time() - self.start_time,
            "rounds_completed": self.rounds_completed,
        }


# ── Ring buffer ────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-capacity deque wrapper with timestamp-based expiry."""

    def __init__(self, maxlen: int = 50) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def append(self, item: object) -> None:
        self._buf.append(item)

    def expire(self, cutoff: float) -> List:
        """Remove and return items whose timestamp <= *cutoff*."""
        expired: List = []
        while self._buf and self._buf[0].timestamp <= cutoff:  # type: ignore[union-attr]
            expired.append(self._buf.popleft())
        return expired

    def pop_match(self, cutoff_lo: float, cutoff_hi: float) -> Optional[object]:
        """Return first item with timestamp in [cutoff_lo, cutoff_hi], or None."""
        for i, item in enumerate(self._buf):
            if cutoff_lo <= item.timestamp <= cutoff_hi:  # type: ignore[union-attr]
                del self._buf[i]
                return item
        return None

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# ── Pad-constraint reclassification ────────────────────────────────────────

def reclassify_punch(
    pad: str,
    cv_type: str,
    secondary_classes: Optional[Sequence[tuple[str, float]]] = None,
    min_conf: float = 0.25,
) -> str:
    """Return a valid punch type for *pad*, reclassifying if needed.

    *secondary_classes* is an optional list of ``(class_name, confidence)``
    sorted by descending confidence (excluding the primary).
    """
    valid = PadLocation.VALID_PUNCHES.get(pad)
    if valid is None:
        return cv_type  # unknown pad — pass through

    if cv_type in valid:
        return cv_type

    # Primary violates constraint — try secondary classes
    if secondary_classes:
        for cls_name, conf in secondary_classes:
            if cls_name in valid and conf >= min_conf:
                return cls_name

    return "unclassified"


def classify_defense(
    arm_events: List[dict],
    cv_blocks: List[dict],
    tracking_snapshots: List[dict],
    *,
    block_cv_min: float = 0.3,
    slip_lateral_px: float = 40.0,
    slip_depth_m: float = 0.15,
    dodge_lateral_px: float = 20.0,
    dodge_depth_m: float = 0.08,
) -> tuple[bool, str]:
    """Determine defense outcome.

    Returns ``(struck, defense_type)`` where *struck* is True when
    the robot arm contacted the user.
    """
    # Check if arm data exists
    if not arm_events:
        return False, DefenseType.UNKNOWN

    struck = any(e.get("contact", False) for e in arm_events)
    if struck:
        return True, DefenseType.HIT

    # Arm missed — check for block via CV
    has_block = any(
        b.get("confidence", 0.0) >= block_cv_min
        for b in cv_blocks
    )
    if has_block:
        return False, DefenseType.BLOCK

    # Check tracking displacement
    if tracking_snapshots:
        max_lateral = max(abs(t.get("lateral_displacement", 0.0)) for t in tracking_snapshots)
        max_depth = max(abs(t.get("depth_displacement", 0.0)) for t in tracking_snapshots)

        if max_lateral >= slip_lateral_px or max_depth >= slip_depth_m:
            return False, DefenseType.SLIP
        if max_lateral >= dodge_lateral_px or max_depth >= dodge_depth_m:
            return False, DefenseType.DODGE

    return False, DefenseType.UNKNOWN
