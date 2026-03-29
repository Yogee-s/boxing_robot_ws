"""Punch processor -- CV + IMU fusion node.

Fuses computer-vision punch detections with pad IMU impacts to produce
confirmed punches.  Manages a defense-event pipeline and keeps a running
session summary.  Heavy lifting lives in ``punch_fusion.py``.
"""
from __future__ import annotations

import time
from typing import Optional

import rclpy
from rclpy.node import Node

from boxbunny_core.constants import (
    PunchType, SessionState as SSConst, Topics,
)
from boxbunny_core.config_loader import load_config
from boxbunny_core.punch_fusion import (
    DefenseWindow, PendingCV, PendingIMU, RingBuffer,
    SessionStats, classify_defense, reclassify_punch,
)

try:
    from boxbunny_msgs.msg import (
        ArmStrikeEvent, ConfirmedPunch, DefenseEvent, PunchDetection,
        PunchEvent, RobotCommand, SessionPunchSummary, SessionState,
        UserTracking,
    )
except ImportError:
    ArmStrikeEvent = ConfirmedPunch = DefenseEvent = None  # type: ignore[assignment,misc]
    PunchDetection = PunchEvent = RobotCommand = None  # type: ignore[assignment,misc]
    SessionPunchSummary = SessionState = UserTracking = None  # type: ignore[assignment,misc]


class PunchProcessorNode(Node):
    """Fuses CV detections and IMU impacts into confirmed punches."""

    def __init__(self) -> None:
        super().__init__("punch_processor")

        # Configuration
        fc = load_config().fusion
        self._fw_s: float = fc.fusion_window_ms / 1000.0
        self._cv_penalty: float = fc.cv_unconfirmed_confidence_penalty
        self._reclass_min: float = fc.reclassify_min_secondary_confidence
        self._dw_s: float = fc.defense_window_ms / 1000.0
        self._slip_lat: float = fc.slip_lateral_threshold_px
        self._slip_dep: float = fc.slip_depth_threshold_m
        self._dodge_lat: float = fc.dodge_lateral_threshold_px
        self._dodge_dep: float = fc.dodge_depth_threshold_m
        self._block_min: float = fc.block_cv_confidence_min

        # State
        self._pcv = RingBuffer(maxlen=64)
        self._pimu = RingBuffer(maxlen=64)
        self._def_win: Optional[DefenseWindow] = None
        self._session_active: bool = False
        self._stats: SessionStats = SessionStats()

        # Publishers
        self._pub_punch = (
            self.create_publisher(ConfirmedPunch, Topics.PUNCH_CONFIRMED, 10)
            if ConfirmedPunch else None)
        self._pub_defense = (
            self.create_publisher(DefenseEvent, Topics.PUNCH_DEFENSE, 10)
            if DefenseEvent else None)
        self._pub_summary = (
            self.create_publisher(SessionPunchSummary, Topics.PUNCH_SESSION_SUMMARY, 10)
            if SessionPunchSummary else None)

        # Subscribers
        _sub = self.create_subscription
        if PunchDetection:
            _sub(PunchDetection, Topics.CV_DETECTION, self._on_cv, 10)
        if PunchEvent:
            _sub(PunchEvent, Topics.IMU_PUNCH_EVENT, self._on_imu, 10)
        if ArmStrikeEvent:
            _sub(ArmStrikeEvent, Topics.IMU_ARM_EVENT, self._on_arm, 10)
        if RobotCommand:
            _sub(RobotCommand, Topics.ROBOT_COMMAND, self._on_robot_cmd, 10)
        if UserTracking:
            _sub(UserTracking, Topics.CV_USER_TRACKING, self._on_tracking, 10)
        if SessionState:
            _sub(SessionState, Topics.SESSION_STATE, self._on_session, 10)

        self.create_timer(0.05, self._tick_expiry)
        self.get_logger().info("PunchProcessorNode initialised")

    # -- CV detection ---------------------------------------------------

    def _on_cv(self, msg: PunchDetection) -> None:  # type: ignore[name-defined]
        ts = msg.timestamp if msg.timestamp > 0.0 else time.time()

        # Feed blocks into open defense window
        if self._def_win is not None and msg.punch_type == PunchType.BLOCK:
            self._def_win.cv_blocks.append({"confidence": msg.confidence, "timestamp": ts})
            return
        if msg.punch_type not in PunchType.OFFENSIVE:
            return

        pend = PendingCV(ts, msg.punch_type, msg.confidence, msg.raw_class)
        imu = self._pimu.pop_match(ts - self._fw_s, ts + self._fw_s)
        if imu is not None:
            self._fuse(pend, imu)
        else:
            self._pcv.append(pend)

    # -- IMU punch ------------------------------------------------------

    def _on_imu(self, msg: PunchEvent) -> None:  # type: ignore[name-defined]
        ts = msg.timestamp if msg.timestamp > 0.0 else time.time()
        pend = PendingIMU(ts, msg.pad, msg.level, msg.force_normalized)
        cv = self._pcv.pop_match(ts - self._fw_s, ts + self._fw_s)
        if cv is not None:
            self._fuse(cv, pend)
        else:
            self._pimu.append(pend)

    # -- Fused match ----------------------------------------------------

    def _fuse(self, cv: PendingCV, imu: PendingIMU) -> None:
        ptype = reclassify_punch(imu.pad, cv.punch_type, min_conf=self._reclass_min)
        self._emit(
            timestamp=cv.timestamp, punch_type=ptype, pad=imu.pad,
            level=imu.level, force=imu.force_normalized,
            confidence=cv.confidence, imu_confirmed=True, cv_confirmed=True,
        )

    # -- Expiry timer ---------------------------------------------------

    def _tick_expiry(self) -> None:
        now = time.time()
        cutoff = now - self._fw_s

        for cv in self._pcv.expire(cutoff):
            self._emit(
                timestamp=cv.timestamp, punch_type=cv.punch_type, pad="",
                level="", force=0.0,
                confidence=max(0.0, cv.confidence - self._cv_penalty),
                imu_confirmed=False, cv_confirmed=True,
            )
        for imu in self._pimu.expire(cutoff):
            self._emit(
                timestamp=imu.timestamp, punch_type="unclassified", pad=imu.pad,
                level=imu.level, force=imu.force_normalized,
                confidence=0.0, imu_confirmed=True, cv_confirmed=False,
            )
        if self._def_win and now - self._def_win.open_time >= self._dw_s:
            self._close_defense()

    # -- Robot command -> defense window --------------------------------

    def _on_robot_cmd(self, msg: RobotCommand) -> None:  # type: ignore[name-defined]
        if msg.command_type != "punch":
            return
        self._def_win = DefenseWindow(time.time(), "", msg.punch_code)
        self.get_logger().debug(f"Defense window opened: code={msg.punch_code}")

    # -- Arm strike callback --------------------------------------------

    def _on_arm(self, msg: ArmStrikeEvent) -> None:  # type: ignore[name-defined]
        if self._def_win is None:
            return
        self._def_win.arm = msg.arm
        self._def_win.arm_events.append({
            "arm": msg.arm, "contact": msg.contact, "timestamp": msg.timestamp,
        })

    # -- User tracking callback -----------------------------------------

    def _on_tracking(self, msg: UserTracking) -> None:  # type: ignore[name-defined]
        snap = {
            "lateral_displacement": msg.lateral_displacement,
            "depth_displacement": msg.depth_displacement,
            "depth": msg.depth, "timestamp": msg.timestamp,
        }
        if self._def_win is not None:
            self._def_win.tracking_snapshots.append(snap)
        if self._session_active:
            self._stats.record_tracking(msg.depth, msg.lateral_displacement)

    # -- Session state --------------------------------------------------

    def _on_session(self, msg: SessionState) -> None:  # type: ignore[name-defined]
        if msg.state == SSConst.ACTIVE and not self._session_active:
            self._session_active = True
            self._stats = SessionStats()
            self.get_logger().info("Session started -- tracking stats")
        elif msg.state == SSConst.REST and self._session_active:
            self._stats.rounds_completed += 1
        elif msg.state == SSConst.COMPLETE and self._session_active:
            self._stats.rounds_completed += 1
            self._publish_summary()
            self._session_active = False
            self.get_logger().info("Session complete -- summary published")

    # -- Close defense window -------------------------------------------

    def _close_defense(self) -> None:
        dw = self._def_win
        if dw is None:
            return
        self._def_win = None

        struck, dtype = classify_defense(
            dw.arm_events, dw.cv_blocks, dw.tracking_snapshots,
            block_cv_min=self._block_min, slip_lateral_px=self._slip_lat,
            slip_depth_m=self._slip_dep, dodge_lateral_px=self._dodge_lat,
            dodge_depth_m=self._dodge_dep,
        )
        if self._pub_defense is not None:
            m = DefenseEvent()
            m.timestamp = dw.open_time
            m.arm = dw.arm
            m.robot_punch_code = dw.punch_code
            m.struck = struck
            m.defense_type = dtype
            self._pub_defense.publish(m)
        if self._session_active:
            self._stats.record_defense(dtype)
        self.get_logger().debug(f"Defense: struck={struck} type={dtype}")

    # -- Emit confirmed punch -------------------------------------------

    def _emit(
        self, *, timestamp: float, punch_type: str, pad: str, level: str,
        force: float, confidence: float, imu_confirmed: bool, cv_confirmed: bool,
    ) -> None:
        if self._pub_punch is not None:
            m = ConfirmedPunch()
            m.timestamp = timestamp
            m.punch_type = punch_type
            m.pad = pad
            m.level = level
            m.force_normalized = force
            m.cv_confidence = confidence
            m.imu_confirmed = imu_confirmed
            m.cv_confirmed = cv_confirmed
            self._pub_punch.publish(m)
        if self._session_active and punch_type != "unclassified":
            self._stats.record_punch(punch_type, pad, force, level, confidence, imu_confirmed)
        self.get_logger().debug(
            f"Punch: {punch_type} pad={pad} conf={confidence:.2f} "
            f"imu={imu_confirmed} cv={cv_confirmed}"
        )

    # -- Session summary ------------------------------------------------

    def _publish_summary(self) -> None:
        if self._pub_summary is None:
            return
        fields = self._stats.to_summary_fields()
        m = SessionPunchSummary()
        for k, v in fields.items():
            setattr(m, k, v)
        self._pub_summary.publish(m)
        self.get_logger().info(
            f"Summary: {fields['total_punches']} punches, "
            f"defense_rate={fields['defense_rate']:.1%}"
        )


def main(args: list[str] | None = None) -> None:
    """ROS 2 entry point."""
    rclpy.init(args=args)
    node = PunchProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
