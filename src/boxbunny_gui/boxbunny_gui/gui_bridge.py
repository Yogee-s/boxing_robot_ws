"""Thread-safe ROS 2 to Qt signal bridge.

Creates a ROS 2 node in a background QThread and emits Qt signals
(auto-queued cross-thread) whenever relevant messages arrive.
Gracefully degrades to offline mode when rclpy is unavailable.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)

# ── Try importing ROS 2 ────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from boxbunny_msgs.msg import (
        ConfirmedPunch,
        CoachTip,
        DefenseEvent,
        DrillProgress,
        IMUStatus,
        NavCommand,
        RobotCommand,
        SessionState,
    )
    from boxbunny_msgs.srv import EndSession, GenerateLlm, StartSession
    from boxbunny_core.constants import Topics, Services

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("rclpy not available -- running in offline mode")


# ── ROS worker (runs in QThread) ───────────────────────────────────────────

class _RosWorker(QObject):
    """Background ROS 2 spin loop.  Lives inside a QThread."""

    punch_confirmed = Signal(dict)
    defense_event = Signal(dict)
    drill_progress = Signal(dict)
    session_state_changed = Signal(str, str)
    coach_tip = Signal(str, str)
    nav_command = Signal(str)
    imu_status = Signal(dict)
    cv_status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._node: Optional[Any] = None
        self._executor: Optional[Any] = None
        self._running: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start_spinning(self) -> None:
        """Initialise the node and enter the spin loop."""
        if not ROS_AVAILABLE:
            logger.info("Offline mode -- ROS worker will not spin")
            return

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("boxbunny_gui")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._subscribe_all()
        self._running = True
        logger.info("ROS worker spinning")
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._running = False
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        logger.info("ROS worker stopped")

    @property
    def node(self) -> Optional[Any]:
        return self._node

    # ── Subscriptions ───────────────────────────────────────────────────

    def _subscribe_all(self) -> None:
        n = self._node
        n.create_subscription(ConfirmedPunch, Topics.PUNCH_CONFIRMED, self._on_punch, 10)
        n.create_subscription(DefenseEvent, Topics.PUNCH_DEFENSE, self._on_defense, 10)
        n.create_subscription(DrillProgress, Topics.DRILL_PROGRESS, self._on_drill_progress, 10)
        n.create_subscription(SessionState, Topics.SESSION_STATE, self._on_session_state, 10)
        n.create_subscription(CoachTip, Topics.COACH_TIP, self._on_coach_tip, 10)
        n.create_subscription(NavCommand, Topics.IMU_NAV_EVENT, self._on_nav, 10)
        n.create_subscription(IMUStatus, Topics.IMU_STATUS, self._on_imu_status, 10)
        # Publisher for robot punch commands
        self._robot_cmd_pub = n.create_publisher(
            RobotCommand, Topics.ROBOT_COMMAND, 10,
        )

    def publish_robot_command(
        self, punch_code: str, speed: str = "medium",
    ) -> None:
        """Publish a RobotCommand to execute a punch."""
        if self._robot_cmd_pub is None:
            return
        msg = RobotCommand()
        msg.command_type = "punch"
        msg.punch_code = punch_code
        msg.speed = speed
        self._robot_cmd_pub.publish(msg)

    # ── Callbacks ───────────────────────────────────────────────────────

    def _on_punch(self, msg: Any) -> None:
        self.punch_confirmed.emit({
            "punch_type": msg.punch_type,
            "pad": msg.pad,
            "level": msg.level,
            "force": msg.force_normalized,
            "cv_confidence": msg.cv_confidence,
            "imu_confirmed": msg.imu_confirmed,
            "cv_confirmed": msg.cv_confirmed,
        })

    def _on_defense(self, msg: Any) -> None:
        self.defense_event.emit({
            "arm": msg.arm,
            "robot_punch_code": msg.robot_punch_code,
            "struck": msg.struck,
            "defense_type": msg.defense_type,
        })

    def _on_drill_progress(self, msg: Any) -> None:
        self.drill_progress.emit({
            "combos_completed": msg.combos_completed,
            "combos_remaining": msg.combos_remaining,
            "overall_accuracy": msg.overall_accuracy,
            "current_streak": msg.current_streak,
            "best_streak": msg.best_streak,
        })

    def _on_session_state(self, msg: Any) -> None:
        self.session_state_changed.emit(msg.state, msg.mode)

    def _on_coach_tip(self, msg: Any) -> None:
        self.coach_tip.emit(msg.tip_text, msg.tip_type)

    def _on_nav(self, msg: Any) -> None:
        self.nav_command.emit(msg.command)

    def _on_imu_status(self, msg: Any) -> None:
        self.imu_status.emit({
            "left_pad": msg.left_pad_connected,
            "centre_pad": msg.centre_pad_connected,
            "right_pad": msg.right_pad_connected,
            "head_pad": msg.head_pad_connected,
            "left_arm": msg.left_arm_connected,
            "right_arm": msg.right_arm_connected,
            "is_simulator": msg.is_simulator,
        })


# ── Public bridge (main-thread object) ─────────────────────────────────────

class GuiBridge(QObject):
    """Main-thread facade that owns the background ROS worker thread.

    All Qt signals are forwarded from the worker so page widgets can
    connect without touching threading details.
    """

    # Forwarded signals (same signatures as worker)
    punch_confirmed = Signal(dict)
    defense_event = Signal(dict)
    drill_progress = Signal(dict)
    session_state_changed = Signal(str, str)
    coach_tip = Signal(str, str)
    nav_command = Signal(str)
    imu_status = Signal(dict)
    cv_status = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_RosWorker] = None
        self.online: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background ROS thread."""
        if not ROS_AVAILABLE:
            logger.info("GuiBridge running in offline mode")
            return

        self._thread = QThread()
        self._worker = _RosWorker()
        self._worker.moveToThread(self._thread)

        # Forward all worker signals to bridge signals
        self._worker.punch_confirmed.connect(self.punch_confirmed)
        self._worker.defense_event.connect(self.defense_event)
        self._worker.drill_progress.connect(self.drill_progress)
        self._worker.session_state_changed.connect(self.session_state_changed)
        self._worker.coach_tip.connect(self.coach_tip)
        self._worker.nav_command.connect(self.nav_command)
        self._worker.imu_status.connect(self.imu_status)
        self._worker.cv_status.connect(self.cv_status)

        self._thread.started.connect(self._worker.start_spinning)
        self._thread.start()
        self.online = True
        logger.info("GuiBridge started")

    def shutdown(self) -> None:
        """Stop the ROS thread gracefully."""
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self.online = False
        logger.info("GuiBridge shut down")

    # ── Service client wrappers ─────────────────────────────────────────

    def call_start_session(
        self, mode: str, difficulty: str, config_json: str, username: str,
        callback: Callable[[bool, str, str], None],
    ) -> None:
        """Call StartSession service asynchronously.

        *callback* receives (success, session_id, message) on the ROS thread.
        """
        if not self._is_ready():
            callback(False, "", "ROS offline")
            return
        cli = self._worker.node.create_client(StartSession, Services.START_SESSION)
        req = StartSession.Request()
        req.mode = mode
        req.difficulty = difficulty
        req.config_json = config_json
        req.username = username
        future = cli.call_async(req)
        future.add_done_callback(
            lambda f: callback(f.result().success, f.result().session_id, f.result().message)
        )

    def call_end_session(
        self, session_id: str,
        callback: Callable[[bool, str, str], None],
    ) -> None:
        """Call EndSession service asynchronously."""
        if not self._is_ready():
            callback(False, "", "ROS offline")
            return
        cli = self._worker.node.create_client(EndSession, Services.END_SESSION)
        req = EndSession.Request()
        req.session_id = session_id
        future = cli.call_async(req)
        future.add_done_callback(
            lambda f: callback(f.result().success, f.result().summary_json, f.result().message)
        )

    def call_generate_llm(
        self, prompt: str, context_json: str, system_prompt_key: str,
        callback: Callable[[bool, str, float], None],
    ) -> None:
        """Call GenerateLlm service asynchronously."""
        if not self._is_ready():
            callback(False, "", 0.0)
            return
        cli = self._worker.node.create_client(GenerateLlm, Services.GENERATE_LLM)
        req = GenerateLlm.Request()
        req.prompt = prompt
        req.context_json = context_json
        req.system_prompt_key = system_prompt_key
        future = cli.call_async(req)
        future.add_done_callback(
            lambda f: callback(
                f.result().success,
                f.result().response,
                f.result().generation_time_sec,
            )
        )

    def publish_punch_command(
        self, punch_code: str, speed: str = "medium",
    ) -> None:
        """Publish a robot punch command (thread-safe via worker)."""
        if not self._is_ready():
            return
        self._worker.publish_robot_command(punch_code, speed)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _is_ready(self) -> bool:
        return (
            ROS_AVAILABLE
            and self._worker is not None
            and self._worker.node is not None
        )
