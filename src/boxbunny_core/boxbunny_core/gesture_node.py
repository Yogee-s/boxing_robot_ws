"""Gesture-based navigation node for BoxBunny.

Uses MediaPipe Hands to detect hand gestures from the RealSense D435i camera,
converting them into NavCommand messages for GUI navigation. Disabled by default.
"""

import logging
import time
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    from cv_bridge import CvBridge
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

from sensor_msgs.msg import Image
from boxbunny_msgs.msg import NavCommand, SessionState

logger = logging.getLogger("boxbunny.gesture_node")

# MediaPipe hand landmark indices
_WRIST = 0
_THUMB_TIP, _THUMB_IP = 4, 3
_INDEX_TIP, _INDEX_PIP = 8, 6
_MIDDLE_TIP, _MIDDLE_PIP = 12, 10
_RING_TIP, _RING_PIP = 16, 14
_PINKY_TIP, _PINKY_PIP = 20, 18

_FINGER_IDS = [
    (_INDEX_TIP, _INDEX_PIP), (_MIDDLE_TIP, _MIDDLE_PIP),
    (_RING_TIP, _RING_PIP), (_PINKY_TIP, _PINKY_PIP),
]

_GESTURE_TO_COMMAND = {"open_palm": "enter", "thumbs_up": "enter", "peace": "back"}


def _thumb_extended(lm) -> bool:
    """Thumb extended if tip is further from wrist than IP joint."""
    wx = lm[_WRIST].x
    return abs(lm[_THUMB_TIP].x - wx) > abs(lm[_THUMB_IP].x - wx)


def classify_gesture(lm) -> Optional[str]:
    """Classify a static gesture. Returns gesture name or None."""
    thumb = _thumb_extended(lm)
    fingers = [lm[t].y < lm[p].y for t, p in _FINGER_IDS]
    if thumb and all(fingers):
        return "open_palm"
    if thumb and not any(fingers):
        return "thumbs_up"
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return "peace"
    return None


class GestureNode(Node):
    """ROS 2 node detecting hand gestures for GUI navigation."""

    def __init__(self) -> None:
        super().__init__("gesture_node")

        # -- Parameters --
        self.declare_parameter("enabled", False)
        self.declare_parameter("hold_duration_s", 0.7)
        self.declare_parameter("cooldown_s", 1.5)
        self.declare_parameter("min_confidence", 0.7)
        self.declare_parameter("swipe_threshold_px", 100.0)
        self.declare_parameter("process_interval", 3)

        self._enabled: bool = self.get_parameter("enabled").value
        self._hold_duration: float = self.get_parameter("hold_duration_s").value
        self._cooldown: float = self.get_parameter("cooldown_s").value
        self._min_confidence: float = self.get_parameter("min_confidence").value
        self._swipe_thresh: float = self.get_parameter("swipe_threshold_px").value
        self._process_every: int = self.get_parameter("process_interval").value

        # -- Graceful degradation --
        if not _MP_AVAILABLE:
            logger.warning("mediapipe not installed -- gesture node disabled")
            self._enabled = False
        if not _CV_AVAILABLE:
            logger.warning("cv2/cv_bridge not available -- gesture node disabled")
            self._enabled = False

        # -- State --
        self._session_active = False
        self._frame_count = 0
        self._last_trigger_time = 0.0
        self._current_gesture: Optional[str] = None
        self._gesture_start: float = 0.0
        self._wrist_history: deque = deque(maxlen=10)
        self._bridge = CvBridge() if _CV_AVAILABLE else None

        # -- MediaPipe hands --
        self._hands = None
        if _MP_AVAILABLE and self._enabled:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self._min_confidence,
                min_tracking_confidence=self._min_confidence,
            )

        # -- Publishers --
        self._pub_nav = self.create_publisher(NavCommand, "/boxbunny/imu/nav_event", 10)
        self._pub_status = self.create_publisher(String, "/boxbunny/gesture/status", 10)

        # -- Subscribers --
        self.create_subscription(
            Image, "/camera/color/image_raw", self._on_image, 10,
        )
        self.create_subscription(
            SessionState, "/boxbunny/session/state", self._on_session_state, 10,
        )

        # -- Status heartbeat --
        self.create_timer(1.0, self._publish_status)

        logger.info("Gesture node initialised (enabled=%s)", self._enabled)

    def _on_session_state(self, msg: SessionState) -> None:
        """Suspend gesture processing during active sessions."""
        self._session_active = msg.state in ("countdown", "active", "rest")

    def _on_image(self, msg: Image) -> None:
        """Process incoming camera frames for gesture detection."""
        if not self._enabled or self._session_active or self._hands is None:
            return

        self._frame_count += 1
        if self._frame_count % self._process_every != 0:
            return

        if self._bridge is None:
            return
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        results = self._hands.process(frame)

        if not results.multi_hand_landmarks:
            self._current_gesture = None
            self._wrist_history.clear()
            return

        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        # Track wrist for swipe detection (pixel coordinates)
        h, w = frame.shape[:2]
        wrist_px = lm[_WRIST].x * w
        self._wrist_history.append((time.time(), wrist_px))

        # Check swipe first (dynamic gesture)
        swipe = self._detect_swipe()
        if swipe:
            self._try_trigger(swipe)
            return

        # Static gesture classification
        gesture = classify_gesture(lm)
        now = time.time()

        if gesture is None:
            self._current_gesture = None
            return

        if gesture != self._current_gesture:
            self._current_gesture = gesture
            self._gesture_start = now
            return

        # Hold requirement
        if now - self._gesture_start >= self._hold_duration:
            command = _GESTURE_TO_COMMAND.get(gesture)
            if command:
                self._try_trigger(command)
            self._current_gesture = None

    def _detect_swipe(self) -> Optional[str]:
        """Detect a lateral swipe from wrist position history."""
        if len(self._wrist_history) < 4:
            return None
        t_start, x_start = self._wrist_history[0]
        t_end, x_end = self._wrist_history[-1]
        if t_end - t_start > 1.0:
            return None  # too slow
        dx = x_end - x_start
        if abs(dx) >= self._swipe_thresh:
            self._wrist_history.clear()
            return "next" if dx > 0 else "prev"
        return None

    def _try_trigger(self, command: str) -> None:
        """Publish a NavCommand if cooldown has elapsed."""
        now = time.time()
        if now - self._last_trigger_time < self._cooldown:
            return
        self._last_trigger_time = now
        nav_msg = NavCommand()
        nav_msg.timestamp = now
        nav_msg.command = command
        self._pub_nav.publish(nav_msg)
        logger.info("Gesture nav: %s", command)

    def _publish_status(self) -> None:
        """Publish gesture node status."""
        msg = String()
        if not self._enabled:
            msg.data = "disabled"
        elif self._session_active:
            msg.data = "inactive"
        else:
            msg.data = "active"
        self._pub_status.publish(msg)


def main(args=None) -> None:
    """Entry point for the gesture node."""
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._hands is not None:
            node._hands.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
