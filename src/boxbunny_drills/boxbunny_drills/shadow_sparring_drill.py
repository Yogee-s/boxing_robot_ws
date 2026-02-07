#!/usr/bin/env python3
"""
Shadow Sparring Drill Manager.

Manages shadow sparring drills by tracking user combos against
target sequences using the action prediction system.
"""

import csv
import os
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

from typing import Optional, List, Dict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
from std_srvs.srv import Trigger
from boxbunny_msgs.msg import ActionPrediction, DrillProgress, DrillDefinition, GloveDetections
from boxbunny_msgs.srv import StartDrill, GenerateLLM


class ShadowSparringDrill(Node):
    """
    ROS 2 node for shadow sparring drill management.
    
    Compares detected actions against target combo sequences
    and tracks progress/success.
    """
    
    def __init__(self):
        super().__init__('shadow_sparring_drill')
        
        # Declare parameters
        data_root = self._default_data_root()
        self.declare_parameter('drill_config', '')
        self.declare_parameter('idle_threshold_s', 1.0)  # Time to consider action complete
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('use_color_tracking', True)
        self.declare_parameter('glove_topic', 'glove_detections')
        self.declare_parameter('glove_distance_threshold_m', 0.45)
        self.declare_parameter('glove_velocity_threshold_mps', 1.5)
        self.declare_parameter('glove_debounce_s', 0.45)
        self.declare_parameter('combo_cooldown_s', 1.5)
        self.declare_parameter('log_dir', str(data_root / "shadow_sparring"))
        
        # Get parameters
        config_path = self.get_parameter('drill_config').value
        self.idle_threshold = self.get_parameter('idle_threshold_s').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.use_color_tracking = self.get_parameter('use_color_tracking').value
        self.glove_topic = self.get_parameter('glove_topic').value
        self.glove_distance_threshold_m = float(self.get_parameter('glove_distance_threshold_m').value)
        self.glove_velocity_threshold_mps = float(self.get_parameter('glove_velocity_threshold_mps').value)
        self.glove_debounce_s = float(self.get_parameter('glove_debounce_s').value)
        self.combo_cooldown_s = float(self.get_parameter('combo_cooldown_s').value)
        
        # Load drill definitions
        self.drills: Dict[str, Dict] = {}
        self._load_drill_config(config_path)
        
        # State
        self.active = False
        self.current_drill: Optional[Dict] = None
        self.current_step = 0
        self.step_completed: List[bool] = []
        self.detected_actions: List[str] = []
        self.start_time = 0.0
        self.last_action = 'idle'
        self.last_action_time = 0.0
        self.action_locked = False  # Prevent rapid duplicate detections
        self._last_glove_punch_time = {"left": 0.0, "right": 0.0}
        self._combo_cooldown_until = 0.0
        self._master_log_path: Optional[str] = None
        self._current_user_id = 1
        self._drill_session_start: Optional[str] = None
        
        # Survival Mode State
        self.max_attempts = 3
        self.attempts_left = 3
        self.iterations = 0
        self.failures = 0

        
        # Publishers
        self.progress_pub = self.create_publisher(DrillProgress, 'drill_progress', 10)
        self.state_pub = self.create_publisher(String, 'drill_state', 10)
        self.summary_pub = self.create_publisher(String, 'drill_summary', 10)

        
        # Subscribers
        self.action_sub = self.create_subscription(
            ActionPrediction, 'action_prediction', self._on_action, 10)
        if self.use_color_tracking:
            self.glove_sub = self.create_subscription(
                GloveDetections, self.glove_topic, self._on_glove_detections, 10)
        
        # Services
        self.start_srv = self.create_service(
            StartDrill, 'start_drill', self._handle_start_drill)
        self.stop_srv = self.create_service(
            Trigger, 'stop_shadow_drill', self._handle_stop_drill)
        self.new_user_srv = self.create_service(
            Trigger, 'shadow_drill/new_user', self._on_new_user)
        
        # LLM client for feedback
        self.llm_client = self.create_client(GenerateLLM, 'llm/generate')
        
        # Update timer (10Hz)
        self.timer = self.create_timer(0.1, self._update)
        
        self.get_logger().info('ShadowSparringDrill node ready')

    @staticmethod
    def _default_data_root() -> Path:
        try:
            here = Path(__file__).resolve()
            for parent in here.parents:
                if parent.name == "boxing_robot_ws":
                    return parent / "data"
        except Exception:
            pass
        return Path(os.path.expanduser("~/boxbunny_data"))

    def _open_log(self, drill_name: str) -> None:
        """Initialize master log file path (creates file with header if needed)."""
        log_dir = Path(os.path.expanduser(str(self.get_parameter("log_dir").value)))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a single master log file
        self._master_log_path = str(log_dir / "shadow_sparring_log.csv")
        self._drill_session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create header if file doesn't exist
        if not os.path.exists(self._master_log_path):
            with open(self._master_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "user_id",
                        "session_timestamp",
                        "timestamp_unix",
                        "elapsed_s",
                        "drill_name",
                        "iteration",
                        "step_index",
                        "expected_action",
                        "detected_action",
                        "correct",
                        "attempts_left",
                        "failures",
                        "combo_complete",
                        "source",
                        "confidence",
                    ]
                )

    def _on_new_user(self, request, response):
        """Mark new user by inserting empty separator row and incrementing user ID."""
        self._current_user_id += 1
        
        # Insert empty row as separator
        if self._master_log_path and os.path.exists(self._master_log_path):
            with open(self._master_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([])  # Empty row separator
        
        response.success = True
        response.message = f"New user marked. User ID: {self._current_user_id}"
        self.get_logger().info(f"New user started. ID: {self._current_user_id}")
        return response

    def _log_attempt(
        self,
        *,
        step_index: int,
        expected: str,
        detected: str,
        correct: bool,
        combo_complete: bool,
        source: str,
        confidence: Optional[float],
    ) -> None:
        """Append attempt row to master log file."""
        if not self._master_log_path:
            return
        elapsed = time.time() - self.start_time
        with open(self._master_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self._current_user_id,
                    self._drill_session_start or "",
                    time.time(),
                    f"{elapsed:.3f}",
                    self.current_drill["name"] if self.current_drill else "",
                    self.iterations,
                    step_index,
                    expected,
                    detected,
                    int(bool(correct)),
                    self.attempts_left,
                    self.failures,
                    int(bool(combo_complete)),
                    source,
                    f"{confidence:.3f}" if confidence is not None else "",
                ]
            )
    
    def _load_drill_config(self, config_path: str):
        """Load drill definitions from YAML config."""
        if not config_path:
            # Try default path
            try:
                pkg_share = get_package_share_directory('boxbunny_drills')
                config_path = str(Path(pkg_share) / 'config' / 'drill_definitions.yaml')
            except Exception:
                self.get_logger().warn('No drill config found, using defaults')
                self._use_default_drills()
                return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Parse shadow sparring drills
            for drill in config.get('shadow_sparring_drills', []):
                name = drill['name']
                self.drills[name] = {
                    'name': name,
                    'sequence': drill['sequence'],
                    'time_limit_s': drill.get('time_limit_s', 10.0),
                }
            
            self.get_logger().info(f'Loaded {len(self.drills)} drills from {config_path}')
            
        except Exception as e:
            self.get_logger().warn(f'Failed to load drill config: {e}')
            self._use_default_drills()
    
    def _use_default_drills(self):
        """Set up default drill definitions."""
        self.drills = {
            '1-1-2 Combo': {
                'name': '1-1-2 Combo',
                'sequence': ['jab', 'jab', 'cross'],
                'time_limit_s': 6.0,
            },
            'Jab-Cross-Hook': {
                'name': 'Jab-Cross-Hook',
                'sequence': ['jab', 'cross', 'left_hook'],
                'time_limit_s': 7.0,
            },
            'Double Jab': {
                'name': 'Double Jab',
                'sequence': ['jab', 'jab'],
                'time_limit_s': 4.0,
            },
            'Cross-Hook-Cross': {
                'name': 'Cross-Hook-Cross',
                'sequence': ['cross', 'left_hook', 'cross'],
                'time_limit_s': 7.0,
            },
        }
        self.get_logger().info(f'Using {len(self.drills)} default drills')
    
    def _handle_start_drill(self, request, response):
        """Handle StartDrill service request."""
        drill_name = request.drill_name
        
        if drill_name not in self.drills:
            response.success = False
            response.message = f"Unknown drill: {drill_name}. Available: {list(self.drills.keys())}"
            return response
        
        # Start the drill
        self.current_drill = self.drills[drill_name]
        self.current_step = 0
        self.step_completed = [False] * len(self.current_drill['sequence'])
        self.detected_actions = []
        self.start_time = time.time()
        self.active = True
        self.action_locked = False
        self._reset_tracking_state()
        
        # Reset Survival Mode State
        self.attempts_left = self.max_attempts
        self.iterations = 0
        self.failures = 0
        self._open_log(drill_name)

        
        # Publish state
        state_msg = String()
        state_msg.data = 'shadow_sparring'
        self.state_pub.publish(state_msg)
        
        response.success = True
        response.message = f"Started drill: {drill_name}"
        self.get_logger().info(f"Starting drill: {drill_name}")
        
        return response

    def _handle_stop_drill(self, request, response):
        """Stop shadow sparring drill and reset tracking."""
        if not self.active:
            response.success = True
            response.message = "Shadow sparring already stopped."
            return response
        self._complete_drill(success=False, reason='stopped')
        self._reset_tracking_state()
        response.success = True
        response.message = "Shadow sparring stopped."
        return response

    def _reset_tracking_state(self) -> None:
        """Reset tracking-related state to avoid stale detections."""
        self.current_step = 0
        self.step_completed = [False] * len(self.current_drill['sequence']) if self.current_drill else []
        self.detected_actions = []
        self.last_action = 'idle'
        self.last_action_time = 0.0
        self.action_locked = False
        self._last_glove_punch_time = {"left": 0.0, "right": 0.0}
        self._pending_punch = {"glove": None, "start_time": 0.0, "count": 0}
        self._combo_cooldown_until = 0.0
    
    def _on_action(self, msg: ActionPrediction):
        """Handle action prediction message."""
        if not self.active or self.current_drill is None:
            return
        if self.attempts_left <= 0:
            return
        if time.time() < self._combo_cooldown_until:
            return
        if self.use_color_tracking:
            return
        
        action = msg.action_label
        confidence = msg.confidence
        
        # Skip low confidence or idle
        if confidence < self.confidence_threshold or action == 'idle':
            # Check if we should unlock action detection after idle
            if action == 'idle':
                now = time.time()
                if now - self.last_action_time > self.idle_threshold:
                    self.action_locked = False
            return
        
        # Avoid rapid duplicate detections
        if self.action_locked:
            return
        
        # Get expected action for current step
        if self.current_step >= len(self.current_drill['sequence']):
            return
        
        expected = self.current_drill['sequence'][self.current_step]
        
        # Record detected action
        self.detected_actions.append(action)
        self.last_action = action
        self.last_action_time = time.time()
        
        self._handle_detected_action(
            action,
            lock_after=True,
            source="action_prediction",
            confidence=confidence,
        )

    def _on_glove_detections(self, msg: GloveDetections) -> None:
        """Handle glove detections (color tracking) as jab/cross.
        
        A punch is only registered when:
        1. One glove is within the distance threshold
        2. That glove is significantly ahead of the other glove (by at least 0.15m)
        This prevents false positives when both gloves are slightly forward.
        """
        if not self.active or self.current_drill is None:
            return
        now = time.time()
        if self.attempts_left <= 0:
            return
        if now < self._combo_cooldown_until:
            return
        
        # Collect distances for both gloves
        left_dist = 999.0
        right_dist = 999.0
        left_det = None
        right_det = None
        
        for det in msg.detections:
            if det.glove == "left" and det.distance_m < left_dist:
                left_dist = det.distance_m
                left_det = det
            elif det.glove == "right" and det.distance_m < right_dist:
                right_dist = det.distance_m
                right_det = det
        
        # Minimum difference required between gloves to register a punch
        MIN_GLOVE_DIFFERENCE = 0.18  # meters - increased to reduce false positives
        
        # Check if left glove is punching (close AND significantly ahead of right)
        left_punching = (
            left_det is not None and
            left_dist < self.glove_distance_threshold_m and 
            (right_dist - left_dist) >= MIN_GLOVE_DIFFERENCE
        )
        
        # Check if right glove is punching (close AND significantly ahead of left)
        right_punching = (
            right_det is not None and
            right_dist < self.glove_distance_threshold_m and 
            (left_dist - right_dist) >= MIN_GLOVE_DIFFERENCE
        )
        
        # Determine which glove to process
        punch_det = None
        if left_punching and right_punching:
            # Both qualify - pick the closer one
            punch_det = left_det if left_dist < right_dist else right_det
        elif left_punching:
            punch_det = left_det
        elif right_punching:
            punch_det = right_det
        
        if punch_det is None:
            # No valid punch - reset pending if gloves moved away
            if not hasattr(self, '_pending_punch'):
                self._pending_punch = {"glove": None, "start_time": 0.0, "count": 0}
            else:
                self._pending_punch = {"glove": None, "start_time": 0.0, "count": 0}
            return
        
        det = punch_det
        
        # For very close punches (under 0.35m), skip velocity check entirely
        very_close = det.distance_m <= 0.35
        
        # Skip if velocity too slow (unless very close)
        if not very_close and det.approach_velocity_mps < self.glove_velocity_threshold_mps:
            return
            
        # Skip if within debounce period
        if now - self._last_glove_punch_time[det.glove] < self.glove_debounce_s:
            return
        
        # Additional validation: require consecutive frames for punch confirmation
        if not hasattr(self, '_pending_punch'):
            self._pending_punch = {"glove": None, "start_time": 0.0, "count": 0}
        
        # Require at least 2 consecutive detections to confirm punch
        if self._pending_punch["glove"] == det.glove:
            self._pending_punch["count"] += 1
        else:
            self._pending_punch = {"glove": det.glove, "start_time": now, "count": 1}
        
        # Only register punch after confirmation threshold
        # Require 3 consecutive frames for punch confirmation (increased from 2)
        if self._pending_punch["count"] >= 3:
            self._last_glove_punch_time[det.glove] = now
            action = "jab" if det.glove == "left" else "cross"
            self._handle_detected_action(
                action,
                lock_after=False,
                source="color_tracking",
                confidence=None,
            )
            # Reset pending punch after registering
            self._pending_punch = {"glove": None, "start_time": 0.0, "count": 0}

    def _handle_detected_action(
        self,
        action: str,
        lock_after: bool,
        *,
        source: str,
        confidence: Optional[float],
    ) -> None:
        """Check detected action against the expected sequence."""
        if self.current_step >= len(self.current_drill['sequence']):
            return
        expected = self.current_drill['sequence'][self.current_step]
        step_index = self.current_step + 1

        self.detected_actions.append(action)
        self.last_action = action
        self.last_action_time = time.time()

        if action == expected:
            self.step_completed[self.current_step] = True
            self.current_step += 1
            if lock_after:
                self.action_locked = True
            
            self.get_logger().info(
                f"Step {self.current_step}/{len(self.current_drill['sequence'])}: "
                f"Detected {action} ✓"
            )
            
            # Check for sequence completion
            combo_complete = self.current_step >= len(self.current_drill['sequence'])
            if combo_complete:
                self.iterations += 1
                self.get_logger().info(f"Combo Complete! Iterations: {self.iterations}")
                # Extended cooldown after combo completion to prevent accidental re-triggers
                self._combo_cooldown_until = time.time() + self.combo_cooldown_s + 0.5
                self._reset_sequence(keep_active=True)
            self._log_attempt(
                step_index=step_index,
                expected=expected,
                detected=action,
                correct=True,
                combo_complete=combo_complete,
                source=source,
                confidence=confidence,
            )
        else:
            self.get_logger().info(
                f"Step {self.current_step + 1}: Expected {expected}, got {action}"
            )
            
            # Reduce attempts
            self.failures += 1
            self.attempts_left -= 1
            self.get_logger().warn(f"Wrong move! Attempts left: {self.attempts_left}")
            self._log_attempt(
                step_index=step_index,
                expected=expected,
                detected=action,
                correct=False,
                combo_complete=False,
                source=source,
                confidence=confidence,
            )
            
            if self.attempts_left <= 0:
                self._complete_drill(success=False, reason='failed')
                return
            else:
                # Publish a status update for the GUI to show 'X'
                self._publish_progress(status='wrong_action')
                # Reset sequence for next try
                self._reset_sequence(keep_active=True)
    
    def _reset_sequence(self, keep_active: bool):
        """Reset the current combo sequence but keep drill active."""
        self.current_step = 0
        self.step_completed = [False] * len(self.current_drill['sequence'])
        # Keep detected actions for history if needed, or clear? 
        # For simplicity, clear detected actions for the new sequence
        self.detected_actions = [] 
        self.action_locked = False
        self.start_time = time.time()  # Reset time for the new iteration? Or keep global?
        # Let's keep global start time for total drill time, but maybe reset idle check?
        self.last_action_time = time.time()

    
    def _update(self):
        """Timer callback to check drill state and publish progress."""
        if not self.active or self.current_drill is None:
            return
        
        elapsed = time.time() - self.start_time
        time_limit = self.current_drill['time_limit_s']
        
        # Survival mode - no time limit, only attempts
        # But maybe we still want to track time or have a per-move timeout?
        # For now, let's remove strict time limit or use it as "idle timeout"?
        # Keeping time limit for now but maybe extended? 
        # Actually plan said "Remove or adjust global time limit". Let's ignore it for now or make it very long.
        # if elapsed > time_limit: ...
        
        # Publish progress
        self._publish_progress(status='in_progress')
        
        # Publish summary
        summary_msg = String()
        summary_msg.data = json.dumps({
            "attempts": self.attempts_left,
            "max_attempts": self.max_attempts,
            "iterations": self.iterations,
            "failures": self.failures,
            "score": self.iterations * 100 
        })
        self.summary_pub.publish(summary_msg)

    def _publish_progress(self, status: str):
        """Helper to publish DrillProgress."""
        msg = DrillProgress()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drill_name = self.current_drill['name']
        msg.current_step = self.current_step
        msg.total_steps = len(self.current_drill['sequence'])
        msg.expected_actions = self.current_drill['sequence']
        msg.detected_actions = self.detected_actions
        msg.step_completed = self.step_completed
        msg.elapsed_time_s = float(time.time() - self.start_time)
        msg.status = status
        self.progress_pub.publish(msg)

    def _complete_drill(self, success: bool, reason: str = ''):
        """Complete the drill and generate feedback."""
        elapsed = time.time() - self.start_time
        
        # Publish final progress
        status = 'success' if success else ('timeout' if reason == 'timeout' else 'failed')
        self._publish_progress(status)


        
        # Publish state
        state_msg = String()
        state_msg.data = 'idle'
        self.state_pub.publish(state_msg)
        
        # Log result
        completed = sum(self.step_completed)
        total = len(self.step_completed)
        self.get_logger().info(
            f"Drill complete: {self.current_drill['name']} - "
            f"{'SUCCESS' if success else 'FAILED'} "
            f"({completed}/{total} steps, {elapsed:.2f}s)"
        )
        
        # Request LLM feedback (async)
        if reason != 'stopped':
            self._request_llm_feedback(success, completed, total, elapsed)
        
        # Reset state
        self.active = False
        self.current_drill = None
        self._reset_tracking_state()
    
    def _request_llm_feedback(self, success: bool, completed: int, total: int, elapsed: float):
        """Request performance feedback from LLM."""
        if not self.llm_client.service_is_ready():
            return
        
        prompt = (
            f"The user just completed a shadow sparring drill. "
            f"Results: {completed}/{total} steps completed in {elapsed:.1f} seconds. "
            f"{'Success!' if success else 'They ran out of time or missed steps.'} "
            f"Give brief, encouraging feedback (2-3 sentences)."
        )
        
        request = GenerateLLM.Request()
        request.prompt = prompt
        request.mode = 'coach'
        request.context = 'drill_feedback'
        
        self.llm_client.call_async(request)
    
    def get_available_drills(self) -> List[str]:
        """Return list of available drill names."""
        return list(self.drills.keys())


def main(args=None):
    rclpy.init(args=args)
    node = ShadowSparringDrill()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
