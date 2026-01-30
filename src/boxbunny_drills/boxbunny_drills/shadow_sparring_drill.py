#!/usr/bin/env python3
"""
Shadow Sparring Drill Manager.

Manages shadow sparring drills by tracking user combos against
target sequences using the action prediction system.
"""

import yaml
import time
from pathlib import Path
from typing import Optional, List, Dict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
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
        self.declare_parameter('drill_config', '')
        self.declare_parameter('idle_threshold_s', 1.0)  # Time to consider action complete
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('use_color_tracking', True)
        self.declare_parameter('glove_topic', 'glove_detections')
        self.declare_parameter('glove_distance_threshold_m', 0.8)
        self.declare_parameter('glove_velocity_threshold_mps', 1.0)
        self.declare_parameter('glove_debounce_s', 0.35)
        
        # Get parameters
        config_path = self.get_parameter('drill_config').value
        self.idle_threshold = self.get_parameter('idle_threshold_s').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.use_color_tracking = self.get_parameter('use_color_tracking').value
        self.glove_topic = self.get_parameter('glove_topic').value
        self.glove_distance_threshold_m = float(self.get_parameter('glove_distance_threshold_m').value)
        self.glove_velocity_threshold_mps = float(self.get_parameter('glove_velocity_threshold_mps').value)
        self.glove_debounce_s = float(self.get_parameter('glove_debounce_s').value)
        
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
        
        # Publishers
        self.progress_pub = self.create_publisher(DrillProgress, 'drill_progress', 10)
        self.state_pub = self.create_publisher(String, 'drill_state', 10)
        
        # Subscribers
        self.action_sub = self.create_subscription(
            ActionPrediction, 'action_prediction', self._on_action, 10)
        if self.use_color_tracking:
            self.glove_sub = self.create_subscription(
                GloveDetections, self.glove_topic, self._on_glove_detections, 10)
        
        # Services
        self.start_srv = self.create_service(
            StartDrill, 'start_drill', self._handle_start_drill)
        
        # LLM client for feedback
        self.llm_client = self.create_client(GenerateLLM, 'llm/generate')
        
        # Update timer (10Hz)
        self.timer = self.create_timer(0.1, self._update)
        
        self.get_logger().info('ShadowSparringDrill node ready')
    
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
        
        # Publish state
        state_msg = String()
        state_msg.data = 'shadow_sparring'
        self.state_pub.publish(state_msg)
        
        response.success = True
        response.message = f"Started drill: {drill_name}"
        self.get_logger().info(f"Starting drill: {drill_name}")
        
        return response
    
    def _on_action(self, msg: ActionPrediction):
        """Handle action prediction message."""
        if not self.active or self.current_drill is None:
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
        
        self._handle_detected_action(action, lock_after=True)

    def _on_glove_detections(self, msg: GloveDetections) -> None:
        """Handle glove detections (color tracking) as jab/cross."""
        if not self.active or self.current_drill is None:
            return
        now = time.time()
        for det in msg.detections:
            if det.distance_m > self.glove_distance_threshold_m:
                continue
            if det.approach_velocity_mps < self.glove_velocity_threshold_mps:
                continue
            if now - self._last_glove_punch_time[det.glove] < self.glove_debounce_s:
                continue
            self._last_glove_punch_time[det.glove] = now
            action = "jab" if det.glove == "left" else "cross"
            self._handle_detected_action(action, lock_after=False)

    def _handle_detected_action(self, action: str, lock_after: bool) -> None:
        """Check detected action against the expected sequence."""
        if self.current_step >= len(self.current_drill['sequence']):
            return
        expected = self.current_drill['sequence'][self.current_step]

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
            if self.current_step >= len(self.current_drill['sequence']):
                self._complete_drill(success=True)
        else:
            self.get_logger().info(
                f"Step {self.current_step + 1}: Expected {expected}, got {action}"
            )
    
    def _update(self):
        """Timer callback to check drill state and publish progress."""
        if not self.active or self.current_drill is None:
            return
        
        elapsed = time.time() - self.start_time
        time_limit = self.current_drill['time_limit_s']
        
        # Check timeout
        if elapsed > time_limit:
            self._complete_drill(success=False, reason='timeout')
            return
        
        # Publish progress
        msg = DrillProgress()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drill_name = self.current_drill['name']
        msg.current_step = self.current_step
        msg.total_steps = len(self.current_drill['sequence'])
        msg.expected_actions = self.current_drill['sequence']
        msg.detected_actions = self.detected_actions
        msg.step_completed = self.step_completed
        msg.elapsed_time_s = float(elapsed)
        msg.status = 'in_progress'
        
        self.progress_pub.publish(msg)
    
    def _complete_drill(self, success: bool, reason: str = ''):
        """Complete the drill and generate feedback."""
        elapsed = time.time() - self.start_time
        
        # Publish final progress
        msg = DrillProgress()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drill_name = self.current_drill['name']
        msg.current_step = self.current_step
        msg.total_steps = len(self.current_drill['sequence'])
        msg.expected_actions = self.current_drill['sequence']
        msg.detected_actions = self.detected_actions
        msg.step_completed = self.step_completed
        msg.elapsed_time_s = float(elapsed)
        msg.status = 'success' if success else ('timeout' if reason == 'timeout' else 'failed')
        
        self.progress_pub.publish(msg)
        
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
        self._request_llm_feedback(success, completed, total, elapsed)
        
        # Reset state
        self.active = False
        self.current_drill = None
    
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
