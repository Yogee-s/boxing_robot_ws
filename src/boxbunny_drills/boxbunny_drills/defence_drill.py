#!/usr/bin/env python3
"""
Defence Drill Manager.

Manages defence training drills by controlling motor positions
and tracking user blocking responses.
"""

import random
import time
from typing import Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from boxbunny_msgs.msg import ActionPrediction, DrillProgress, MotorCommand
from boxbunny_msgs.srv import StartDrill


class DefenceDrill(Node):
    """
    ROS 2 node for defence drill management.
    
    Controls motor positions to create blocking targets
    and uses action prediction to verify blocks.
    """
    
    def __init__(self):
        super().__init__('defence_drill')
        
        # Declare parameters
        self.declare_parameter('attack_interval_s', 2.5)  # Time between attacks
        self.declare_parameter('response_window_s', 1.5)  # Time to respond
        self.declare_parameter('num_attacks', 10)  # Default attacks per drill
        self.declare_parameter('confidence_threshold', 0.4)
        
        # Get parameters
        self.attack_interval = self.get_parameter('attack_interval_s').value
        self.response_window = self.get_parameter('response_window_s').value
        self.default_num_attacks = self.get_parameter('num_attacks').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # State
        self.active = False
        self.current_drill: Optional[str] = None
        self.attack_positions: List[int] = []
        self.current_attack = 0
        self.successful_blocks = 0
        self.missed_blocks = 0
        self.attack_start_time = 0.0
        self.drill_start_time = 0.0
        self.awaiting_block = False
        
        # Publishers
        self.motor_pub = self.create_publisher(MotorCommand, 'motor_command', 10)
        self.progress_pub = self.create_publisher(DrillProgress, 'drill_progress', 10)
        self.state_pub = self.create_publisher(String, 'drill_state', 10)
        
        # Subscribers
        self.action_sub = self.create_subscription(
            ActionPrediction, 'action_prediction', self._on_action, 10)
        
        # Services
        self.start_srv = self.create_service(
            StartDrill, 'start_defence_drill', self._handle_start_drill)
        
        # Timer for drill progression
        self.drill_timer = self.create_timer(0.1, self._update)
        
        self.get_logger().info('DefenceDrill node ready')
    
    def _handle_start_drill(self, request, response):
        """Handle StartDrill service request."""
        drill_name = request.drill_name
        num_attacks = request.repetitions if request.repetitions > 0 else self.default_num_attacks
        
        # Generate attack sequence based on drill type
        if drill_name == 'Head Defense':
            positions = [1, 2, 3]  # Head-level positions
        elif drill_name == 'Body Defense':
            positions = [4, 5, 6]  # Body-level positions  
        elif drill_name == 'Full Defense':
            positions = [1, 2, 3, 4, 5, 6]
        else:
            # Random pattern
            positions = list(range(1, 7))
        
        # Generate random attack sequence
        self.attack_positions = [random.choice(positions) for _ in range(num_attacks)]
        self.current_attack = 0
        self.successful_blocks = 0
        self.missed_blocks = 0
        self.drill_start_time = time.time()
        self.awaiting_block = False
        self.current_drill = drill_name
        self.active = True
        
        # Publish state
        state_msg = String()
        state_msg.data = 'defence_drill'
        self.state_pub.publish(state_msg)
        
        response.success = True
        response.message = f"Started {drill_name} with {num_attacks} attacks"
        self.get_logger().info(response.message)
        
        # Start first attack
        self._send_attack()
        
        return response
    
    def _send_attack(self):
        """Send motor command for current attack position."""
        if self.current_attack >= len(self.attack_positions):
            self._complete_drill()
            return
        
        position = self.attack_positions[self.current_attack]
        
        # Publish motor command
        msg = MotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.target_position = position
        msg.speed = 0.8  # Fast attack
        msg.pattern = self.current_drill or 'random'
        
        self.motor_pub.publish(msg)
        
        self.attack_start_time = time.time()
        self.awaiting_block = True
        
        self.get_logger().info(f"Attack {self.current_attack + 1}: Position {position}")
    
    def _on_action(self, msg: ActionPrediction):
        """Handle action prediction message."""
        if not self.active or not self.awaiting_block:
            return
        
        # Check for block action
        if msg.action_label == 'block' and msg.confidence >= self.confidence_threshold:
            elapsed = time.time() - self.attack_start_time
            
            if elapsed <= self.response_window:
                self.successful_blocks += 1
                self.get_logger().info(f"Block successful! ({elapsed:.2f}s)")
            else:
                self.missed_blocks += 1
                self.get_logger().info(f"Block too slow ({elapsed:.2f}s)")
            
            self.awaiting_block = False
            self.current_attack += 1
            
            # Small delay then next attack
            self.create_timer(
                self.attack_interval,
                self._send_attack_once,
                callback_group=None
            )
    
    def _send_attack_once(self):
        """One-shot attack sender."""
        self._send_attack()
    
    def _update(self):
        """Timer callback to check for missed blocks and publish progress."""
        if not self.active:
            return
        
        # Check timeout on current attack
        if self.awaiting_block:
            elapsed = time.time() - self.attack_start_time
            if elapsed > self.response_window + 0.5:  # Grace period
                self.missed_blocks += 1
                self.awaiting_block = False
                self.current_attack += 1
                self.get_logger().info("Block missed (timeout)")
                
                # Next attack after interval
                if self.current_attack < len(self.attack_positions):
                    self.create_timer(self.attack_interval, self._send_attack_once)
                else:
                    self._complete_drill()
        
        # Publish progress
        total = len(self.attack_positions)
        current = self.current_attack
        
        msg = DrillProgress()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drill_name = self.current_drill or 'Defence Drill'
        msg.current_step = current
        msg.total_steps = total
        msg.expected_actions = ['block'] * total
        msg.detected_actions = ['block'] * self.successful_blocks + ['miss'] * self.missed_blocks
        msg.step_completed = [True] * self.successful_blocks + [False] * self.missed_blocks
        msg.elapsed_time_s = float(time.time() - self.drill_start_time)
        msg.status = 'in_progress'
        
        self.progress_pub.publish(msg)
    
    def _complete_drill(self):
        """Complete the defence drill."""
        elapsed = time.time() - self.drill_start_time
        total = len(self.attack_positions)
        
        # Final progress message
        msg = DrillProgress()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drill_name = self.current_drill or 'Defence Drill'
        msg.current_step = total
        msg.total_steps = total
        msg.expected_actions = ['block'] * total
        msg.detected_actions = ['block'] * self.successful_blocks + ['miss'] * self.missed_blocks
        msg.step_completed = [True] * self.successful_blocks + [False] * self.missed_blocks
        msg.elapsed_time_s = float(elapsed)
        msg.status = 'success' if self.successful_blocks >= total * 0.7 else 'failed'
        
        self.progress_pub.publish(msg)
        
        # Return motor to home
        home_msg = MotorCommand()
        home_msg.header.stamp = self.get_clock().now().to_msg()
        home_msg.target_position = 0  # Home position
        home_msg.speed = 0.5
        home_msg.pattern = 'return_home'
        self.motor_pub.publish(home_msg)
        
        # Publish idle state
        state_msg = String()
        state_msg.data = 'idle'
        self.state_pub.publish(state_msg)
        
        self.get_logger().info(
            f"Drill complete: {self.successful_blocks}/{total} blocks "
            f"({self.missed_blocks} missed) in {elapsed:.1f}s"
        )
        
        self.active = False


def main(args=None):
    rclpy.init(args=args)
    node = DefenceDrill()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
