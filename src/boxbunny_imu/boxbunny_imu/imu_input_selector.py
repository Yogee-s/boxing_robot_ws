#!/usr/bin/env python3
"""
IMU Input Selector Node.

Converts IMU punch events to GUI menu selections,
allowing users to navigate using punches.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
from std_srvs.srv import SetBool
from boxbunny_msgs.msg import ImuPunch


class ImuInputSelector(Node):
    """
    ROS 2 node that maps IMU punch events to menu selections.
    
    Punch type mapping:
    - jab = 1
    - cross = 2
    - left_hook = 3
    - right_hook = 4
    - left_uppercut = 5
    - right_uppercut = 6
    """
    
    PUNCH_TO_SELECTION = {
        'jab': 1,
        'cross': 2,
        'left_hook': 3,
        'right_hook': 4,
        'left_uppercut': 5,
        'right_uppercut': 6,
    }
    
    def __init__(self):
        super().__init__('imu_input_selector')
        
        # Declare parameters
        self.declare_parameter('enabled', False)  # Disabled by default
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('cooldown_s', 1.0)  # Prevent rapid selections
        
        # Get parameters
        self.enabled = self.get_parameter('enabled').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.cooldown = self.get_parameter('cooldown_s').value
        
        # State
        self.last_selection_time = 0.0
        
        # Publishers
        self.selection_pub = self.create_publisher(Int32, 'imu_selection', 10)
        self.enabled_pub = self.create_publisher(Bool, 'imu_input_enabled', 10)
        
        # Subscribers
        self.punch_sub = self.create_subscription(
            ImuPunch, 'imu/punch', self._on_punch, 10)
        
        # Services
        self.enable_srv = self.create_service(
            SetBool, 'imu_input_selector/enable', self._handle_enable)
        
        # Publish initial state
        self._publish_enabled_state()
        
        self.get_logger().info(
            f"ImuInputSelector ready (enabled={self.enabled})"
        )
    
    def _handle_enable(self, request, response):
        """Handle enable/disable service."""
        self.enabled = request.data
        self._publish_enabled_state()
        
        response.success = True
        response.message = f"IMU input {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(response.message)
        
        return response
    
    def _publish_enabled_state(self):
        """Publish current enabled state."""
        msg = Bool()
        msg.data = self.enabled
        self.enabled_pub.publish(msg)
    
    def _on_punch(self, msg: ImuPunch):
        """Handle IMU punch message."""
        if not self.enabled:
            return
        
        # Check confidence
        if msg.confidence < self.confidence_threshold:
            return
        
        # Check cooldown
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_selection_time < self.cooldown:
            return
        
        # Map punch type to selection
        punch_type = msg.punch_type.lower()
        selection = self.PUNCH_TO_SELECTION.get(punch_type)
        
        if selection is not None:
            # Publish selection
            sel_msg = Int32()
            sel_msg.data = selection
            self.selection_pub.publish(sel_msg)
            
            self.last_selection_time = now
            self.get_logger().info(f"IMU selection: {punch_type} -> {selection}")


def main(args=None):
    rclpy.init(args=args)
    node = ImuInputSelector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
