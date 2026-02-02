#!/usr/bin/env python3
"""
Simple RealSense camera node using pyrealsense2 directly.
Bypasses the ROS2 realsense wrapper which has version mismatch issues.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import time


class SimpleCameraNode(Node):
    def __init__(self):
        super().__init__('simple_camera_node')
        
        self.bridge = CvBridge()
        self.pipeline = None
        self.pipeline_started = False
        
        # Publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
        
        self.get_logger().info("Simple Camera Node initializing...")
        
        # Try to start the camera with retries
        self.startup_attempts = 0
        self.max_startup_attempts = 5
        
        # Use a timer for startup to not block constructor
        self.startup_timer = self.create_timer(2.0, self._try_start_camera)
        
        # Frame capture timer (will be started after camera is ready)
        self.capture_timer = None
        self.frame_count = 0
        self.align = None
        
    def _try_start_camera(self):
        """Attempt to start the camera pipeline."""
        if self.pipeline_started:
            self.startup_timer.cancel()
            return
            
        self.startup_attempts += 1
        self.get_logger().info(f"Camera startup attempt {self.startup_attempts}/{self.max_startup_attempts}...")
        
        try:
            # Configure RealSense pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Get device info
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                self.get_logger().warn("No RealSense device found, will retry...")
                if self.startup_attempts >= self.max_startup_attempts:
                    self.get_logger().error("Max startup attempts reached. Camera not available.")
                    self.startup_timer.cancel()
                return
            
            # Get first device
            device = devices[0]
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            
            self.get_logger().info(f"Found device: {name}")
            self.get_logger().info(f"Serial: {serial}")
            
            # Enable specific device by serial
            config.enable_device(serial)
            
            # Enable streams - prefer higher resolution, fall back if unsupported
            preferred_resolutions = [
                (1280, 720),
                (960, 540),
                (640, 480),
            ]
            started = False
            last_error = None
            for width, height in preferred_resolutions:
                try:
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
                    self.profile = self.pipeline.start(config)
                    self.pipeline_started = True
                    self.get_logger().info(f"✅ Camera stream set to {width}x{height} @ 30 FPS")
                    started = True
                    break
                except Exception as stream_err:
                    last_error = stream_err
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
                    self.pipeline = rs.pipeline()

            if not started:
                raise RuntimeError(f\"Unable to start RealSense at preferred resolutions: {last_error}\")
            
            # Align depth to color
            self.align = rs.align(rs.stream.color)
            
            self.get_logger().info("✅ RealSense pipeline started successfully!")
            
            # Cancel startup timer and start capture timer
            self.startup_timer.cancel()
            self.capture_timer = self.create_timer(1.0/30.0, self.capture_frame)
            
        except Exception as e:
            self.get_logger().warn(f"Camera startup failed: {e}")
            # Clean up failed pipeline
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except:
                    pass
                self.pipeline = None
            
            if self.startup_attempts >= self.max_startup_attempts:
                self.get_logger().error("Max startup attempts reached. Camera not available.")
                self.startup_timer.cancel()
        
    def capture_frame(self):
        if not self.pipeline_started or not self.pipeline:
            return
            
        try:
            # Wait for frames with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Get timestamp
            stamp = self.get_clock().now().to_msg()
            
            # Publish color image
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = 'camera_color_optical_frame'
            self.color_pub.publish(color_msg)
            
            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = 'camera_depth_optical_frame'
            self.depth_pub.publish(depth_msg)
            
            self.frame_count += 1
            if self.frame_count % 90 == 0:  # Log every 3 seconds
                self.get_logger().info(f"📹 Published {self.frame_count} frames")
                
        except Exception as e:
            self.get_logger().warn(f"Frame capture error: {e}")
    
    def destroy_node(self):
        self.get_logger().info("Stopping camera pipeline...")
        if self.capture_timer:
            self.capture_timer.cancel()
        if self.startup_timer:
            self.startup_timer.cancel()
        if self.pipeline and self.pipeline_started:
            try:
                self.pipeline.stop()
            except:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleCameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
