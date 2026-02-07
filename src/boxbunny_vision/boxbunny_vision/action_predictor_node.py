#!/usr/bin/env python3
"""
Action Predictor ROS 2 Node.

Bridges the action_prediction RGBD model to ROS 2 for real-time 
boxing action recognition and publishing predictions.
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from boxbunny_msgs.msg import ActionPrediction

# Add action_prediction to path
_ws_root = Path(__file__).resolve().parents[4]
_action_pred_root = _ws_root / 'action_prediction'
if str(_action_pred_root) not in sys.path:
    sys.path.insert(0, str(_action_pred_root))

try:
    import torch
    from tools.lib.yolo_person_crop import YOLOPersonCrop, create_rgbd_tensor
    from tools.lib.rgbd_model import load_model
    HAS_ACTION_MODEL = True
except ImportError as e:
    print(f"Warning: Could not import action prediction modules: {e}")
    HAS_ACTION_MODEL = False


# Default labels matching the trained model
DEFAULT_LABELS = ['jab', 'cross', 'left_hook', 'right_hook',
                  'left_uppercut', 'right_uppercut', 'block', 'idle']


class ActionPredictorNode(Node):
    """
    ROS 2 node for real-time action prediction using RGBD data.
    
    Subscribes to RGB and depth images, runs inference, and publishes
    ActionPrediction messages.
    """
    
    def __init__(self):
        super().__init__('action_predictor')
        
        # Declare parameters
        self.declare_parameter('model_config', 
            str(_action_pred_root / 'configs' / 'rgbd_boxing_anticipation.py'))
        self.declare_parameter('model_checkpoint',
            str(_ws_root.parent / 'best_acc_82.4_epoch_161.pth'))
        self.declare_parameter('yolo_model',
            str(_action_pred_root / 'checkpoints' / 'yolo26m.pt'))
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('window_size', 16)
        self.declare_parameter('crop_size', 224)
        self.declare_parameter('max_depth', 4.0)
        self.declare_parameter('enabled', True)
        self.declare_parameter('prediction_rate', 10.0)  # Hz
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        
        # Get parameters
        self.model_config = self.get_parameter('model_config').value
        self.model_checkpoint = self.get_parameter('model_checkpoint').value
        self.yolo_model = self.get_parameter('yolo_model').value
        self.device = self.get_parameter('device').value
        self.window_size = self.get_parameter('window_size').value
        self.crop_size = self.get_parameter('crop_size').value
        self.max_depth = self.get_parameter('max_depth').value
        self.enabled = self.get_parameter('enabled').value
        self.prediction_rate = self.get_parameter('prediction_rate').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        
        # State
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.model = None
        self.cropper = None
        self.labels = DEFAULT_LABELS
        
        # Frame buffers
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.frame_buffer = deque(maxlen=self.window_size)
        self.pred_history = deque(maxlen=5)  # Smoothing
        
        # Publishers
        self.pred_pub = self.create_publisher(ActionPrediction, 'action_prediction', 10)
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self._on_rgb, 5)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self._on_depth, 5)
        
        # Services
        self.enable_srv = self.create_service(
            SetBool, 'action_predictor/enable', self._handle_enable)
        
        # Prediction timer
        period = 1.0 / self.prediction_rate
        self.pred_timer = self.create_timer(period, self._run_prediction)
        
        # Initialize model in background
        self._init_done = False
        threading.Thread(target=self._init_model, daemon=True).start()
        
        self.get_logger().info('ActionPredictorNode starting...')
    
    def _init_model(self):
        """Initialize YOLO cropper and action model."""
        if not HAS_ACTION_MODEL:
            self.get_logger().error('Action prediction modules not available')
            return
        
        try:
            self.get_logger().info('Loading YOLO person cropper...')
            self.cropper = YOLOPersonCrop(self.yolo_model, self.device)
            
            self.get_logger().info('Loading action recognition model...')
            
            # Enable cuDNN benchmark
            if 'cuda' in self.device and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            
            self.model = load_model(
                config_path=self.model_config,
                checkpoint_path=self.model_checkpoint,
                device=self.device,
            )
            
            # Convert to FP16 for speed on CUDA
            if 'cuda' in self.device and hasattr(self.model, 'half'):
                self.model.half()
                self.get_logger().info('Model converted to FP16')
            
            self._init_done = True
            self.get_logger().info('Action predictor ready!')
            
        except Exception as e:
            self.get_logger().error(f'Model initialization failed: {e}')
            import traceback
            traceback.print_exc()
    
    def _on_rgb(self, msg: Image):
        """Handle RGB image callback."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self.lock:
                self.latest_rgb = img
        except Exception as e:
            self.get_logger().warn(f'RGB conversion error: {e}')
    
    def _on_depth(self, msg: Image):
        """Handle depth image callback."""
        try:
            # Depth images are typically 16UC1 (mm) or 32FC1 (m)
            if msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth = depth.astype(np.float32) * 0.001  # mm to m
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')
    
    def _handle_enable(self, request, response):
        """Handle enable/disable service."""
        self.enabled = request.data
        response.success = True
        response.message = f"Action predictor {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(response.message)
        return response
    
    def _run_prediction(self):
        """Run prediction on latest frames."""
        if not self.enabled or not self._init_done:
            return
        
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth
        
        if rgb is None or depth is None:
            return
        
        try:
            probs = self._process_frame(rgb, depth)
            
            if probs is not None:
                # Get top prediction
                idx = np.argmax(probs)
                conf = float(probs[idx])
                label = self.labels[idx]
                
                # Create and publish message
                msg = ActionPrediction()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.action_label = label
                msg.probabilities = probs.tolist()
                msg.class_labels = self.labels
                msg.confidence = conf
                
                self.pred_pub.publish(msg)
                
        except Exception as e:
            self.get_logger().warn(f'Prediction error: {e}')
    
    def _process_frame(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame and return probabilities."""
        if self.cropper is None or self.model is None:
            return None
        
        # Crop person
        rgb_crop, depth_crop, bbox = self.cropper(rgb, depth, output_size=self.crop_size)
        
        # Create RGBD tensor
        rgbd = create_rgbd_tensor(rgb_crop, depth_crop, max_depth=self.max_depth)
        rgbd = np.transpose(rgbd, (2, 0, 1))  # (4, H, W)
        
        # Add to buffer
        self.frame_buffer.append(rgbd)
        
        # Allow early predictions with at least 4 frames by padding
        min_frames_for_prediction = 4
        if len(self.frame_buffer) < min_frames_for_prediction:
            return None
        
        # Pad buffer to window_size by repeating first frame
        frames_list = list(self.frame_buffer)
        while len(frames_list) < self.window_size:
            frames_list.insert(0, frames_list[0])  # Duplicate first frame

        
        frames = np.stack(frames_list)  # (T, C, H, W) - Use padded list
        frames = torch.from_numpy(frames).float().unsqueeze(0).to(self.device)
        
        # FP16 if on CUDA and model supports it
        is_onnx = self.model.__class__.__name__ == 'ONNXWrapper'
        if 'cuda' in self.device and not is_onnx:
            frames = frames.half()
        
        with torch.inference_mode():
            probs = self.model.predict(frames, return_probs=True)
        
        probs = probs.float().cpu().numpy()[0]
        
        # Smooth predictions
        self.pred_history.append(probs)
        smoothed = np.mean(list(self.pred_history), axis=0)
        
        return smoothed


def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
