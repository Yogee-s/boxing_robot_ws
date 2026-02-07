#!/usr/bin/env python3
"""
Hybrid Detectors for Boxing Robot.

1. YOLOPoseWrapper: Wraps YOLOv8-Pose to track wrists and detect extension.
2. RosImuHandler: Handles ROS2 communication for IMU punch detection and calibration.
"""

import numpy as np
import threading
import queue
import time
import json
import os
from typing import Optional, Tuple, List

# Try imports, handle missing deps gracefully
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None

try:
    import rclpy
    from rclpy.node import Node
    from boxbunny_msgs.msg import ImuPunch, ActionPrediction
    from boxbunny_msgs.srv import CalibrateImuPunch
    from std_srvs.srv import SetBool, Trigger
    from std_msgs.msg import Float32
except ImportError:
    rclpy = None
    Node = object # Mock

# =============================================================================
# 1. Pose Wrapper
# =============================================================================
class YOLOPoseWrapper:
    """
    Wraps YOLO Pose model to specific boxing logic.
    Tracks Left/Right wrists to determine 'extension' (jab vs cross).
    """
    def __init__(self, model_path: str, device: str):
        if YOLO is None:
            raise ImportError("Ultralytics not installed.")
            
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        
        # Keypoint Indices (COCO Keypoints)
        # 0: Nose
        # 5: L-Shoulder, 6: R-Shoulder
        # 7: L-Elbow,    8: R-Elbow
        # 9: L-Wrist,   10: R-Wrist
        self.IDX_NOSE = 0
        self.IDX_L_SHOULDER = 5
        self.IDX_R_SHOULDER = 6
        self.IDX_L_WRIST = 9
        self.IDX_R_WRIST = 10
        
    def predict(self, bgr_frame: np.ndarray) -> dict:
        """
        Run pose estimation.
        Returns dict with:
          - 'left_wrist': (x, y, conf)
          - 'right_wrist': (x, y, conf)
          - 'extended_hand': 'left' | 'right' | None
          - 'keypoints': Full keypoints array (17, 3)
          - 'bbox': Bounding box [x1, y1, x2, y2] if available
        """
        results = self.model(bgr_frame, verbose=False, save=False)
        if not results or not results[0].keypoints:
            return {'extended_hand': None}

        keypoints = results[0].keypoints.data
        if keypoints is None or len(keypoints) == 0:
            return {'extended_hand': None}

        boxes = results[0].boxes.xyxy if results[0].boxes is not None else None
        h, w = bgr_frame.shape[:2]
        center_x = w * 0.5
        center_y = h * 0.5

        def _bbox_for_index(idx: int) -> Optional[np.ndarray]:
            if boxes is not None and len(boxes) > idx:
                return boxes[idx].cpu().numpy()
            kp = keypoints[idx].cpu().numpy()
            valid = kp[:, 2] > 0.1
            if not np.any(valid):
                return None
            xs = kp[valid, 0]
            ys = kp[valid, 1]
            return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

        best_idx = None
        best_score = None
        for i in range(int(keypoints.shape[0])):
            bbox_i = _bbox_for_index(i)
            if bbox_i is None:
                continue
            bx = (bbox_i[0] + bbox_i[2]) * 0.5
            by = (bbox_i[1] + bbox_i[3]) * 0.5
            dist2 = (bx - center_x) ** 2 + (by - center_y) ** 2
            area = max(0.0, (bbox_i[2] - bbox_i[0])) * max(0.0, (bbox_i[3] - bbox_i[1]))
            score = (dist2, -area)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return {'extended_hand': None}

        # Get keypoints for the person closest to image center
        # Shape: (17, 3) [x, y, conf]
        kp = keypoints[best_idx].cpu().numpy()

        # Get bounding box if available
        bbox = None
        if boxes is not None and len(boxes) > best_idx:
            bbox = boxes[best_idx].cpu().numpy()  # [x1, y1, x2, y2]
        
        l_wrist = kp[self.IDX_L_WRIST]
        r_wrist = kp[self.IDX_R_WRIST]
        nose = kp[self.IDX_NOSE]
        
        # Determine specific hand extension logic
        # Simple heuristic: Who is closer to camera? (Larger bounding box? Depth map?)
        # Or relative to nose/shoulders? 
        # For top-down/front view, we can check Y-axis (height) or just Z-depth if passed.
        
        # Here we just return the raw positions, logic can be augmented with Depth map outside.
        return {
            'left_wrist': l_wrist,
            'right_wrist': r_wrist,
            'nose': nose,
            'keypoints': kp,
            'bbox': bbox
        }

# =============================================================================
# 2. ROS IMU Handler
# =============================================================================
class RosImuHandler(threading.Thread):
    """
    Runs a background ROS node to listen for IMU punches.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.node = None  # Node is created in run()
        
        if rclpy is None:
            print("Warning: rclpy not found. Hybrid mode disabled.")
            return

        self.ready = False
        self.latest_punch = None
        self.punch_queue = queue.Queue(maxsize=10)
        self.status_msg = "Initializing ROS..."
        
        # Callbacks
        self.on_punch_callback = None
        self.calib_file = os.path.join(os.path.dirname(__file__), '../..', 'models', 'calibration.json')
        
        # Stats for Testing mode
        self.stats = {
            'total': 0,
            'jab': 0,
            'cross': 0,
            'hook': 0,
            'uppercut': 0,
            'straight': 0, # Generic straight
            'last_type': 'None',
            'last_timestamp': 0
        }
        
        # Debounce/Cooldown for Spring Oscillations
        self.last_punch_time = 0.0
        self.cooldown_duration = 0.4 # 400ms cooldown
        
        # Flags for external control
        self.simple_mode = True  # Default to color tracking (matches launch default)
        self.calibrate_height_req = False
        
    def reset_stats(self):
        """Reset punch statistics."""
        for k in self.stats:
            self.stats[k] = 0
        self.stats['last_type'] = 'None'
        
    def run(self):
        if rclpy is None: return
        
        # Define Node class inline to capture self
        class ImuNode(Node):
            def __init__(node_self):
                super().__init__('hybrid_inference_node')
                
                # Subscriber
                node_self.create_subscription(
                    ImuPunch, 
                    'imu/punch', 
                    self._on_punch_msg, 
                    10
                )
                
                # Service Client for Calibration
                self.calib_client = node_self.create_client(
                    CalibrateImuPunch, 
                    'calibrate_imu_punch'
                )

                # Action Prediction Pub
                self.action_pub = node_self.create_publisher(ActionPrediction, 'action_prediction', 10)
                # Check if publisher already exists/reuse? No, create_publisher is safe usually.
                self.height_pub = node_self.create_publisher(Float32, '/player_height', 10)
                
                # New Services
                node_self.create_service(SetBool, 'action_predictor/set_simple_mode', node_self._handle_simple_mode)
                node_self.create_service(Trigger, 'action_predictor/calibrate_height', node_self._handle_calib_height) 
                
            def log(self, msg):
                self.get_logger().info(msg)
                
            def _handle_simple_mode(self, req, res):
                # Access parent class (handler) via closure if needed, but 'self' here is ImuNode.
                # We need to set flag on Handler. 
                # Inner class can access outer instance? No, not automatically in Python unless passed.
                # Used 'node_self' for ImuNode instance. 'self' refers to RosImuHandler instance in run() scope?
                # No, run() defines ImuNode. 'self' inside ImuNode methods is ImuNode instance.
                # But 'self' in run() is RosImuHandler instance.
                # So we can close over 'self' (RosImuHandler).
                outer.simple_mode = req.data
                res.success = True
                res.message = f"Simple mode set to {req.data}"
                outer.status_msg = res.message
                return res

            def _handle_calib_height(self, req, res):
                outer.calibrate_height_req = True
                res.success = True
                res.message = "Height calibration triggered"
                return res

        # Init ROS context (check if already initialized?)
        if not rclpy.ok():
            try:
                rclpy.init()
            except Exception:
                pass 
            
        outer = self # Capture reference for inner class
        self.node = ImuNode()
        self.ready = True
        self.status_msg = "ROS Ready."
        
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        
        try:
            executor.spin()
        except Exception as e:
            print(f"ROS Executor Error: {e}")
        finally:
            try:
                executor.shutdown()
                self.node.destroy_node()
            except:
                pass
            
            # Do NOT shutdown if we didn't init, or just leave it to the process exit
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except:
                    pass

    def _on_punch_msg(self, msg):
        """Called by ROS thread when punch detected."""
        now = time.time()
        
        # Filter Oscillations (Cooldown)
        if (now - self.last_punch_time) < self.cooldown_duration:
            return
            
        self.last_punch_time = now
        self.latest_punch = msg
        try:
            # Update stats
            ptype = msg.punch_type.lower()
            self.stats['total'] += 1
            if ptype in self.stats:
                self.stats[ptype] += 1
            else:
                self.stats['straight'] += 1 # Default bucket
            
            self.stats['last_type'] = msg.punch_type
            self.stats['last_timestamp'] = time.time()
            
            self.punch_queue.put_nowait(msg)
        except queue.Full:
            pass
            
    def get_latest_punch(self):
        """Get latest punch (clears queue)."""
        try:
            return self.punch_queue.get_nowait()
        except queue.Empty:
            return None
            
    def calibrate(self, punch_type: str, duration_s: float = 3.5):
        """
        Call calibration service async.
        Returns Future object.
        """
        if not self.ready:
            return None
            
        if not self.node.calib_client.wait_for_service(timeout_sec=1.0):
            self.status_msg = "Calibration Service Unavailable!"
            return None
            
        req = CalibrateImuPunch.Request()
        req.punch_type = punch_type
        req.duration_s = float(duration_s)
        
        future = self.node.calib_client.call_async(req)
        
        # Attach callback to save result
        def done_callback(fut):
            try:
                res = fut.result()
                if res.success:
                    self.save_calibration({req.punch_type: res.threshold})
            except Exception as e:
                print(f"Calib fail: {e}")
                
        future.add_done_callback(done_callback)
        return future

    def save_calibration(self, new_data: dict):
        """Save calibration data to JSON."""
        data = {}
        if os.path.exists(self.calib_file):
            try:
                with open(self.calib_file, 'r') as f:
                    data = json.load(f)
            except: pass
            
        data.update(new_data)
        
        try:
            os.makedirs(os.path.dirname(self.calib_file), exist_ok=True)
            with open(self.calib_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved calibration to {self.calib_file}")
        except Exception as e:
            print(f"Failed to save calib: {e}")

    def publish_action(self, label: str, conf: float, probs: List[float] = None, labels: List[str] = None):
        """Publish ActionPrediction msg."""
        if not self.ready: return
        
        msg = ActionPrediction()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.action_label = label
        msg.confidence = conf
        if probs:
            msg.probabilities = probs
        if labels:
            msg.class_labels = labels
            
        self.action_pub.publish(msg)

    def publish_height(self, height: float):
        if not self.ready: return
        msg = Float32()
        msg.data = height
        self.height_pub.publish(msg)

    def load_calibration(self) -> dict:
        """Load calibration data."""
        if os.path.exists(self.calib_file):
            try:
                with open(self.calib_file, 'r') as f:
                    return json.load(f)
            except: pass
        return {}
