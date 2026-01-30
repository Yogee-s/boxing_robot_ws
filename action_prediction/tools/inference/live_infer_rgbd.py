#!/usr/bin/env python3
"""
Live RGB-D Inference GUI with Intel RealSense.

Real-time action recognition using RGB-D video model.
Optimized with FP16 (Half Precision) and CUDNN Benchmark.
"""

import argparse
import os
import sys
import time
import threading
import subprocess
import psutil
from pathlib import Path
from collections import deque
from queue import Queue, Empty
from typing import Optional, List, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

# Add project root to path
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not found", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Error: Pillow not found. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
except ImportError:
    print("Warning: cv_bridge or sensor_msgs not found. ROS publishing disabled.", file=sys.stderr)
    CvBridge = None
    Image = None

import torch
from tools.lib.yolo_person_crop import YOLOPersonCrop, create_rgbd_tensor
from tools.lib.rgbd_model import load_model
from tools.lib.hybrid_detectors import YOLOPoseWrapper, RosImuHandler

class SimpleActionDetector:
    """
    Simple glove color based detector (Red=Right/Cross, Green=Left/Jab).
    """
    def __init__(self):
        # Default HSV ranges (can be tuned via ROS or config)
        # Green (Left/Jab)
        self.green_lower = np.array([40, 80, 80])
        self.green_upper = np.array([90, 255, 255])
        
        # Red (Right/Cross) - wrap around hue
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.min_area = 5000 # px
        self.depth_threshold = 1.0 # meters
        self.cooldown = 0.5
        self.last_time = 0.0
        
    def detect(self, rgb: np.ndarray, depth: np.ndarray) -> str:
        label, _ = self.detect_debug(rgb, depth, draw=False)
        return label

    def detect_debug(self, rgb: np.ndarray, depth: np.ndarray, draw: bool = True) -> Tuple[str, Optional[np.ndarray]]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Green
        mask_g = cv2.inRange(hsv, self.green_lower, self.green_upper)
        # Red
        mask_r1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_r2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)
        
        # Cleanup noise
        kernel = np.ones((5,5), np.uint8)
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        
        area_g = np.count_nonzero(mask_g)
        area_r = np.count_nonzero(mask_r)
        
        # Determine candidate
        candidate = "IDLE"
        active_mask = None
        active_color = (0, 0, 0)
        
        # Debug Image
        vis = None
        if draw:
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        if area_g > self.min_area and area_g > area_r:
            candidate = "jab"
            active_mask = mask_g
            active_color = (0, 255, 0) # Green
        elif area_r > self.min_area and area_r > area_g:
            candidate = "cross"
            active_mask = mask_r
            active_color = (0, 0, 255) # Red
            
        # Draw Contours
        if draw:
            # Draw Green
            cnts_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_g:
                if cv2.contourArea(c) > 1000:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(vis, (x,y), (x+w,y+h), (0, 255, 0), 2)
            # Draw Red
            cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_r:
                if cv2.contourArea(c) > 1000:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(vis, (x,y), (x+w,y+h), (0, 0, 255), 2)
            
        if candidate == "IDLE":
             return "IDLE", vis

        # Check Depth
        dist = self._get_median_depth(active_mask, depth)
        
        if draw and active_mask is not None:
             # Add text
             cv2.putText(vis, f"{candidate.upper()} {dist:.2f}m" if dist else candidate.upper(), 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, active_color, 2)

        if dist is None or dist > self.depth_threshold:
             # print(f"Ignored {candidate} at {dist}m")
             return "IDLE", vis
        
        now = time.time()
        if now - self.last_time < self.cooldown:
            return "IDLE", vis
            
        self.last_time = now
        return candidate, vis

    def _get_median_depth(self, mask, depth):
        if depth is None: return None
        # Apply mask
        valid_depth = depth[mask > 0]
        if valid_depth.size == 0: return None
        # Filter 0s
        valid_depth = valid_depth[valid_depth > 0]
        if valid_depth.size < 50: return None
        return float(np.median(valid_depth))


def _load_config(path: str) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    config = {}
    for key in dir(cfg_module):
        if not key.startswith('_'):
            config[key] = getattr(cfg_module, key)
    return config


# Default labels
DEFAULT_LABELS = ['jab', 'cross', 'left_hook', 'right_hook',
                  'left_uppercut', 'right_uppercut', 'block', 'idle']


class ResourceMonitor:
    """Monitors system resources (CPU, GPU, RAM)."""
    
    def __init__(self):
        self.cpu_usage = 0.0
        self.ram_usage = 0.0
        self.gpu_util = 0.0
        self.gpu_mem = 0.0
        self.fps = 0.0
        self._last_time = time.time()
        self._frame_count = 0
        
        # Start monitoring thread
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def update_fps(self):
        """Update FPS counter."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = now
            
    def _monitor_loop(self):
        """Background loop to fetch stats."""
        while self.running:
            try:
                # CPU & RAM
                self.cpu_usage = psutil.cpu_percent()
                self.ram_usage = psutil.virtual_memory().percent
                
                # GPU (nvidia-smi)
                try:
                    cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
                    output = subprocess.check_output(cmd.split(), encoding='utf-8')
                    util, mem = output.strip().split(',')
                    self.gpu_util = float(util)
                    self.gpu_mem = float(mem)
                except Exception:
                    self.gpu_util = 0.0
                    self.gpu_mem = 0.0
                
                time.sleep(1.0)
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(2.0)
    
    def stop(self):
        self.running = False


class LiveRGBDInference:
    """
    Live RGB-D action recognition with Intel RealSense.
    """
    
    def __init__(
        self,
        root: tk.Tk,
        model_config: str,
        model_checkpoint: str,
        yolo_model: str = 'checkpoints/yolo26m.pt',
        window_size: int = 16,
        device: str = 'cuda:0',
        causal: bool = True,
        labels: Optional[List[str]] = None,
        fps: int = 30,
        rgb_res: str = '640x480',
        depth_res: str = '640x480',
        detect_interval: int = 1,
    ):
        self.root = root
        self.root.title(f"RGB-D Boxing Action Recognition ({fps} FPS)")
        self.device = device
        self.window_size = window_size
        self.mode = 'hybrid' # 'anticipation' or 'hybrid'
        self.causal = causal
        self.labels = labels or DEFAULT_LABELS
        self.fps = fps
        self.detect_interval = detect_interval
        self.pose_conf_thresh = 0.5
        self.rgb_res = self._parse_res(rgb_res)
        self.depth_res = self._parse_res(depth_res)
        
        # Camera Intrinsics (Approx for D435i@640x480)
        # fx ~ 610, fy ~ 610, cx ~ 320, cy ~ 240
        self.intrinsics = {'fx': 610.0, 'fy': 610.0, 'cx': 320.0, 'cy': 240.0}
        
        # Height Publishing
        self.height_pub = None
        self.glove_debug_pub = None 
        
        # Theme
        self.c_bg_main = '#0d1117'
        self.c_bg_panel = '#161b22'
        self.c_accent = '#ff6b35'
        self.c_text_main = '#e6edf3'
        self.c_text_dim = '#8b949e'
        self.c_border = '#30363d'
        
        self.font_family = 'Segoe UI' if os.name == 'nt' else 'DejaVu Sans'
        self.font_main = (self.font_family, 11)
        self.font_large = (self.font_family, 24, 'bold')
        self.font_huge = (self.font_family, 48, 'bold')
        
        # State
        self.running = True
        self.pipeline = None
        self.model = None
        self.model = None
        self.cropper = None
        self.model = None
        self.cropper = None
        self.monitor = ResourceMonitor()
        
        # Hybrid Components
        self.pose_wrapper = None
        self.imu_handler = None
        self.simple_detector = SimpleActionDetector()
        self.bridge = CvBridge() if CvBridge else None
        
        # Buffers
        self.frame_buffer = deque(maxlen=window_size)
        self.pred_history = deque(maxlen=5)
        self.pipeline_lock = threading.Lock()
        
        # Threading Queues
        # queue_size=1 ensures we always process the LATEST frame and drop old ones
        self.input_queue = Queue(maxsize=1) 
        self.result_queue = Queue(maxsize=1)
        
        # Current prediction state
        self.current_probs = None
        self.current_bbox = None
        self.current_rgb_crop = None
        self.current_depth_crop = None
        
        # Store config paths
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.yolo_model = yolo_model

        # Load config for data/model params
        cfg = _load_config(self.model_config) if self.model_config else {}
        data_cfg = cfg.get('data', {})
        self.crop_size = data_cfg.get('crop_size', 224)
        self.max_depth = data_cfg.get('max_depth', 4.0)
        self.use_delta = data_cfg.get('use_delta', False)
        if labels is None:
            label_map = cfg.get('label_map', None)
            if isinstance(label_map, dict) and label_map:
                self.labels = [k for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]
        
        # Setup GUI
        self._setup_gui()
        
        # Initialize in background
        self.root.after(100, self._init_async)
    
    def _worker_loop(self):
        """Background inference loop."""
        print("Worker thread started")
        while self.running:
            try:
                # Get latest frame (timeout to allow checking self.running)
                try:
                    rgb, depth = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Run Inference (Heavy!)
                # Run Inference (Heavy!)
                # Check for Simple Mode
                use_simple = self.imu_handler and self.imu_handler.simple_mode
                
                if use_simple:
                    # Simple Color Mode with Depth Trigger
                    label, debug_img = self.simple_detector.detect_debug(rgb, depth, draw=True)
                    probs = np.zeros(len(self.labels), dtype=np.float32)
                    bbox = None
                    rgb_crop, depth_crop = None, None
                    
                    if label != "IDLE":
                         # Find index
                         if label in self.labels:
                             probs[self.labels.index(label)] = 1.0
                         # Publish
                         self.imu_handler.publish_action(label, 1.0, probs.tolist(), self.labels)
                    else:
                         # Idle
                         if 'idle' in self.labels:
                             probs[self.labels.index('idle')] = 1.0
                    
                    # Publish Debug Image
                    if debug_img is not None and self.glove_debug_pub:
                        try:
                            msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                            self.glove_debug_pub.publish(msg)
                        except Exception:
                            pass

                else:
                    # Use AI Model
                    probs, rgb_crop, depth_crop, bbox = self._process_frame(rgb, depth)
                    
                    # Publish result if valid
                    if probs is not None and self.imu_handler:
                         idx = np.argmax(probs)
                         self.imu_handler.publish_action(self.labels[idx], float(probs[idx]), probs.tolist(), self.labels)
                    
                    # Publish Debug Image (AI Mode)
                    if self.glove_debug_pub:
                        try:
                            # Convert RGB to BGR for ROS
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            msg = self.bridge.cv2_to_imgmsg(bgr, "bgr8")
                            self.glove_debug_pub.publish(msg)
                        except Exception:
                            pass
                
                
                # Put result (non-blocking, drop old result if main thread is slow)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put((probs, rgb_crop, depth_crop, bbox))
                
            except Exception as e:
                print(f"Worker Error: {e}")
                time.sleep(0.1)
                
    def _poll_predictions(self):
        """Poll for new predictions from worker."""
        if not self.running: return
        
        try:
            # Check if new result is available
            while not self.result_queue.empty():
                # Get latest (drain queue if multiple accumulated, though maxsize=1 prevents that)
                probs, rgb_crop, depth_crop, bbox = self.result_queue.get_nowait()
                
                # Update state
                self.current_probs = probs
                self.current_rgb_crop = rgb_crop
                self.current_depth_crop = depth_crop
                self.current_bbox = bbox
                
                # Update UI elements
                self._update_prediction_ui(probs)
                self._update_crop_ui(rgb_crop, depth_crop)
                
        except Empty:
            pass
        except Exception as e:
            print(f"Poll Error: {e}")
            
        self.root.after(30, self._poll_predictions) # Check ~30 times a sec
    
    def _setup_gui(self):
        """Setup the GUI layout."""
        self.root.configure(bg=self.c_bg_main)
        
        # Style Configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.c_bg_main)
        style.configure('Panel.TFrame', background=self.c_bg_panel, borderwidth=1, relief='flat')
        style.configure('TLabel', background=self.c_bg_main, foreground=self.c_text_main, font=self.font_main)
        
        # Progress Bar Style
        style.configure('Horizontal.TProgressbar',
                        background=self.c_accent,
                        troughcolor=self.c_bg_panel,
                        bordercolor=self.c_border,
                        lightcolor=self.c_bg_panel,
                        darkcolor=self.c_bg_panel)
        
        # Header
        header = tk.Frame(self.root, bg=self.c_bg_panel, height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        
        tk.Label(header, text="[LIVE INFERENCE]", font=self.font_main, bg=self.c_bg_panel, fg=self.c_accent).pack(side=tk.LEFT, padx=(20,8), pady=10)
        tk.Label(header, text="RGB-D Boxing Recognition", font=(self.font_family, 14, 'bold'), bg=self.c_bg_panel, fg=self.c_text_main).pack(side=tk.LEFT, pady=10)
        
        # Controls (right side)
        controls = tk.Frame(header, bg=self.c_bg_panel)
        controls.pack(side=tk.RIGHT, padx=12, pady=6)

        self.fps_var = tk.StringVar(value=str(self.fps))
        self.rgb_res_var = tk.StringVar(value=f"{self.rgb_res[0]}x{self.rgb_res[1]}")
        self.depth_res_var = tk.StringVar(value=f"{self.depth_res[0]}x{self.depth_res[1]}")

        fps_box = ttk.Combobox(controls, textvariable=self.fps_var, width=5, state='readonly')
        fps_box['values'] = ['15', '30', '60']
        fps_box.pack(side=tk.RIGHT, padx=4)

        tk.Label(controls, text="FPS", bg=self.c_bg_panel, fg=self.c_text_dim, font=self.font_main).pack(side=tk.RIGHT, padx=2)

        depth_box = ttk.Combobox(controls, textvariable=self.depth_res_var, width=9, state='readonly')
        depth_box['values'] = ['640x480', '848x480', '1280x720', '424x240']
        depth_box.pack(side=tk.RIGHT, padx=4)
        tk.Label(controls, text="Depth", bg=self.c_bg_panel, fg=self.c_text_dim, font=self.font_main).pack(side=tk.RIGHT, padx=2)

        rgb_box = ttk.Combobox(controls, textvariable=self.rgb_res_var, width=9, state='readonly')
        rgb_box['values'] = ['640x480', '848x480', '960x540', '1280x720']
        rgb_box.pack(side=tk.RIGHT, padx=4)
        tk.Label(controls, text="RGB", bg=self.c_bg_panel, fg=self.c_text_dim, font=self.font_main).pack(side=tk.RIGHT, padx=2)

        apply_btn = tk.Button(
            controls,
            text="Apply",
            command=self._apply_stream_settings,
            bg=self.c_bg_panel,
            fg=self.c_text_main,
            relief='flat',
            padx=8,
            pady=2,
        )
        apply_btn.pack(side=tk.RIGHT, padx=6)

        # Status Bar
        self.status_label = tk.Label(header, text="Initializing...", font=self.font_main, bg=self.c_bg_panel, fg=self.c_text_dim)
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)
        

        
        # Main Container
        main_container = tk.Frame(self.root, bg=self.c_bg_main)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left: Video (960x720)
        video_frame = tk.Frame(main_container, bg=self.c_bg_panel, bd=1)
        video_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.video_label = tk.Label(video_frame, bg='black', width=960, height=720)
        self.video_label.pack(padx=2, pady=2)
        
        # Right: Info Panel
        info_panel = tk.Frame(main_container, bg=self.c_bg_main)
        info_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 1. Prediction
        pred_group = tk.Frame(info_panel, bg=self.c_bg_main)
        pred_group.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(pred_group, text="DETECTED ACTION", font=(self.font_family, 10, 'bold'), bg=self.c_bg_main, fg=self.c_text_dim).pack(anchor='w')
        self.pred_label = tk.Label(pred_group, text="WAITING", font=self.font_huge, bg=self.c_bg_main, fg=self.c_text_main)
        self.pred_label.pack(anchor='w', pady=(5, 0))
        self.conf_label = tk.Label(pred_group, text="0%", font=self.font_large, bg=self.c_bg_main, fg=self.c_accent)
        self.conf_label.pack(anchor='w')
        
        # 2. Stats (Moved here)
        self.lbl_stats = tk.Label(
            info_panel, 
            text="FPS: 0  |  GPU: 0%", 
            font=(self.font_family, 10), 
            bg=self.c_bg_main, 
            fg=self.c_text_dim,
            justify=tk.LEFT
        )
        self.lbl_stats.pack(anchor='w', pady=(0, 10))
        
        # 3. Controls
        self.var_show_crops = tk.BooleanVar(value=True)
        self.var_show_bbox = tk.BooleanVar(value=False)
        
        control_group = tk.Frame(info_panel, bg=self.c_bg_panel, padx=10, pady=10)
        control_group.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(control_group, text="Optimization Controls", font=(self.font_family, 9, 'bold'), bg=self.c_bg_panel, fg=self.c_text_dim).pack(anchor='w', pady=(0,5))
        
        chk_crops = tk.Checkbutton(control_group, text="Show Crops", variable=self.var_show_crops, 
                                   bg=self.c_bg_panel, fg=self.c_text_main, selectcolor=self.c_bg_panel, activebackground=self.c_bg_panel)
        chk_crops.pack(anchor='w')
        
        chk_bbox = tk.Checkbutton(control_group, text="Show Bounding Box", variable=self.var_show_bbox, 
                                  bg=self.c_bg_panel, fg=self.c_text_main, selectcolor=self.c_bg_panel, activebackground=self.c_bg_panel)
        chk_bbox.pack(anchor='w', pady=(0, 10))

        # Tuning Sliders
        tk.Label(control_group, text="Vision Sensitivity", font=(self.font_family, 9, 'bold'), bg=self.c_bg_panel, fg=self.c_text_dim).pack(anchor='w', pady=(5,5))
        
        # YOLO Confidence
        tk.Label(control_group, text="YOLO Conf", bg=self.c_bg_panel, fg=self.c_text_dim, font=(self.font_family, 8)).pack(anchor='w')
        self.scale_yolo = tk.Scale(control_group, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                   bg=self.c_bg_panel, fg=self.c_text_main, highlightthickness=0,
                                   command=self._update_vision_params)
        self.scale_yolo.set(0.25) # Default
        self.scale_yolo.pack(fill=tk.X, pady=(0, 5))

        # Pose Confidence (Reaction Sensitivity)
        tk.Label(control_group, text="Pose Thresh (Reaction)", bg=self.c_bg_panel, fg=self.c_text_dim, font=(self.font_family, 8)).pack(anchor='w')
        self.scale_pose = tk.Scale(control_group, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                   bg=self.c_bg_panel, fg=self.c_text_main, highlightthickness=0,
                                   command=self._update_vision_params)
        self.scale_pose.set(0.5)
        self.scale_pose.pack(fill=tk.X, pady=(0, 5))

        # Mode Selector
        self.btn_mode = tk.Button(control_group, text="MODE: Hybrid (IMU+Pose)", command=self._toggle_mode,
                                   bg='#2B3240', fg=self.c_text_main, font=(self.font_family, 9, 'bold'))
        self.btn_mode.pack(fill=tk.X, pady=(10, 5))

        # Calibration Button
        self.btn_calib = tk.Button(control_group, text="Calibrate IMU", command=self._start_calibration,
                                   bg='#2B3240', fg=self.c_text_main, font=(self.font_family, 9))
        self.btn_calib.pack(fill=tk.X)
        
        # 4. Model Inputs (Crops)
        crop_group = tk.Frame(info_panel, bg=self.c_bg_panel, padx=10, pady=10)
        crop_group.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(crop_group, text="Model Inputs (224x224)", font=(self.font_family, 9, 'bold'), bg=self.c_bg_panel, fg=self.c_text_dim).pack(anchor='w', pady=(0,5))
        
        self.crop_imgs_frame = tk.Frame(crop_group, bg=self.c_bg_panel)
        self.crop_imgs_frame.pack(anchor='w')
        
        # RGB Crop
        v_rgb = tk.Frame(self.crop_imgs_frame, bg=self.c_bg_panel)
        v_rgb.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(v_rgb, text="RGB Crop", font=(self.font_family, 8), bg=self.c_bg_panel, fg=self.c_text_dim).pack()
        self.lbl_crop_rgb = tk.Label(v_rgb, bg='black', width=224, height=224)
        self.lbl_crop_rgb.pack()
        
        # Depth Crop
        v_depth = tk.Frame(self.crop_imgs_frame, bg=self.c_bg_panel)
        v_depth.pack(side=tk.LEFT)
        tk.Label(v_depth, text="Depth Input", font=(self.font_family, 8), bg=self.c_bg_panel, fg=self.c_text_dim).pack()
        self.lbl_crop_depth = tk.Label(v_depth, bg='black', width=224, height=224)
        self.lbl_crop_depth.pack()
        
        # 5. Probabilities
        prob_container = tk.Frame(info_panel, bg=self.c_bg_panel, padx=10, pady=10)
        prob_container.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(prob_container, text="Class Probabilities", font=(self.font_family, 10, 'bold'), bg=self.c_bg_panel, fg=self.c_text_dim).pack(anchor='w', pady=(0,10))
        
        self.prob_bars = {}
        for i, label in enumerate(self.labels):
            frame = tk.Frame(prob_container, bg=self.c_bg_panel)
            frame.pack(fill=tk.X, pady=2)
            
            lbl = tk.Label(frame, text=label.upper(), width=15, anchor='w', bg=self.c_bg_panel, fg=self.c_text_main, font=(self.font_family, 9))
            lbl.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(frame, length=150, mode='determinate', style='Horizontal.TProgressbar')
            bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            val = tk.Label(frame, text="0%", width=4, anchor='e', bg=self.c_bg_panel, fg=self.c_text_dim, font=(self.font_family, 9))
            val.pack(side=tk.LEFT)
            
            self.prob_bars[label] = (bar, val)
            
        # 6. Advanced Controls (Height)
        adv_group = tk.Frame(info_panel, bg=self.c_bg_panel, padx=10, pady=10)
        adv_group.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(adv_group, text="Advanced", font=(self.font_family, 9, 'bold'), bg=self.c_bg_panel, fg=self.c_text_dim).pack(anchor='w')
        
        self.lbl_height = tk.Label(adv_group, text="Height: -- m", font=(self.font_family, 10), bg=self.c_bg_panel, fg=self.c_text_main)
        self.lbl_height.pack(anchor='w', pady=2)
        
        btn_send_h = tk.Button(adv_group, text="Send Height", command=self._send_height,
                             bg='#2B3240', fg=self.c_text_main, font=(self.font_family, 9))
        btn_send_h.pack(fill=tk.X)
    
    def _init_async(self):
        """Initialize camera and models in background."""
        def init():
            try:
                # 1. Start ROS/IMU Handler
                self._update_status("Starting ROS/IMU...")
                self.imu_handler = RosImuHandler()
                self.imu_handler.start()
                
                # Setup Height Publisher (hacky access to node)
                if self.imu_handler.node:
                    from std_msgs.msg import Float32
                    self.height_pub = self.imu_handler.node.create_publisher(Float32, '/player_height', 10)
                    if Image:
                        self.glove_debug_pub = self.imu_handler.node.create_publisher(Image, 'glove_debug_image', 10)
                
                # 2. Load Models
                # Only load Pose Wrapper by default for Hybrid Mode
                self._update_status("Loading Pose Model...")
                self.pose_wrapper = YOLOPoseWrapper(
                    '../models/checkpoints/yolo26n-pose.pt',
                    self.device
                )

                if self.mode == 'anticipation':
                    self._update_status("Loading Action Models (Slow)...")
                    # Load Object Detection YOLO
                    self.cropper = YOLOPersonCrop(
                        self.yolo_model, 
                        self.device,
                        detect_interval=self.detect_interval
                    )
                    
                    # Load Action Model
                    # Enable cuDNN benchmark for fixed input size optimization
                    if 'cuda' in self.device:
                        torch.backends.cudnn.benchmark = True
                    
                    self.model = load_model(
                        config_path=self.model_config,
                        checkpoint_path=self.model_checkpoint,
                        device=self.device,
                    )
                    
                    # Convert to Half Precision (FP16) if on CUDA
                    if 'cuda' in self.device:
                        if hasattr(self.model, 'half'):
                            self.model.half()
                            print("Model converted to FP16")
                
                self._update_status("Initializing camera...")
                self._init_realsense()
                
                self._update_status("Ready - Start throwing punches!")
                
                # Start video loop
                self.root.after(10, self._video_loop)
                # Start stats loop
                self.root.after(500, self._update_stats_display)
                # Start prediction poller
                self.root.after(30, self._poll_predictions)
                
                # Start worker thread
                threading.Thread(target=self._worker_loop, daemon=True).start()
                
            except Exception as e:
                self._update_status(f"Error: {e}")
                print(f"Initialization error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=init, daemon=True).start()
    
    def _init_realsense(self):
        """Initialize RealSense camera."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        rgb_w, rgb_h = self.rgb_res
        depth_w, depth_h = self.depth_res
        config.enable_stream(rs.stream.color, rgb_w, rgb_h, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, self.fps)
        
        # Start pipeline
        try:
            profile = self.pipeline.start(config)
        except Exception as exc:
            self.pipeline = None
            raise RuntimeError(f"No device connected: {exc}") from exc
        
        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
    
    def _update_status(self, text: str):
        """Update status label (thread-safe)."""
        def update():
            self.status_label.config(text=text)
        self.root.after(0, update)
    
    def _video_loop(self):
        """Main video processing loop."""
        if not self.running or self.pipeline is None:
            return
        
        try:
            # Wait for frames
            with self.pipeline_lock:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                self.root.after(10, self._video_loop)
                return
            
            # Convert to numpy
            rgb = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth = depth * self.depth_scale  # Convert to meters
            
            # Push to inference queue (non-blocking)
            if not self.input_queue.full():
                self.input_queue.put((rgb, depth))
            
            # --- Hybrid Logic: Check IMU ---
            if self.mode == 'hybrid' and self.imu_handler:
                punch_msg = self.imu_handler.get_latest_punch()
                if punch_msg:
                    self._handle_hybrid_punch(punch_msg, rgb, depth)

            # Height Calc (Update GUI + Publish to ROS)
            if self.current_bbox is not None:
                h_val = self._calc_height(depth, self.current_bbox)
                self.lbl_height.config(text=f"Height: {h_val:.2f} m")
                # Publish to ROS topic for other nodes
                if self.height_pub and h_val > 0.1:
                    from std_msgs.msg import Float32
                    msg = Float32()
                    msg.data = h_val
                    self.height_pub.publish(msg)

            # Check Height Trigger (Service)
            if self.imu_handler and self.imu_handler.calibrate_height_req:
                self.imu_handler.calibrate_height_req = False
                # Perform instant calculation (or could average)
                # We need a bbox to calculate height. If model mode, we have current_bbox.
                # If simple mode, we might not have bbox.
                # Use current_bbox if available, else try to find one or warn.
                if self.current_bbox is not None:
                     h_val = self._calc_height(depth, self.current_bbox)
                     self.imu_handler.publish_height(h_val)
                     self._update_status(f"Calibrated Height: {h_val:.2f} m")
                else:
                     self._update_status("Height Calib Failed: No person detected")

            # Update display with LATEST AVAILABLE predictions
            self._update_display(
                rgb, 
                self.current_bbox
            )
            self.monitor.update_fps()
            
        except Exception as e:
            print(f"Frame error: {e}")
        
        # Schedule next iteration (aim for high FPS)
        self.root.after(10, self._video_loop)

    def _send_height(self):
        """Publish current height to ROS."""
        txt = self.lbl_height.cget("text")
        try:
            val = float(txt.split(":")[1].replace("m", "").strip())
            if self.height_pub:
                from std_msgs.msg import Float32
                msg = Float32()
                msg.data = val
                self.height_pub.publish(msg)
                self._update_status(f"Sent Height: {val:.2f}m")
            else:
                self._update_status("Error: Publisher not ready")
        except:
            self._update_status("Error: No valid height")

    def _update_vision_params(self, _=None):
        """Update vision parameters from GUI sliders."""
        if hasattr(self, 'scale_yolo') and hasattr(self, 'cropper'):
            val = self.scale_yolo.get()
            # Attempt to set on wrapper (assuming it stores it)
            if hasattr(self.cropper, 'conf'):
                self.cropper.conf = val
        
        if hasattr(self, 'scale_pose'):
            self.pose_conf_thresh = self.scale_pose.get()
            
        print(f"Updated Params - YOLO Conf: {self.cropper.conf}, Pose Thresh: {self.pose_conf_thresh}")

    def _toggle_mode(self):
        """Switch between Anticipation and Hybrid modes."""
        if self.mode == 'hybrid':
            self.mode = 'anticipation'
            self.btn_mode.config(text="MODE: Anticipation (AI)")
            self._update_status("Switched to Full AI Model")
        else:
            self.mode = 'hybrid'
            self.btn_mode.config(text="MODE: Hybrid (IMU+Pose)")
            self._update_status("Switched to Hybrid Mode")

    def _start_calibration(self):
        """Trigger IMU calibration wizard."""
        if not self.imu_handler or not self.imu_handler.ready:
            self._update_status("Error: ROS IMU inactive")
            return
            
        # Dialog
        win = tk.Toplevel(self.root)
        win.title("IMU Setup: Calibration & Testing")
        win.geometry("400x350")
        win.configure(bg=self.c_bg_panel)
        
        # Tabs
        notebook = ttk.Notebook(win)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Calibrate
        f_calib = tk.Frame(notebook, bg=self.c_bg_panel)
        notebook.add(f_calib, text='Calibrate')
        
        lbl = tk.Label(f_calib, text="Hit the bag 3 times continuously\nwhen you press Start.", 
                       bg=self.c_bg_panel, fg=self.c_text_main, pady=10)
        lbl.pack()
        
        def run_calib(astype):
            lbl.config(text=f"CALIBRATING {astype.upper()}... PUNCH NOW!", fg=self.c_accent)
            future = self.imu_handler.calibrate(astype, 3.5)
            if not future:
                lbl.config(text="Service Call Failed")
                return
            
            # Close wait
            self.root.after(4000, lambda: lbl.config(text="Done! Saved.", fg=self.c_text_main))

        btn_straight = tk.Button(f_calib, text="Calibrate STRAIGHT (3.5s)", 
                               command=lambda: run_calib('straight'), bg=self.c_bg_main, fg='white')
        btn_straight.pack(pady=5, fill=tk.X, padx=20)
        
        btn_hook = tk.Button(f_calib, text="Calibrate HOOK (3.5s)", 
                               command=lambda: run_calib('hook'), bg=self.c_bg_main, fg='white')
        btn_hook.pack(pady=5, fill=tk.X, padx=20)

        btn_upper = tk.Button(f_calib, text="Calibrate UPPERCUT (3.5s)", 
                               command=lambda: run_calib('uppercut'), bg=self.c_bg_main, fg='white')
        btn_upper.pack(pady=5, fill=tk.X, padx=20)
        
        # Tab 2: Test
        f_test = tk.Frame(notebook, bg=self.c_bg_panel)
        notebook.add(f_test, text='Test Mode')
        
        self.imu_handler.reset_stats()
        
        lbl_stats = tk.Label(f_test, text="Punches: 0\nLast: None", font=("Arial", 14),
                             bg=self.c_bg_panel, fg=self.c_text_main, pady=20)
        lbl_stats.pack()
        
        def update_test():
            if not win.winfo_exists(): return
            s = self.imu_handler.stats
            txt = f"Total: {s['total']}\nLast: {s['last_type'].upper()}\n\n"
            txt += f"Jab/Cross: {s.get('jab', 0) + s.get('cross', 0) + s.get('straight', 0)}\n"
            txt += f"Hook: {s.get('hook', 0)}\nUppercut: {s.get('uppercut', 0)}"
            lbl_stats.config(text=txt)
            win.after(100, update_test)
            
        update_test()

    def _calc_height(self, depth, bbox):
        """Estimate height from Depth + BBox."""
        if bbox is None: return 0.0
        x1, y1, x2, y2 = bbox
        
        # Get person centroid depth
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        h, w = depth.shape
        cy = max(0, min(h-1, cy))
        cx = max(0, min(w-1, cx))
        
        d_val = depth[cy, cx] * self.depth_scale # Meters
        if d_val == 0: return 0.0
        
        # Assume Standing: Top of bbox is head.
        # Height pixels = (y2 - y1)  <-- This is just crop height, might be partial
        # Better: Assume camera is at specific height (e.g. 1.0m) or assume full body visible?
        # User asked: "measure height... maybe finding a bounding box"
        # Formula: H_real = (H_pix * Depth) / fy
        
        h_pix = y2 - y1
        real_h = (h_pix * d_val) / self.intrinsics['fy']
        
        # Correction? If only upper body, this is wrong.
        # Assume full body 3m away fills 400px?
        # Let's provide raw calc first.
        return real_h

    def _apply_stream_settings(self):
        try:
            fps = int(self.fps_var.get())
            rgb_w, rgb_h = self._parse_res(self.rgb_res_var.get())
            depth_w, depth_h = self._parse_res(self.depth_res_var.get())
        except Exception:
            self._update_status("Invalid stream settings")
            return

        self.fps = fps
        self.rgb_res = (rgb_w, rgb_h)
        self.depth_res = (depth_w, depth_h)
        self.root.title(f"RGB-D Boxing Action Recognition ({fps} FPS)")

        with self.pipeline_lock:
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self._init_realsense()
            # Clear buffers to avoid mixing old frames
            self.frame_buffer.clear()
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except Empty:
                    break
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    break
        self._update_status("Stream settings applied")

    @staticmethod
    def _parse_res(text: str):
        parts = text.lower().split('x')
        return int(parts[0].strip()), int(parts[1].strip())
    
    def _process_frame(self, rgb: np.ndarray, depth: np.ndarray):
        """Process a single frame. Returns (probs, rgb_crop, depth_crop)."""
        if self.cropper is None or self.model is None:
            return None, None, None
        
        # Crop person (returns crops + bbox)
        rgb_crop, depth_crop, bbox = self.cropper(rgb, depth, output_size=self.crop_size)
        
        # Create RGBD tensor
        rgbd = create_rgbd_tensor(rgb_crop, depth_crop, max_depth=self.max_depth)
        rgbd = np.transpose(rgbd, (2, 0, 1))  # (4, H, W)
        
        # Add to buffer
        self.frame_buffer.append(rgbd)
        
        # Run inference if buffer is full
        probs = None
        if len(self.frame_buffer) >= self.window_size:
            frames = np.stack(list(self.frame_buffer))  # (T, C, H, W)
            if self.use_delta:
                delta = np.diff(frames, axis=0, prepend=frames[:1])
                frames = np.concatenate([frames, delta], axis=1)  # (T, 8, H, W)
            frames = torch.from_numpy(frames).float().unsqueeze(0).to(self.device)
            
            # Use Half Precision (FP16) for speed if on CUDA AND supported by model
            # ONNX models are typically exported as FP32, so we skip conversion for them
            is_onnx = self.model.__class__.__name__ == 'ONNXWrapper'
            if 'cuda' in self.device and not is_onnx:
                frames = frames.half()
            
            # Use inference_mode for slightly lower overhead than no_grad
            with torch.inference_mode():
                probs = self.model.predict(frames, return_probs=True)
            
            probs = probs.float().cpu().numpy()[0]
            
            # Smooth predictions
            self.pred_history.append(probs)
            smoothed = np.mean(list(self.pred_history), axis=0)
            probs = smoothed

        # Return bbox from the original crop call (already stored above)
        return probs, rgb_crop, depth_crop, bbox
    
    def _handle_hybrid_punch(self, punch_msg, rgb, depth):
        """Handle punch detected by IMU in Hybrid Mode."""
        # Get Pose
        pose = self.pose_wrapper.predict(rgb)
        
        # Determine Hand: Use Depth Z of wrist
        # Note: YOLO Pose returns (x, y, conf). We map (x, y) to Depth map (Z).
        l_wrist = pose['left_wrist'] # [x, y, conf]
        r_wrist = pose['right_wrist']
        
        hand_label = "UNKNOWN"
        
        
        if l_wrist[2] > self.pose_conf_thresh and r_wrist[2] > self.pose_conf_thresh:
            # Map coordinates
            lx, ly = int(l_wrist[0]), int(l_wrist[1])
            rx, ry = int(r_wrist[0]), int(r_wrist[1])
            
            # Safety clamp
            h, w = depth.shape
            lx, ly = min(w-1, max(0, lx)), min(h-1, max(0, ly))
            rx, ry = min(w-1, max(0, rx)), min(h-1, max(0, ry))
            
            l_z = depth[ly, lx] * self.depth_scale
            r_z = depth[ry, rx] * self.depth_scale
            
            # Closer hand is punching?
            # Or assume standard stance?
            # User Rule: Left=Green, Right=Red.
            # Logic: If IMU says "Straight", look for extended hand.
            
            if l_z < r_z - 0.1: # Left is 10cm closer
                hand_label = "JAB"
            elif r_z < l_z - 0.1:
                hand_label = "CROSS"
            else:
                # Fallback to IMU type if depths are similar
                hand_label = punch_msg.punch_type.upper()
        else:
            hand_label = punch_msg.punch_type.upper()
        
        # Display Result directly
        self.pred_label.config(text=f"{hand_label}!", fg='#4DFF88' if hand_label=='JAB' else '#FF6B6B')
        self.conf_label.config(text="IMU TRIGGER")
        
        # Flash effect?
        self.root.after(1000, lambda: self.pred_label.config(text="READY", fg=self.c_text_main))

    
    def _update_display(self, rgb: np.ndarray, bbox=None):
        """Update Main Video Display (Fastest Loop)."""
        # Main Video Logic
        display_h, display_w = 720, 960
        
        # Draw Bounding Box if enabled
        if bbox is not None and self.var_show_bbox.get():
             # Bbox is [x1, y1, x2, y2] in original 640x480 coords
             # Need to scale to display_w(960) x display_h(720)
             scale_x = display_w / 640.0
             scale_y = display_h / 480.0
             
             x1, y1, x2, y2 = bbox.astype(int)
             x1 = int(x1 * scale_x)
             y1 = int(y1 * scale_y)
             x2 = int(x2 * scale_x)
             y2 = int(y2 * scale_y)
             
             # Draw rectangle on copy
             # Optimization: If resize is bottleneck, draw on small then resize? 
             # But drawing on small is less precise. Resize is optimized in cv2.
             # Actually, drawing on large is fast enough.
             
             rgb_vis = rgb.copy()
             display_rgb = cv2.resize(rgb_vis, (display_w, display_h))
             cv2.rectangle(display_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
             
        else:
             display_rgb = cv2.resize(rgb, (display_w, display_h))
        
        # Convert to PhotoImage (This is the main cost, but necessary)
        img = Image.fromarray(display_rgb)
        photo = ImageTk.PhotoImage(img)
        
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def _update_crop_ui(self, rgb_crop, depth_crop):
        """Update Crop UI (Medium Speed)."""
        if not self.var_show_crops.get():
            # If toggled off, clear and return
            self.lbl_crop_rgb.config(image='')
            self.lbl_crop_depth.config(image='')
            return

        if rgb_crop is not None:
             img_c = Image.fromarray(rgb_crop)
             ph_c = ImageTk.PhotoImage(img_c)
             self.lbl_crop_rgb.config(image=ph_c)
             self.lbl_crop_rgb.image = ph_c
             
        if depth_crop is not None:
             # Normalize depth for display
             d_vis = depth_crop.copy()
             valid = d_vis > 0
             if valid.any():
                 d_vis[valid] = (d_vis[valid] / 4.0) * 255 # max depth 4m
             d_vis = np.clip(d_vis, 0, 255).astype(np.uint8)
             d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
             d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
             
             img_d = Image.fromarray(d_color)
             ph_d = ImageTk.PhotoImage(img_d)
             self.lbl_crop_depth.config(image=ph_d)
             self.lbl_crop_depth.image = ph_d

    def _update_prediction_ui(self, probs):
        """Update Probability UI (Slow/Poll Speed)."""
        if probs is None: return
        
        idx = np.argmax(probs)
        conf = probs[idx]
        label = self.labels[idx]
        
        # Update main prediction (Only if changed to reduce flicker? Tkinter handles it ok)
        self.pred_label.config(text=label.upper())
        self.conf_label.config(text=f"{conf*100:.0f}%")
        
        # Update probability bars
        for i, lbl in enumerate(self.labels):
            if lbl in self.prob_bars:
                bar, val_label = self.prob_bars[lbl]
                prob = probs[i] * 100
                bar['value'] = prob
                val_label.config(text=f"{prob:.0f}%")
    
    def _update_stats_display(self):
        """Update resource stats in GUI."""
        if not self.running: return
        
        m = self.monitor
        m = self.monitor
        # Vertical format for side panel
        text = f"FPS: {m.fps:.1f}\nCPU: {m.cpu_usage:.0f}% / RAM: {m.ram_usage:.0f}%\nGPU: {m.gpu_util:.0f}% / VRAM: {m.gpu_mem:.0f}MB"
        self.lbl_stats.config(text=text)
        
        # Color code GPU usage
        if m.gpu_util > 90:
            self.lbl_stats.config(fg='#ff6b35') # Alert Orange
        else:
            self.lbl_stats.config(fg=self.c_text_dim)
            
        self.root.after(500, self._update_stats_display)

    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.monitor.stop()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass


class HeadlessInference:
    """Headless version of LiveRGBDInference for ROS integration."""
    def __init__(
        self,
        model_config,
        model_checkpoint,
        yolo_model,
        window_size=16,
        device='cuda:0',
        causal=True,
        fps=30,
        rgb_res='640x480',
        depth_res='640x480',
        detect_interval=5,
        enable_action_model=False,
        mode='pose'
    ):
        self.device = device
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.window_size = window_size
        self.use_delta = False # Config loaded later usually
        self.labels = DEFAULT_LABELS
        self.running = True
        
        # Initialize ROS / IMU Handler
        print("Initializing ROS handler...")
        self.imu_handler = RosImuHandler()
        self.imu_handler.start()
        while not self.imu_handler.ready:
            time.sleep(0.1)
        self.node = self.imu_handler.node
        
        # Load Config
        cfg = _load_config(model_config)
        data_cfg = cfg.get('data', {})
        self.crop_size = data_cfg.get('crop_size', 224)
        self.max_depth = data_cfg.get('max_depth', 4.0)
        self.use_delta = data_cfg.get('use_delta', False)
        
        # Buffers
        self.frame_buffer = deque(maxlen=window_size)
        self.pred_history = deque(maxlen=5)

        # Pose Tracking
        self.prev_keypoints = None
        self.prev_time = 0.0
        self.VEL_THRESH = 1.5 # m/s
        self.last_trigger_time = 0.0
        
        # Color Tracking Params
        self.color_enabled = True
        self.glove_debug_pub = None # Will be created after node init
        from boxbunny_msgs.msg import GloveDetections, GloveDetection
        
        # HSV Defaults
        self.hsv_green_lower = np.array([45, 80, 50], dtype=np.uint8)
        self.hsv_green_upper = np.array([85, 255, 255], dtype=np.uint8)
        self.hsv_red_lower1 = np.array([0, 90, 50], dtype=np.uint8)
        self.hsv_red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.hsv_red_lower2 = np.array([160, 90, 50], dtype=np.uint8)
        self.hsv_red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Tracking State
        self.dist_hist = {"left": deque(maxlen=5), "right": deque(maxlen=5)}
        self.vel_hist = {"left": deque(maxlen=3), "right": deque(maxlen=3)}
        self.last_color_punch = {"left": 0.0, "right": 0.0}
        
        # Initialize ROS
        self.imu_handler = RosImuHandler()
        self.imu_handler.start()
        
        # Publisher
        # Publisher
        waited = 0
        while self.imu_handler.node is None and waited < 50: # 5 seconds timeout
             time.sleep(0.1)
             waited += 1
             
        if self.imu_handler.node is None:
            print("FATAL: ROS Node failed to initialize in handler.")
            self.running = False
            return

        self.node = self.imu_handler.node
        
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        from std_srvs.srv import Trigger
        self.debug_pub = self.node.create_publisher(Image, '/action_debug_image', 10)
        self.glove_debug_pub = self.node.create_publisher(Image, '/glove_debug_image', 10)
        self.glove_det_pub = self.node.create_publisher(GloveDetections, 'glove_detections', 10)
        self.bridge = CvBridge()
        self.node.create_service(Trigger, 'action_predictor/load_model', self._handle_load_model)
        
        # Mode Subscription / Setting
        from std_msgs.msg import String
        self.mode_sub = self.node.create_subscription(String, '/boxbunny/detection_mode', self._on_mode_switch, 10)
        
        # Initial Mode
        if mode == 'color':
            self.color_enabled = True
            self.pose_enabled = False
            print("Mode: Color Tracking (Simple)")
        else:
            self.color_enabled = False
            self.pose_enabled = True # Default
            print("Mode: Pose Model")
        
        # Initialize Models
        print("Loading Pose model...")
        self.pose_wrapper = YOLOPoseWrapper(os.path.join(os.path.dirname(__file__), '../models/checkpoints/yolo26n-pose.pt'), device)
        
        self.action_model = None
        self.cropper = None
        
        if enable_action_model:
            self._load_action_model()
            
        if 'cuda' in self.device:
            torch.backends.cudnn.benchmark = True
        print("Starting camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        rw, rh = self._parse_res(rgb_res)
        dw, dh = self._parse_res(depth_res)
        config.enable_stream(rs.stream.color, rw, rh, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
        
        profile = None
        for i in range(5):
            try:
                profile = self.pipeline.start(config)
                print("Camera started successfully.")
                break
            except RuntimeError as e:
                print(f"Camera start failed (Attempt {i+1}/5): {e}. Retrying in 2s...")
                time.sleep(2.0)
        
        if profile is None:
             print("FATAL: Could not start camera after 5 attempts.")
             self.running = False
             return

        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        # Intrinsics for velocity calc (pixels -> meters)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        
        print("Headless Inference Running. Press Ctrl+C to stop.")
        self._loop()

    def _load_action_model(self):
        if self.action_model is not None: return
        print("Loading Action Model...")
        self.cropper = YOLOPersonCrop(os.path.join(os.path.dirname(__file__), '../models/checkpoints/yolo26n.pt'), self.device, detect_interval=5)
        self.model = load_model(self.model_config, self.model_checkpoint, self.device)
        if 'cuda' in self.device and hasattr(self.model, 'half'):
            self.model.half()
        self.action_model = self.model
        print("Action Model Loaded.")

    def _handle_load_model(self, req, res):
        self._load_action_model()
        res.success = True
        res.message = "Action Model Loaded"
        return res

    @staticmethod
    def _parse_res(text):
        parts = text.lower().split('x')
        return int(parts[0].strip()), int(parts[1].strip())
        
    def _on_mode_switch(self, msg):
        mode = msg.data.lower()
        if "color" in mode:
            self.color_enabled = True
            self.pose_enabled = False
            print("Switched to Color Tracking Mode")
        else:
            self.color_enabled = False
            self.pose_enabled = True
            print("Switched to Pose Model Mode")

    def _loop(self):
        try:
            while self.running:
                # Sync mode from IMU Handler (Service Driven)
                if self.imu_handler:
                    if self.imu_handler.simple_mode and not self.color_enabled:
                        self.color_enabled = True
                        self.pose_enabled = False
                        print("Service Sync: Switched to Color Tracking Mode")
                    elif not self.imu_handler.simple_mode and self.color_enabled:
                        self.color_enabled = False
                        self.pose_enabled = True
                        print("Service Sync: Switched to Pose Model Mode")

                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                    
                rgb = np.asanyarray(color_frame.get_data())
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
                
                # Pose Estimation
                kp = None
                probs = None
                bbox = None
                
                if self.pose_enabled:
                    pose_res = self.pose_wrapper.predict(rgb)
                    kp = pose_res.get('keypoints')
                    
                    # Calculate & Publish Height
                    if kp is not None and len(kp) > 0:
                        self._calculate_height(kp, depth)
                    
                    # Check for Sudden Pose Change (Velocity Trigger)
                    trigger_label = self._check_velocity_trigger(kp, depth, time.time())
                    if trigger_label:
                         self.imu_handler.publish_action(trigger_label, 1.0, None, self.labels)

                # Color Tracking
                if self.color_enabled:
                    self._process_color_tracking(rgb, depth)
                    
                # Action Model (Runs only if Pose is active usually, or if force enabled?)
                # Logic: action model relies on pose crop? No, YOLOPersonCrop.
                # If pose is disabled, do we run action model?
                # Probably not. 'action' mode is usually pose+action.
                if self.pose_enabled and self.action_model is not None and not trigger_label:
                     probs, rgb_crop, depth_crop, bbox = self._process_frame(rgb, depth)
                     if probs is not None:
                        idx = np.argmax(probs)
                        label = self.labels[idx]
                        conf = float(probs[idx])
                        self.imu_handler.publish_action(label, conf, probs.tolist(), self.labels)
                
                # Publish Debug Image (Pose Debug)
                if self.pose_enabled:
                    self._publish_debug(rgb, bbox, probs, kp)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()

    def _check_velocity_trigger(self, kp, depth, current_time):
        now = current_time
        if kp is None:
             self.prev_keypoints = None
             return None
             
        # Indices: 9=L_Wrist, 10=R_Wrist
        if self.prev_keypoints is None:
             self.prev_keypoints = kp
             self.prev_time = now
             return None
             
        dt = now - self.prev_time
        if dt <= 0: return None
        if dt > 0.1: # Filter breaks
             self.prev_keypoints = kp
             self.prev_time = now
             return None
        if now - self.last_trigger_time < 0.5:
             self.prev_keypoints = kp
             self.prev_time = now
             return None

        h, w = depth.shape
        detected_label = None
        
        for idx, label in [(9, 'jab'), (10, 'cross')]:
             u, v, c = kp[idx]
             pu, pv, pc = self.prev_keypoints[idx]
             
             if c < 0.5 or pc < 0.5: continue
             
             dist_px = np.sqrt((u-pu)**2 + (v-pv)**2)
             vel_px = dist_px / dt
             
             u_i, v_i = int(u), int(v)
             u_i = max(0, min(w-1, u_i))
             v_i = max(0, min(h-1, v_i))
             z = depth[v_i, u_i]
             
             # Heuristic: Fast pixels (approx 200px/s) and close range
             if vel_px > 200 and z > 0 and z < 1.5:
                  self.last_trigger_time = now
                  detected_label = label
                  break
        
        self.prev_time = now
        self.prev_keypoints = kp
        return detected_label

    def _calculate_height(self, kp, depth):
        # kp shape: (17, 3) or (N, 17, 3). Assume single person or first person.
        # Ultralytics: (1, 17, 3) or (17, 3)
        kp_arr = np.array(kp)
        if kp_arr.ndim == 3:
            kp_arr = kp_arr[0] # Take first person
            
        # Filter confident points (>0.5)
        valid = kp_arr[:, 2] > 0.5
        if np.sum(valid) < 4: return # Need at least some body parts
        
        pts = kp_arr[valid, :2]
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])
        h_px = ymax - ymin
        
        # Determine Depth (Median of valid points)
        # Map (x,y) to integer coords
        zs = []
        h, w = depth.shape
        for i in range(len(kp_arr)):
            if valid[i]:
                x, y = int(kp_arr[i, 0]), int(kp_arr[i, 1])
                if 0 <= x < w and 0 <= y < h:
                    z = depth[y, x]
                    if z > 0: zs.append(z)
        
        if not zs: return
        z_m = np.median(zs)
        
        # Calculate Height in Meters
        # H = z * h_px / fy
        # Note: This is an estimation. 
        if self.fy > 0:
            height_m = (z_m * h_px) / self.fy
            # Publish
            self.imu_handler.publish_height(height_m)

    def _process_color_tracking(self, rgb, depth):
        """Run color blob tracking logic."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) # RGB input
        # Note: Tracker expects BGR-based HSV values usually? 
        # OpenCV reads BGR. But rs gives RGB.
        # If thresholds were tuned on BGR->HSV, then Red is around 0/180.
        # If RGB->HSV, Red is still around 0/180. The Hue channel is consistent.
        # But verify thresholds?
        # Realsense gives RGB.
        
        detections = []
        debug_img = rgb.copy() # RGB
        
        # Build Masks
        green_mask = cv2.inRange(hsv, self.hsv_green_lower, self.hsv_green_upper)
        red_mask1 = cv2.inRange(hsv, self.hsv_red_lower1, self.hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv, self.hsv_red_lower2, self.hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Improve Masks with Morphology
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # User requested: Green=Right, Red=Left
        masks = {"right": green_mask, "left": red_mask}
        
        now = time.time()
        
        for glove, mask in masks.items():
            # Find Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < 800: continue
            
            x, y, w, h = cv2.boundingRect(contour)
            # Depth logic
            roi_depth = depth[y:y+h, x:x+w]
            dist_m = 0.0
            valid = roi_depth[roi_depth > 0]
            if valid.size > 10:
                # depth is already in meters (scaled in _loop)
                dist_m = float(np.median(valid))
            else:
                continue

            # Tracking
            from boxbunny_msgs.msg import GloveDetection
            
            self.dist_hist[glove].append(dist_m)
            smooth_dist = np.mean(self.dist_hist[glove])
            
            # Velocity
            velocity = 0.0
            if len(self.dist_hist[glove]) >= 2:
                prev = self.dist_hist[glove][-2]
                velocity = (prev - smooth_dist) * 30.0 # ~30fps
            self.vel_hist[glove].append(velocity)
            
            # Trigger Logic
            # Thresholds: Vel > 0.8m/s (easier trigger), Dist < 0.8m
            if velocity > 0.8 and smooth_dist < 0.8:
                 if (now - self.last_color_punch[glove]) > 0.5:
                     self.last_color_punch[glove] = now
                     print(f"COLOR PUNCH: {glove} v={velocity:.2f}")
                     # Trigger Drill
                     label = 'jab' if glove == 'left' else 'cross'
                     self.imu_handler.publish_action(label, 1.0, None, self.labels)
            
            # Draw
            # Left=Red, Right=Green (RGB)
            color = (255, 0, 0) if glove == 'left' else (0, 255, 0)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"{glove} {smooth_dist:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            det = GloveDetection()
            det.glove = glove
            det.x, det.y, det.w, det.h = x, y, w, h
            detections.append(det)

        # Publish Debug
        if self.color_enabled:
             vis_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR) # Publish as BGR for consistency
             msg = self.bridge.cv2_to_imgmsg(vis_bgr, "bgr8")
             self.glove_debug_pub.publish(msg)
            
    def _process_frame(self, rgb, depth):
        # Same logic as main class
        rgb_crop, depth_crop, bbox = self.cropper(rgb, depth, output_size=self.crop_size)
        
        rgbd = create_rgbd_tensor(rgb_crop, depth_crop, max_depth=self.max_depth)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        self.frame_buffer.append(rgbd)
        
        probs = None
        if len(self.frame_buffer) >= self.window_size:
            frames = np.stack(list(self.frame_buffer))
            if self.use_delta:
                delta = np.diff(frames, axis=0, prepend=frames[:1])
                frames = np.concatenate([frames, delta], axis=1)
                
            frames = torch.from_numpy(frames).float().unsqueeze(0).to(self.device)
            if 'cuda' in self.device: frames = frames.half()
            
            with torch.inference_mode():
                probs = self.model.predict(frames, return_probs=True)
            
            probs = probs.float().cpu().numpy()[0]
            self.pred_history.append(probs)
            probs = np.mean(list(self.pred_history), axis=0)
            
        return probs, rgb_crop, depth_crop, bbox

    def _publish_debug(self, rgb, bbox, probs, keypoints=None):
        vis = rgb.copy()
        
        # Draw Skeleton
        if keypoints is not None:
             self._draw_skeleton(vis, keypoints)
             
        if bbox is not None:
             x1, y1, x2, y2 = bbox.astype(int)
             cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
             
        if probs is not None:
             idx = np.argmax(probs)
             label = f"{self.labels[idx]}: {probs[idx]:.2f}"
             cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        try:
             vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
             msg = self.bridge.cv2_to_imgmsg(vis_bgr, "bgr8")
             self.debug_pub.publish(msg)
        except Exception:
             pass

    def _draw_skeleton(self, img, kp):
        # Connect keypoints
        # 5-7 (L-Shoulder to L-Elbow), 7-9 (L-Elbow to L-Wrist)
        # 6-8 (R-Shoulder to R-Elbow), 8-10 (R-Elbow to R-Wrist)
        # 5-6 (Shoulders)
        connections = [(5,7), (7,9), (6,8), (8,10), (5,6), (5,11), (6,12), (11,12)]
        
        h, w, _ = img.shape
        
        for i, (p1, p2) in enumerate(connections):
            if p1 < len(kp) and p2 < len(kp):
                pt1 = kp[p1]
                pt2 = kp[p2]
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw Points
        for i, p in enumerate(kp):
            if p[2] > 0.5:
                 cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser(
        description='Live RGB-D Action Recognition GUI'
    )
    
    parser.add_argument('--model-config', type=str, default='configs/rgbd_boxing.py',
                        help='Path to model config')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--yolo-model', type=str, default='../models/checkpoints/yolo26n.pt',
                        help='Path to YOLO model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Inference device')
    parser.add_argument('--window-size', type=int, default=16,
                        help='Inference window size')
    parser.add_argument('--causal', action='store_true', default=True,
                        help='Use causal (anticipation) mode')
    parser.add_argument('--no-causal', action='store_false', dest='causal',
                        help='Use non-causal mode')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera FPS (default: 30)')

    parser.add_argument('--rgb-res', type=str, default='640x480',
                        help='RGB resolution (e.g. 640x480, 960x540, 1280x720)')
    parser.add_argument('--depth-res', type=str, default='640x480',
                        help='Depth resolution (e.g. 640x480, 848x480, 1280x720)')
    parser.add_argument('--detect-interval', type=int, default=5,
                        help='Run YOLO detection every N frames (default: 5)')
    
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI and publish to ROS')
    parser.add_argument('--enable-action-model', action='store_true',
                        help='Enable Action Model in Headless mode')
    parser.add_argument('--mode', type=str, default='pose',
                        help='Detection mode: pose, color, hybrid')
    
    args = parser.parse_args()
    
    if args.headless:
        HeadlessInference(
            model_config=args.model_config,
            model_checkpoint=args.model_checkpoint,
            yolo_model=args.yolo_model,
            window_size=args.window_size,
            device=args.device,
            causal=args.causal,
            fps=args.fps,
            rgb_res=args.rgb_res,
            depth_res=args.depth_res,
            detect_interval=args.detect_interval,
            enable_action_model=args.enable_action_model,
            mode=args.mode,
        )
        return
    
    # Create GUI
    root = tk.Tk()
    root.geometry("1400x800")
    
    app = LiveRGBDInference(
        root=root,
        model_config=args.model_config,
        model_checkpoint=args.model_checkpoint,
        yolo_model=args.yolo_model,
        window_size=args.window_size,
        device=args.device,
        causal=args.causal,
        fps=args.fps,
        rgb_res=args.rgb_res,
        depth_res=args.depth_res,
        detect_interval=args.detect_interval,
        enable_action_model=args.enable_action_model,
    )
    
    def on_close():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == '__main__':
    main()
