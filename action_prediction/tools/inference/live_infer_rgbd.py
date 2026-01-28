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
from typing import Optional, List

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

import torch
from tools.lib.yolo_person_crop import YOLOPersonCrop, create_rgbd_tensor
from tools.lib.rgbd_model import load_model


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
    ):
        self.root = root
        self.root.title(f"RGB-D Boxing Action Recognition ({fps} FPS)")
        self.device = device
        self.window_size = window_size
        self.causal = causal
        self.labels = labels or DEFAULT_LABELS
        self.fps = fps
        self.rgb_res = self._parse_res(rgb_res)
        self.depth_res = self._parse_res(depth_res)
        
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
        self.monitor = ResourceMonitor()
        
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
                probs, rgb_crop, depth_crop, bbox = self._process_frame(rgb, depth)
                
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
        chk_bbox.pack(anchor='w')
        
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
    
    def _init_async(self):
        """Initialize camera and models in background."""
        def init():
            try:
                self._update_status("Loading YOLO model...")
                self.cropper = YOLOPersonCrop(self.yolo_model, self.device)
                
                self._update_status("Loading action model...")
                
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
                    # Check if it's an ONNX wrapper (which doesn't support .half() the same way)
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


def main():
    parser = argparse.ArgumentParser(
        description='Live RGB-D Action Recognition GUI'
    )
    
    parser.add_argument('--model-config', type=str, default='configs/rgbd_boxing.py',
                        help='Path to model config')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--yolo-model', type=str, default='checkpoints/yolo26m.pt',
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
    
    args = parser.parse_args()
    
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
    )
    
    def on_close():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
