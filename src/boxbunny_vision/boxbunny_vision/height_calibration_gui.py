#!/usr/bin/env python3
"""
Height Calibration GUI - Standalone
Dedicated interface for calibrating user height using pose estimation + depth.
This is completely independent and runs its own camera + pose detection.
"""

import sys
import os
import time
import signal
import threading
import numpy as np
import cv2
from typing import Optional
from collections import deque

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    print("Please install PySide6: pip install PySide6")
    sys.exit(1)

import pyrealsense2 as rs

# Add action_prediction to path for YOLO wrapper
sys.path.insert(0, "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/action_prediction")
from tools.lib.hybrid_detectors import YOLOPoseWrapper


APP_STYLESHEET = """
QWidget { background-color: #111317; color: #E6E6E6; font-family: 'DejaVu Sans'; }
QGroupBox { border: 1px solid #2A2E36; border-radius: 8px; margin-top: 8px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #C0C4CC; }
QPushButton { background-color: #2B3240; border: 1px solid #394151; padding: 8px 16px; border-radius: 6px; font-size: 14px; }
QPushButton:hover { background-color: #394151; }
QPushButton:pressed { background-color: #202633; }
QPushButton:disabled { background-color: #1a1d24; color: #666; }
QLabel { color: #E6E6E6; }
QProgressBar { border: 1px solid #30363d; border-radius: 4px; background-color: #0d1117; text-align: center; }
QProgressBar::chunk { background-color: #238636; border-radius: 3px; }
"""


class CameraWorker(QtCore.QThread):
    """Worker thread that runs camera + pose detection."""
    image = QtCore.Signal(object)
    height = QtCore.Signal(float)
    status = QtCore.Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.pipeline = None
        self.pose_wrapper = None
        self.depth_scale = 0.001
        self.fy = 0.0
        
    def run(self):
        try:
            self._init_camera()
            self._init_pose()
            self._loop()
        except Exception as e:
            self.status.emit(f"Error: {e}")
        finally:
            if self.pipeline:
                self.pipeline.stop()
    
    def _init_camera(self):
        self.status.emit("Starting camera...")
        os.environ['LD_PRELOAD'] = '/usr/local/lib/librealsense2.so'
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = None
        for i in range(5):
            try:
                profile = self.pipeline.start(config)
                break
            except RuntimeError as e:
                self.status.emit(f"Camera retry {i+1}/5...")
                time.sleep(1.0)
        
        if profile is None:
            raise RuntimeError("Could not start camera")
            
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        # Get intrinsics for height calculation
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fy = intr.fy
        
        self.status.emit("Camera ready")
    
    def _init_pose(self):
        self.status.emit("Loading pose model...")
        model_path = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/models/checkpoints/yolo26n-pose.pt"
        self.pose_wrapper = YOLOPoseWrapper(model_path, 'cuda:0')
        self.status.emit("Ready - Stand in front of camera")
    
    def _loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Get frames
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
                
                # Run pose detection
                vis = rgb.copy()
                pose_res = self.pose_wrapper.predict(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                kp = pose_res.get('keypoints')
                bbox = pose_res.get('bbox')
                
                # Draw skeleton and bbox
                if kp is not None:
                    self._draw_skeleton(vis, kp)
                
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Calculate height using hybrid approach (bbox top + ankle keypoints)
                    height_m = self._calculate_height_hybrid(kp, bbox, depth)
                    if height_m > 0.5:  # Sanity check: at least 50cm
                        self.height.emit(height_m)
                
                self.image.emit(vis)
                
            except Exception as e:
                pass  # Frame timeout or error
    
    def _draw_skeleton(self, img, kp):
        """Draw skeleton on image."""
        kp_arr = np.array(kp)
        if kp_arr.ndim == 3:
            kp_arr = kp_arr[0]
        
        connections = [(5,7), (7,9), (6,8), (8,10), (5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
        
        for p1, p2 in connections:
            if p1 < len(kp_arr) and p2 < len(kp_arr):
                pt1, pt2 = kp_arr[p1], kp_arr[p2]
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 2)
        
        for i, p in enumerate(kp_arr):
            if p[2] > 0.5:
                cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
    
    def _calculate_height_hybrid(self, kp, bbox, depth):
        """
        Calculate height using hybrid approach for maximum accuracy:
        - TOP: Bounding box top edge (captures top of head including hair)
        - BOTTOM: Ankle keypoints (more precise than bbox bottom)
        
        This combines the best of both methods:
        - BBox top is better than nose keypoint (captures full head)
        - Ankle keypoints are better than bbox bottom (avoids ground padding)
        """
        kp_arr = np.array(kp)
        if kp_arr.ndim == 3:
            kp_arr = kp_arr[0]
        
        h, w = depth.shape
        
        # COCO keypoint indices for ankles
        IDX_L_ANKLE = 15
        IDX_R_ANKLE = 16
        
        # Get bbox top (y_top)
        x1, y_top, x2, y_bottom_bbox = bbox
        y_top = int(y_top)
        
        # Get ankle positions (more accurate bottom than bbox)
        l_ankle = kp_arr[IDX_L_ANKLE]
        r_ankle = kp_arr[IDX_R_ANKLE]
        
        # Use the lowest confident ankle as bottom
        ankles = []
        if l_ankle[2] > 0.3:  # Lower confidence threshold for ankles
            ankles.append(l_ankle)
        if r_ankle[2] > 0.3:
            ankles.append(r_ankle)
        
        if not ankles:
            # Fallback: use bbox bottom if no ankles detected
            y_bottom = int(y_bottom_bbox)
        else:
            # Use the average Y of detected ankles
            y_bottom = int(np.mean([a[1] for a in ankles]))
        
        # Calculate pixel height
        h_px = y_bottom - y_top
        if h_px <= 0:
            return 0.0
        
        # Get depth at the person's torso (more stable than head/feet)
        # Use center of bbox horizontally, and 1/3 down from top (chest area)
        center_x = int((x1 + x2) / 2)
        torso_y = int(y_top + h_px * 0.35)  # Chest area
        
        # Sample depth in a small region around torso for robustness
        sample_radius = 10
        depths = []
        for dy in range(-sample_radius, sample_radius + 1, 3):
            for dx in range(-sample_radius, sample_radius + 1, 3):
                sx, sy = center_x + dx, torso_y + dy
                if 0 <= sx < w and 0 <= sy < h:
                    z = depth[sy, sx]
                    if 0.3 < z < 5.0:  # Valid depth range (30cm to 5m)
                        depths.append(z)
        
        if not depths or self.fy <= 0:
            return 0.0
        
        z_m = np.median(depths)
        
        # Convert pixel height to meters using pinhole camera model
        # height_meters = (pixel_height * depth) / focal_length_y
        height_m = (z_m * h_px) / self.fy
        
        return height_m
    
    def _calculate_height(self, kp, depth):
        """Legacy method - kept for reference. Use _calculate_height_hybrid instead."""
        kp_arr = np.array(kp)
        if kp_arr.ndim == 3:
            kp_arr = kp_arr[0]
        
        # Filter confident points
        valid = kp_arr[:, 2] > 0.5
        if np.sum(valid) < 4:
            return 0.0
        
        pts = kp_arr[valid, :2]
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])
        h_px = ymax - ymin
        
        # Get median depth of valid keypoints
        h, w = depth.shape
        zs = []
        for i in range(len(kp_arr)):
            if valid[i]:
                x, y = int(kp_arr[i, 0]), int(kp_arr[i, 1])
                if 0 <= x < w and 0 <= y < h:
                    z = depth[y, x]
                    if z > 0:
                        zs.append(z)
        
        if not zs or self.fy <= 0:
            return 0.0
        
        z_m = np.median(zs)
        height_m = (z_m * h_px) / self.fy
        return height_m
    
    def stop(self):
        self.running = False


class HeightCalibrationGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BoxBunny Height Calibration")
        self.resize(950, 700)
        
        self.worker = CameraWorker()
        self.worker.image.connect(self._update_image)
        self.worker.height.connect(self._on_height)
        self.worker.status.connect(self._on_status)
        self.worker.start()
        
        # Calibration state
        self.calibrating = False
        self.countdown_value = 0
        self.countdown_timer = QtCore.QTimer()
        self.countdown_timer.timeout.connect(self._countdown_tick)
        
        # Height samples for averaging
        self.height_samples = deque(maxlen=60)
        self.calibrated_height: Optional[float] = None
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        # Left: Video Feed
        video_group = QtWidgets.QGroupBox("Live Camera Feed (Pose + BBox)")
        v_layout = QtWidgets.QVBoxLayout(video_group)
        
        self.image_label = QtWidgets.QLabel("Starting camera and pose model...")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; border-radius: 4px; font-size: 14px;")
        self.image_label.setMinimumSize(640, 480)
        v_layout.addWidget(self.image_label)
        
        layout.addWidget(video_group, 2)
        
        # Right: Controls
        right_panel = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # Instructions
        instr_group = QtWidgets.QGroupBox("Instructions")
        i_layout = QtWidgets.QVBoxLayout(instr_group)
        
        instr_items = [
            "1. Position yourself so your FULL BODY is visible.",
            "2. Stand approximately 2-3 meters from the camera.",
            "3. Make sure skeleton overlay appears on you.",
            "4. Click CALIBRATE HEIGHT and hold still.",
            "5. The system will measure your height using pose + depth."
        ]
        
        for item in instr_items:
            lbl = QtWidgets.QLabel(item)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 13px; margin: 2px 0;")
            i_layout.addWidget(lbl)
        
        r_layout.addWidget(instr_group)
        
        # Live Height Display
        height_group = QtWidgets.QGroupBox("Height Measurement")
        h_layout = QtWidgets.QVBoxLayout(height_group)
        
        self.lbl_live_height = QtWidgets.QLabel("Live: -- m")
        self.lbl_live_height.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_live_height.setStyleSheet("font-size: 24px; color: #8b949e;")
        h_layout.addWidget(self.lbl_live_height)
        
        self.lbl_calibrated = QtWidgets.QLabel("Calibrated: -- m")
        self.lbl_calibrated.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_calibrated.setStyleSheet("font-size: 32px; font-weight: bold; color: #3fb950;")
        h_layout.addWidget(self.lbl_calibrated)
        
        self.lbl_status = QtWidgets.QLabel("Status: Initializing...")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 14px; color: #e3b341;")
        h_layout.addWidget(self.lbl_status)
        
        r_layout.addWidget(height_group)
        
        # Calibrate Button
        self.btn_calibrate = QtWidgets.QPushButton("CALIBRATE HEIGHT")
        self.btn_calibrate.setStyleSheet("""
            QPushButton { background-color: #238636; font-weight: bold; font-size: 16px; padding: 16px; }
            QPushButton:hover { background-color: #2ea043; }
            QPushButton:disabled { background-color: #1a1d24; color: #666; }
        """)
        self.btn_calibrate.clicked.connect(self._start_calibration)
        r_layout.addWidget(self.btn_calibrate)
        
        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        r_layout.addWidget(self.progress)
        
        r_layout.addStretch()
        
        # Tips
        tips_group = QtWidgets.QGroupBox("Tips")
        t_layout = QtWidgets.QVBoxLayout(tips_group)
        tips_text = QtWidgets.QLabel(
            "• Ensure good lighting for accurate pose detection.\n"
            "• Stand straight with arms at your sides.\n"
            "• The bounding box should cover your entire body.\n"
            "• Multiple samples are averaged for accuracy."
        )
        tips_text.setWordWrap(True)
        tips_text.setStyleSheet("font-size: 12px; color: #8b949e;")
        t_layout.addWidget(tips_text)
        r_layout.addWidget(tips_group)
        
        layout.addWidget(right_panel, 1)

    def _start_calibration(self):
        if self.calibrating:
            return
            
        self.calibrating = True
        self.height_samples.clear()
        self.btn_calibrate.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        
        self.countdown_value = 3
        self.lbl_status.setText("Get ready! Stand back...")
        self.lbl_status.setStyleSheet("font-size: 14px; color: #58a6ff;")
        self.countdown_timer.start(1000)
        
    def _countdown_tick(self):
        if self.countdown_value > 0:
            self.lbl_status.setText(f"Calibrating in {self.countdown_value}...")
            self.countdown_value -= 1
        else:
            self.countdown_timer.stop()
            self._do_calibration()
    
    def _do_calibration(self):
        self.lbl_status.setText("Hold still... Measuring...")
        self.lbl_status.setStyleSheet("font-size: 14px; color: #e3b341;")
        
        self.sample_timer = QtCore.QTimer()
        self.sample_start = time.time()
        self.sample_timer.timeout.connect(self._sample_tick)
        self.sample_timer.start(50)
        
    def _sample_tick(self):
        elapsed = time.time() - self.sample_start
        self.progress.setValue(int(min(100, elapsed / 2.0 * 100)))
        
        if elapsed >= 2.0:
            self.sample_timer.stop()
            self._finish_calibration()
    
    def _finish_calibration(self):
        self.calibrating = False
        self.btn_calibrate.setEnabled(True)
        
        if len(self.height_samples) >= 10:
            samples = list(self.height_samples)
            samples.sort()
            trim = len(samples) // 5
            if trim > 0:
                samples = samples[trim:-trim]
            
            self.calibrated_height = sum(samples) / len(samples) if samples else 0.0
            
            if self.calibrated_height > 0.5:
                self.lbl_calibrated.setText(f"Calibrated: {self.calibrated_height:.2f} m")
                self.lbl_calibrated.setStyleSheet("font-size: 32px; font-weight: bold; color: #3fb950;")
                self.lbl_status.setText("✓ Height calibrated successfully!")
                self.lbl_status.setStyleSheet("font-size: 14px; color: #3fb950;")
            else:
                self.lbl_status.setText("⚠ Could not measure height. Try standing further back.")
                self.lbl_status.setStyleSheet("font-size: 14px; color: #f85149;")
        else:
            self.lbl_status.setText("⚠ Not enough samples. Is pose detection working?")
            self.lbl_status.setStyleSheet("font-size: 14px; color: #f85149;")
        
        self.progress.hide()

    def _update_image(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Add calibration overlay
        if self.calibrating and self.countdown_value > 0:
            cv2.putText(rgb_img, str(self.countdown_value), 
                       (w//2 - 40, h//2 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (88, 166, 255), 8)
        elif self.calibrating:
            cv2.putText(rgb_img, "HOLD STILL", 
                       (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (227, 179, 65), 4)
        
        qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _on_height(self, height_val):
        if height_val > 0.1:
            self.lbl_live_height.setText(f"Live: {height_val:.2f} m")
            
            if self.calibrating and self.countdown_value <= 0:
                self.height_samples.append(height_val)

    def _on_status(self, status):
        self.lbl_status.setText(f"Status: {status}")

    def closeEvent(self, event):
        self.countdown_timer.stop()
        if hasattr(self, 'sample_timer'):
            self.sample_timer.stop()
        self.worker.stop()
        self.worker.quit()
        self.worker.wait(2000)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    gui = HeightCalibrationGui()
    gui.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
