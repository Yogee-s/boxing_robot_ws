#!/usr/bin/env python3
"""
Punch Detection GUI - Standalone Color Tracking
Detects punches using color tracking:
- Left glove = RED = Jab
- Right glove = GREEN = Cross
When glove approaches camera (distance decreases), detect as punch.
"""

import sys
import os
import time
import signal
import numpy as np
import cv2
from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    print("Please install PySide6: pip install PySide6")
    sys.exit(1)

import pyrealsense2 as rs


APP_STYLESHEET = """
QWidget { background-color: #111317; color: #E6E6E6; font-family: 'DejaVu Sans'; }
QGroupBox { border: 1px solid #2A2E36; border-radius: 8px; margin-top: 8px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #C0C4CC; }
QPushButton { background-color: #2B3240; border: 1px solid #394151; padding: 8px 16px; border-radius: 6px; font-size: 14px; }
QPushButton:hover { background-color: #394151; }
QPushButton:pressed { background-color: #202633; }
QLabel { color: #E6E6E6; }
QListWidget { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; }
QListWidget::item { padding: 4px; }
"""


@dataclass
class GloveDetection:
    glove: str  # "left" or "right"
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)
    distance_m: float
    velocity_mps: float


class ColorTracker:
    """Color-based glove tracker using HSV."""
    
    def __init__(self):
        # HSV ranges for red (left glove) - red wraps around in HSV
        self.hsv_red_lower1 = np.array([0, 100, 80], dtype=np.uint8)
        self.hsv_red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.hsv_red_lower2 = np.array([160, 100, 80], dtype=np.uint8)
        self.hsv_red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # HSV ranges for green (right glove)
        self.hsv_green_lower = np.array([40, 80, 60], dtype=np.uint8)
        self.hsv_green_upper = np.array([85, 255, 255], dtype=np.uint8)
        
        # Tracking state
        self.prev_dist = {"left": deque(maxlen=5), "right": deque(maxlen=5)}
        self.prev_time = {"left": 0.0, "right": 0.0}
        
        # Max detection distance - ignore background colors beyond this
        self.max_detection_distance_m = 2.0
        
        # Morphology kernel
        self.kernel = np.ones((5, 5), np.uint8)
        
    def detect(self, rgb: np.ndarray, depth: np.ndarray, depth_scale: float) -> List[GloveDetection]:
        """Detect gloves and calculate distances/velocities."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        detections = []
        now = time.time()
        
        # Detect red (left glove - Jab)
        red_mask1 = cv2.inRange(hsv, self.hsv_red_lower1, self.hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv, self.hsv_red_lower2, self.hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)
        
        red_det = self._find_glove(red_mask, depth, depth_scale, "left", now)
        if red_det:
            detections.append(red_det)
        
        # Detect green (right glove - Cross)
        green_mask = cv2.inRange(hsv, self.hsv_green_lower, self.hsv_green_upper)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.kernel)
        
        green_det = self._find_glove(green_mask, depth, depth_scale, "right", now)
        if green_det:
            detections.append(green_det)
        
        return detections
    
    def _find_glove(self, mask: np.ndarray, depth: np.ndarray, depth_scale: float, 
                    glove: str, now: float) -> Optional[GloveDetection]:
        """Find largest contour and calculate detection."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 500:  # Minimum area threshold
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, x + w, y + h)
        cx, cy = x + w // 2, y + h // 2
        
        # Get distance from depth
        dh, dw = depth.shape
        if 0 <= cx < dw and 0 <= cy < dh:
            # Sample depth in center region
            x1 = max(0, cx - 10)
            x2 = min(dw, cx + 10)
            y1 = max(0, cy - 10)
            y2 = min(dh, cy + 10)
            
            region = depth[y1:y2, x1:x2]
            valid = region[region > 0]
            
            if len(valid) > 0:
                dist_m = float(np.median(valid)) * depth_scale
            else:
                dist_m = 0.0
        else:
            dist_m = 0.0
        
        # Filter out background colors beyond max detection distance
        if dist_m > self.max_detection_distance_m:
            return None  # Ignore - too far away (likely background)
        
        # Calculate velocity (approach speed)
        velocity = 0.0
        if dist_m > 0:
            self.prev_dist[glove].append(dist_m)
            
            if len(self.prev_dist[glove]) >= 2:
                dt = now - self.prev_time[glove]
                if 0.01 < dt < 0.5:  # Valid time delta
                    prev_d = list(self.prev_dist[glove])[-2]
                    # Negative velocity = approaching camera
                    velocity = (prev_d - dist_m) / dt
            
            self.prev_time[glove] = now
        
        return GloveDetection(
            glove=glove,
            bbox=bbox,
            center=(cx, cy),
            distance_m=dist_m,
            velocity_mps=velocity
        )


class CameraWorker(QtCore.QThread):
    """Worker thread that runs camera + color tracking."""
    image = QtCore.Signal(object)
    detection = QtCore.Signal(object)  # GloveDetection
    punch = QtCore.Signal(str, float)  # punch_type, velocity
    status = QtCore.Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.pipeline = None
        self.tracker = ColorTracker()
        self.depth_scale = 0.001
        
        # Punch detection state
        self.last_punch_time = {"left": 0.0, "right": 0.0}
        self.punch_cooldown = 0.5  # seconds
        self.velocity_threshold = 1.0  # m/s approaching
        self.distance_threshold = 0.8  # meters - detect when close
        
    def run(self):
        try:
            self._init_camera()
            self.status.emit("Tracking active - Throw some punches!")
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
        self.status.emit("Camera ready")
    
    def _loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                
                # Run color tracking
                detections = self.tracker.detect(rgb, depth, self.depth_scale)
                
                # Draw detections on image
                vis = rgb.copy()
                now = time.time()
                
                for det in detections:
                    # Draw bounding box
                    x1, y1, x2, y2 = det.bbox
                    color = (0, 0, 255) if det.glove == "left" else (0, 255, 0)  # Red/Green in BGR
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label
                    label = f"{det.glove.upper()}: {det.distance_m:.2f}m"
                    cv2.putText(vis, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw velocity indicator
                    if det.velocity_mps > 0.5:
                        vel_label = f"V: {det.velocity_mps:.1f} m/s"
                        cv2.putText(vis, vel_label, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # Emit detection
                    self.detection.emit(det)
                    
                    # Check for punch
                    self._check_punch(det, now)
                
                self.image.emit(vis)
                
            except Exception as e:
                pass
    
    def _check_punch(self, det: GloveDetection, now: float):
        """Check if detection constitutes a punch."""
        glove = det.glove
        
        # Check cooldown
        if now - self.last_punch_time[glove] < self.punch_cooldown:
            return
        
        # Punch detected if:
        # 1. Glove is close (within threshold distance)
        # 2. Glove is approaching fast (positive velocity = moving toward camera)
        if det.distance_m < self.distance_threshold and det.velocity_mps > self.velocity_threshold:
            punch_type = "JAB" if glove == "left" else "CROSS"
            self.punch.emit(punch_type, det.velocity_mps)
            self.last_punch_time[glove] = now
    
    def stop(self):
        self.running = False


class PunchDetectionGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BoxBunny Punch Detection (Color Tracking)")
        self.resize(1000, 700)
        
        self.worker = CameraWorker()
        self.worker.image.connect(self._update_image)
        self.worker.detection.connect(self._on_detection)
        self.worker.punch.connect(self._on_punch)
        self.worker.status.connect(self._on_status)
        self.worker.start()
        
        self.punch_count = {"JAB": 0, "CROSS": 0}
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        # Left: Video Feed
        video_group = QtWidgets.QGroupBox("Live Camera Feed (Color Tracking)")
        v_layout = QtWidgets.QVBoxLayout(video_group)
        
        self.image_label = QtWidgets.QLabel("Starting camera...")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; border-radius: 4px; font-size: 14px;")
        self.image_label.setMinimumSize(640, 480)
        v_layout.addWidget(self.image_label)
        
        # Punch indicator
        self.lbl_punch = QtWidgets.QLabel("")
        self.lbl_punch.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_punch.setStyleSheet("font-size: 48px; font-weight: bold;")
        self.lbl_punch.setFixedHeight(70)
        v_layout.addWidget(self.lbl_punch)
        
        layout.addWidget(video_group, 2)
        
        # Right: Controls & Stats
        right_panel = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # Instructions
        instr_group = QtWidgets.QGroupBox("How It Works")
        i_layout = QtWidgets.QVBoxLayout(instr_group)
        
        instr_items = [
            "• LEFT glove (RED) = JAB",
            "• RIGHT glove (GREEN) = CROSS",
            "• Punch is detected when glove approaches camera quickly",
            "• Make sure gloves are clearly visible",
        ]
        
        for item in instr_items:
            lbl = QtWidgets.QLabel(item)
            lbl.setStyleSheet("font-size: 13px; margin: 2px 0;")
            i_layout.addWidget(lbl)
        
        r_layout.addWidget(instr_group)
        
        # Live Stats
        stats_group = QtWidgets.QGroupBox("Live Tracking")
        s_layout = QtWidgets.QVBoxLayout(stats_group)
        
        self.lbl_left = QtWidgets.QLabel("LEFT (Red): -- m | -- m/s")
        self.lbl_left.setStyleSheet("font-size: 14px; color: #FF6B6B; font-weight: bold;")
        s_layout.addWidget(self.lbl_left)
        
        self.lbl_right = QtWidgets.QLabel("RIGHT (Green): -- m | -- m/s")
        self.lbl_right.setStyleSheet("font-size: 14px; color: #4DFF88; font-weight: bold;")
        s_layout.addWidget(self.lbl_right)
        
        self.lbl_status = QtWidgets.QLabel("Status: Initializing...")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #e3b341;")
        s_layout.addWidget(self.lbl_status)
        
        r_layout.addWidget(stats_group)
        
        # Punch Counter
        counter_group = QtWidgets.QGroupBox("Punch Counter")
        c_layout = QtWidgets.QHBoxLayout(counter_group)
        
        self.lbl_jab_count = QtWidgets.QLabel("JAB\n0")
        self.lbl_jab_count.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_jab_count.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF6B6B;")
        c_layout.addWidget(self.lbl_jab_count)
        
        self.lbl_cross_count = QtWidgets.QLabel("CROSS\n0")
        self.lbl_cross_count.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_cross_count.setStyleSheet("font-size: 24px; font-weight: bold; color: #4DFF88;")
        c_layout.addWidget(self.lbl_cross_count)
        
        r_layout.addWidget(counter_group)
        
        # Punch Log
        log_group = QtWidgets.QGroupBox("Punch Log")
        l_layout = QtWidgets.QVBoxLayout(log_group)
        
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setMaximumHeight(200)
        l_layout.addWidget(self.log_list)
        
        btn_clear = QtWidgets.QPushButton("Clear Log & Reset")
        btn_clear.clicked.connect(self._clear_log)
        l_layout.addWidget(btn_clear)
        
        r_layout.addWidget(log_group)
        
        r_layout.addStretch()
        
        layout.addWidget(right_panel, 1)

    def _update_image(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _on_detection(self, det: GloveDetection):
        """Update live tracking display."""
        txt = f"{det.glove.upper()}: {det.distance_m:.2f}m | {det.velocity_mps:.1f} m/s"
        if det.glove == "left":
            self.lbl_left.setText(f"LEFT (Red): {det.distance_m:.2f}m | {det.velocity_mps:.1f} m/s")
        else:
            self.lbl_right.setText(f"RIGHT (Green): {det.distance_m:.2f}m | {det.velocity_mps:.1f} m/s")

    def _on_punch(self, punch_type: str, velocity: float):
        """Handle punch detection."""
        self.punch_count[punch_type] += 1
        
        # Update counter
        self.lbl_jab_count.setText(f"JAB\n{self.punch_count['JAB']}")
        self.lbl_cross_count.setText(f"CROSS\n{self.punch_count['CROSS']}")
        
        # Show punch indicator
        color = "#FF6B6B" if punch_type == "JAB" else "#4DFF88"
        self.lbl_punch.setText(punch_type)
        self.lbl_punch.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {color};")
        
        # Clear indicator after 500ms
        QtCore.QTimer.singleShot(500, lambda: self.lbl_punch.setText(""))
        
        # Add to log
        ts = time.strftime("%H:%M:%S")
        item = QtWidgets.QListWidgetItem(f"[{ts}] {punch_type} ({velocity:.1f} m/s)")
        item.setForeground(QtGui.QColor(color))
        self.log_list.insertItem(0, item)
        
        # Keep log size reasonable
        while self.log_list.count() > 50:
            self.log_list.takeItem(self.log_list.count() - 1)

    def _on_status(self, status: str):
        self.lbl_status.setText(f"Status: {status}")

    def _clear_log(self):
        self.log_list.clear()
        self.punch_count = {"JAB": 0, "CROSS": 0}
        self.lbl_jab_count.setText("JAB\n0")
        self.lbl_cross_count.setText("CROSS\n0")

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.quit()
        self.worker.wait(2000)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    gui = PunchDetectionGui()
    gui.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
