import sys
import argparse
import time
import queue
import signal
import numpy as np
import cv2
import json
from typing import Optional

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    print("Please install PySide6: pip install PySide6")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32, Float32
from boxbunny_msgs.msg import PunchEvent, GloveDetections


APP_STYLESHEET = """
QWidget { background-color: #111317; color: #E6E6E6; font-family: 'DejaVu Sans'; }
QGroupBox { border: 1px solid #2A2E36; border-radius: 8px; margin-top: 8px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #C0C4CC; }
QPushButton { background-color: #2B3240; border: 1px solid #394151; padding: 6px 10px; border-radius: 6px; }
QPushButton:hover { background-color: #394151; }
QPushButton:pressed { background-color: #202633; }
QLabel { color: #E6E6E6; }
QListWidget { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; }
"""

class VisionNode(Node):
    def __init__(self, signals):
        super().__init__("vision_debug_gui")
        self.signals = signals
        self.bridge = CvBridge()
        
        # Subscriptions
        self.create_subscription(Image, "/glove_debug_image", self._on_glove_image, 10)
        self.create_subscription(Image, "/action_debug_image", self._on_action_image, 10)
        
        self.create_subscription(PunchEvent, "punch_events_raw", self._on_punch, 10)
        self.create_subscription(GloveDetections, "glove_detections", self._on_detections, 10)
        
        # Drill Subscriptions
        self.create_subscription(String, "drill_state", self._on_drill_state, 10)
        self.create_subscription(Int32, "drill_countdown", self._on_drill_countdown, 10)
        self.create_subscription(String, "drill_summary", self._on_drill_summary, 10)
        from boxbunny_msgs.msg import DrillEvent
        self.create_subscription(DrillEvent, "drill_events", self._on_drill_event, 10)
        
        self.create_subscription(Float32, "/player_height", self._on_player_height, 10)
        
        # Service Clients
        # Service Clients
        from boxbunny_msgs.srv import StartStopDrill
        self.drill_client = self.create_client(StartStopDrill, "start_stop_drill")
        
        # Mode Publisher
        self.mode_pub = self.create_publisher(String, "/boxbunny/detection_mode", 10)
        
        self.get_logger().info("Vision GUI Node initialized")

    def _on_glove_image(self, msg):
        self._emit_image(msg, "glove")

    def _on_action_image(self, msg):
        self._emit_image(msg, "action")

    def _emit_image(self, msg, source):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.signals.image.emit(cv_img, source)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

    def _on_punch(self, msg):
        # If in Reaction Drill, suppress raw punch logs to keep it clean (user request)
        if self.chk_drill_active:
             return
             
        # Add to list
        item = f"Punch: {msg.punch_type} ({msg.velocity:.1f} m/s)"
        self.list_events.addItem(item)
        self.list_events.scrollToBottom()

    def _on_height(self, h):
        self.lbl_height.setText(f"Height: {h:.2f} m")
        

    def _on_detections(self, msg):
        self.signals.detections.emit(msg)

    def _on_drill_state(self, msg):
        self.signals.drill_state.emit(msg.data)
        
    def _on_drill_countdown(self, msg):
        self.signals.drill_countdown.emit(msg.data)
        
    def _on_drill_summary(self, msg):
        self.signals.drill_summary.emit(msg.data)

    def _on_drill_event(self, msg):
        if msg.event_type == "punch_detected":
            self.signals.reaction.emit(msg.value)
        elif msg.event_type == "early_start":
            # Direct UI update not ideal from thread, but we can pass invalid value or new signal
            # Actually easier to use reaction signal with negative value? 
            # Or assume run() loop handles it? 
            # Wait, signals.reaction takes float.
            # I can emit -1.0 to signify early? Or create new signal.
            # Let's create new signal "early_start".
            pass # See RosWorker changes below

    def _on_player_height(self, msg):
        self.signals.height.emit(msg.data)

    def start_stop_drill(self, start=True):
        if not self.drill_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Drill service not ready")
            # Force UI reset if we couldn't even reach service
            self.signals.drill_state.emit("idle")
            return
            
        from boxbunny_msgs.srv import StartStopDrill
        req = StartStopDrill.Request()
        req.start = start
        req.num_trials = 3
        
        future = self.drill_client.call_async(req)
        future.add_done_callback(self._service_response)

    def _service_response(self, future):
        try:
            resp = future.result()
            if not resp.accepted:
                self.get_logger().warn(f"Drill request rejected: {resp.message}")
                # Reset UI if rejected
                self.signals.drill_state.emit("idle")
        except Exception as e:
             self.get_logger().error(f"Service call failed: {e}")
             self.signals.drill_state.emit("idle")

    def publish_mode(self, mode):
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)


class RosWorker(QtCore.QThread):
    image = QtCore.Signal(object, str)
    punch = QtCore.Signal(object)
    detections = QtCore.Signal(object)
    drill_state = QtCore.Signal(str)
    drill_countdown = QtCore.Signal(int)
    reaction = QtCore.Signal(float)
    early = QtCore.Signal(str)
    drill_summary = QtCore.Signal(str)
    height = QtCore.Signal(float)
    
    def run(self):
        rclpy.init()
        self.node = VisionNode(self)
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
                
    def toggle_drill(self, start):
        if hasattr(self, 'node'):
            QtCore.QMetaObject.invokeMethod(self, "_do_toggle", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, start))


    @QtCore.Slot(bool)
    def _do_toggle(self, start):
        self.node.start_stop_drill(start)

    def set_mode(self, mode):
        if hasattr(self, 'node'):
            QtCore.QMetaObject.invokeMethod(self, "_do_set_mode", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, mode))

    @QtCore.Slot(str)
    def _do_set_mode(self, mode):
        self.node.publish_mode(mode)

class VisionDebugGui(QtWidgets.QWidget):
    def __init__(self, start_mode="color"):
        super().__init__()
        self.start_mode = start_mode
        self.setWindowTitle("BoxBunny Vision & Drill Debugger")
        self.resize(1200, 850)
        
        self.worker = RosWorker()
        self.worker.image.connect(self._update_image)
        self.worker.punch.connect(self._on_punch)
        self.worker.detections.connect(self._update_stats)
        self.worker.drill_state.connect(self._on_drill_state)
        self.worker.drill_countdown.connect(self._on_drill_countdown)
        self.worker.reaction.connect(self._on_reaction)
        self.worker.drill_summary.connect(self._on_drill_summary)
        self.worker.early.connect(self._on_early)
        self.worker.height.connect(self._on_height)
        self.worker.start()
        
        self.chk_drill_active = False # Track state locally
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        # Left: Video Feed
        video_group = QtWidgets.QGroupBox("Live Camera Feed")
        v_layout = QtWidgets.QVBoxLayout(video_group)
        self.image_label = QtWidgets.QLabel("Waiting for video stream...")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; border-radius: 4px;")
        self.image_label.setMinimumSize(640, 480)
        v_layout.addWidget(self.image_label)
        layout.addWidget(video_group, 2)
        
        # Right: Info & Controls
        right_panel = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # 1. Drill Controls
        drill_group = QtWidgets.QGroupBox("Reaction Drill")
        d_layout = QtWidgets.QVBoxLayout(drill_group)
        
        self.lbl_drill_status = QtWidgets.QLabel("Status: IDLE")
        self.lbl_drill_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_drill_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #8b949e;")
        d_layout.addWidget(self.lbl_drill_status)
        
        self.lbl_countdown = QtWidgets.QLabel("--")
        self.lbl_countdown.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_countdown.setStyleSheet("font-size: 40px; color: #58a6ff;")
        d_layout.addWidget(self.lbl_countdown)

        self.lbl_reaction = QtWidgets.QLabel("Reaction: --")
        self.lbl_reaction.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_reaction.setStyleSheet("font-size: 24px; color: #E6E6E6;")
        d_layout.addWidget(self.lbl_reaction)
        
        self.btn_drill = QtWidgets.QPushButton("START REACTION DRILL")
        self.btn_drill.setCheckable(True)
        self.btn_drill.setStyleSheet("""
            QPushButton { background-color: #2ea043; font-weight: bold; font-size: 14px; padding: 12px; }
            QPushButton:checked { background-color: #da3633; }
        """)
        self.btn_drill.toggled.connect(self._toggle_drill)
        d_layout.addWidget(self.btn_drill)
        r_layout.addWidget(drill_group)

        # 2. Stats
        stats_group = QtWidgets.QGroupBox("Live Tracking Stats")
        form = QtWidgets.QFormLayout(stats_group)
        self.lbl_left = QtWidgets.QLabel("Dist: -- | Vel: --")
        self.lbl_right = QtWidgets.QLabel("Dist: -- | Vel: --")
        self.lbl_left.setStyleSheet("color: #4DFF88; font-weight: bold;")
        self.lbl_right.setStyleSheet("color: #FF6B6B; font-weight: bold;")
        form.addRow("Left Glove:", self.lbl_left)
        form.addRow("Right Glove:", self.lbl_right)
        
        self.lbl_height = QtWidgets.QLabel("Height: -- m")
        self.lbl_height.setStyleSheet("color: #00AAFF; font-weight: bold; font-size: 14px;")
        form.addRow("Player Height:", self.lbl_height)
        r_layout.addWidget(stats_group)
        
        # 3. Detection Mode
        mode_group = QtWidgets.QGroupBox("Detection Settings")
        m_layout = QtWidgets.QVBoxLayout(mode_group)
        m_layout.addWidget(QtWidgets.QLabel("Detection Source:"))
        self.combo_mode = QtWidgets.QComboBox()
        
        # Add both options
        self.combo_mode.addItems(["Pose Model", "Color Tracking"])
        
        if self.start_mode == "action":
            self.combo_mode.setCurrentIndex(0)
        else:
            self.combo_mode.setCurrentIndex(1)
            
        self.combo_mode.setStyleSheet("background-color: #2B3240; padding: 5px;")
        
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        m_layout.addWidget(self.combo_mode)
        r_layout.addWidget(mode_group)
        
        # 4. Reaction Log
        log_group = QtWidgets.QGroupBox("Reaction Log")
        l_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_list = QtWidgets.QListWidget()
        l_layout.addWidget(self.log_list)
        clean_btn = QtWidgets.QPushButton("Clear Log")
        clean_btn.clicked.connect(self.log_list.clear)
        l_layout.addWidget(clean_btn)
        r_layout.addWidget(log_group)
        
        layout.addWidget(right_panel, 1)

    def _toggle_drill(self, checked):
        if checked:
            self.chk_drill_active = True  # Optimistic update
            self.btn_drill.setText("STOP DRILL")
            self.worker.toggle_drill(True)
            self.lbl_reaction.setText("Reaction: --")
            self.log_list.clear()
        else:
            self.chk_drill_active = False # Optimistic update
            self.btn_drill.setText("START REACTION DRILL")
            self.worker.toggle_drill(False)

    def _on_drill_state(self, state):
        self.lbl_drill_status.setText(f"Status: {state.upper()}")
        
        if state == "idle":
             self.chk_drill_active = False
             self.lbl_countdown.setText("--")
             # Reset Button if it's still checked
             if self.btn_drill.isChecked():
                 self.btn_drill.blockSignals(True)
                 self.btn_drill.setChecked(False)
                 self.btn_drill.setText("START REACTION DRILL")
                 self.btn_drill.blockSignals(False)
        else:
             self.chk_drill_active = True
             
             # Sync Button State
             if not self.btn_drill.isChecked():
                 self.btn_drill.blockSignals(True)
                 self.btn_drill.setChecked(True)
                 self.btn_drill.setText("STOP DRILL")
                 self.btn_drill.blockSignals(False)
             
             if state == "cue":
                  self.lbl_countdown.setText("PUNCH!")
                  self.lbl_countdown.setStyleSheet("color: #ff7b72; font-size: 48px; font-weight: bold;")
             elif state == "waiting":
                  self.lbl_countdown.setText("Wait...")
                  self.lbl_countdown.setStyleSheet("color: #e3b341; font-size: 32px;")
             elif state == "countdown":
                  self.log_list.clear() # Clear old logs on start
             elif state == "early_penalty":
                  self.lbl_countdown.setText("EARLY!")
                  self.lbl_countdown.setStyleSheet("color: #ffa500; font-size: 48px; font-weight: bold;") # Orange

    def _on_mode_changed(self, index):
        # Reset Button if it's still checked (safety when switching modes)
        if self.btn_drill.isChecked():
             self.btn_drill.blockSignals(True)
             self.btn_drill.setChecked(False)
             self.btn_drill.setText("START REACTION DRILL")
             self.btn_drill.blockSignals(False)

        # Clear image to prevent "freeze" look
        self.image_label.setPixmap(QtGui.QPixmap())
        self.image_label.setText("Waiting for stream...")
        
        # Publish Mode Change
        mode_text = self.combo_mode.currentText()
        data = "color" if "Color" in mode_text else "pose"
        self.worker.set_mode(data)
        print(f"Requested Mode: {data}")
             
    def _on_drill_countdown(self, count):
        if not self.chk_drill_active:
             self.lbl_countdown.setText("--")
             return
             
        self.lbl_countdown.setText(str(count))
        self.lbl_countdown.setStyleSheet("color: #58a6ff; font-size: 64px; font-weight: bold;")

    def _on_drill_summary(self, data):
        """Handle drill stats"""
        try:
            summary = json.loads(data)
            trials = summary.get("trial_index", 0)
            avg = summary.get("mean_reaction_time_s")
            last = summary.get("last_reaction_time_s")
            
            # Log specific result
            if last is not None:
                item = f"Trial #{trials}: {last:.3f}s"
                self.log_list.addItem(item)
                self.log_list.scrollToBottom()

            # Update Label
            if avg is not None:
                self.lbl_reaction.setText(f"Last: {last:.3f}s | Avg: {avg:.3f}s")
            elif last is not None:
                self.lbl_reaction.setText(f"Last: {last:.3f}s")
                
        except Exception as e:
            print(f"Error parsing summary: {e}")

    def _on_reaction(self, value):
        # We handle this in summary now mostly, but keeping for immediate feedback if needed
        pass

    def _update_image(self, cv_img, source):
        # Filter based on mode
        mode = self.combo_mode.currentText()
        
        if "Pose" in mode and source == "action":
             pass # Allow Pose/Action feed
             
        elif "Color" in mode and source == "glove":
             pass # Allow Color/Glove feed
             
        elif "Pose" in mode and source == "glove":
             return # Suppress Glove in Pose mode
             
        elif "Color" in mode and source == "action":
             return # Suppress Action in Color mode


        # Resize to fit label while keeping aspect ratio
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Scale pixmap
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _on_punch(self, msg):
        # If in Reaction Drill, suppress raw punch logs (user request)
        if self.chk_drill_active:
             return
             
        ts = time.strftime("%H:%M:%S")
        glove = msg.glove.upper()
        etype = "PUNCH" if msg.is_punch else "EVENT"
        
        text = f"[{ts}] {glove} {etype}: {msg.approach_velocity_mps:.2f} m/s, Conf: {msg.confidence:.2f}"
        item = QtWidgets.QListWidgetItem(text)
        
        if msg.glove == "left":
            item.setForeground(QtGui.QColor("#4DFF88")) # Green
        else:
            item.setForeground(QtGui.QColor("#FF6B6B")) # Red
            
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _update_stats(self, msg):
        # msg.detections is a list of GloveDetection
        left_found = False
        right_found = False
        
        for det in msg.detections:
            txt = f"Dist: {det.distance_m:.2f}m | Vel: {det.approach_velocity_mps:.2f}m/s"
            if det.glove == "left":
                self.lbl_left.setText(txt)
                left_found = True
            elif det.glove == "right":
                self.lbl_right.setText(txt)
                right_found = True
        
        if not right_found:
            self.lbl_right.setText("Dist: -- | Vel: --")

    def _on_early(self, glove):
        self.lbl_countdown.setText("EARLY!")
        self.lbl_countdown.setStyleSheet("color: #ffa500; font-size: 48px; font-weight: bold;") # Orange

    def _on_height(self, h):
        self.lbl_height.setText(f"Height: {h:.2f} m")

    def closeEvent(self, event):
        self.worker.quit()
        self.worker.wait()
        event.accept()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="color", choices=["color", "action"], help="Start mode (color or action)")
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    
    # Handle CTRL+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    gui = VisionDebugGui(start_mode=args.mode)
    gui.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
