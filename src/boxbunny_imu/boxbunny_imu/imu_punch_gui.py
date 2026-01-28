import json
import os
import queue
import sys
import time
import site
import math
import subprocess
from typing import Optional

try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception as exc:
    raise SystemExit(
        f"PySide6 not available: {exc}\nInstall with: python3 -m pip install --user PySide6"
    ) from exc

import rclpy
from rclpy.node import Node
from boxbunny_msgs.msg import ImuDebug, ImuPunch
from boxbunny_msgs.srv import CalibrateImuPunch


PUNCH_TYPE_CHOICES = [
    ("Straight", "straight"),
    ("Hook", "hook"),
    ("Uppercut", "uppercut"),
]

APP_STYLESHEET = """
QWidget { background-color: #111317; color: #E6E6E6; font-family: 'DejaVu Sans'; }
QGroupBox { border: 1px solid #2A2E36; border-radius: 8px; margin-top: 8px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #C0C4CC; }
QPushButton { background-color: #2B3240; border: 1px solid #394151; padding: 6px 10px; border-radius: 6px; }
QPushButton:hover { background-color: #394151; }
QPushButton:pressed { background-color: #202633; }
QLineEdit, QComboBox { background-color: #1A1E25; border: 1px solid #2A2E36; padding: 4px 6px; border-radius: 6px; }
QLabel { color: #E6E6E6; }
"""


def _apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyleSheet(APP_STYLESHEET)


class ImuGuiNode(Node):
    def __init__(self, imu_signal, punch_signal, status_signal, requests) -> None:
        super().__init__("imu_punch_gui")
        self._imu_signal = imu_signal
        self._punch_signal = punch_signal
        self._status_signal = status_signal
        self._requests: queue.Queue = requests

        self.create_subscription(ImuDebug, "imu/debug", self._on_imu_debug, 10)
        self.create_subscription(ImuPunch, "imu/punch", self._on_punch, 10)
        self._calib_client = self.create_client(CalibrateImuPunch, "calibrate_imu_punch")
        self.create_timer(0.1, self._poll_requests)

        self._status_signal.emit("Waiting for IMU data...")

    def _on_imu_debug(self, msg: ImuDebug) -> None:
        self._imu_signal.emit(msg)

    def _on_punch(self, msg: ImuPunch) -> None:
        self._punch_signal.emit(msg)

    def _poll_requests(self) -> None:
        while True:
            try:
                punch_type, duration_s = self._requests.get_nowait()
            except queue.Empty:
                return
            if not self._calib_client.service_is_ready():
                self._status_signal.emit("Calibration service not available. Is imu_punch_classifier running?")
                continue
            req = CalibrateImuPunch.Request()
            req.punch_type = punch_type
            req.duration_s = float(duration_s)
            future = self._calib_client.call_async(req)
            future.add_done_callback(self._on_calibration_done)

    def _on_calibration_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._status_signal.emit(f"Calibration failed: {exc}")
            return
        if not result.accepted:
            self._status_signal.emit(f"Calibration rejected: {result.message}")
            return
        self._status_signal.emit(result.message)


class RosWorker(QtCore.QThread):
    imu = QtCore.Signal(object)
    punch = QtCore.Signal(object)
    status = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._requests: queue.Queue = queue.Queue()

    def run(self) -> None:
        try:
            rclpy.init()
            node = ImuGuiNode(self.imu, self.punch, self.status, self._requests)
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            node.destroy_node()
        except Exception as e:
            print(f"ROS Worker error: {e}")
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass  # Already shutdown or never initialized

    def request_calibration(self, punch_type: str, duration: float) -> None:
        self._requests.put((punch_type, duration))

    def set_calibration_path(self, path: str) -> None:
        self.set_parameter_value("calibration_path", path)

    def set_parameter_value(self, name: str, value: str, node: str = "/imu_punch_classifier") -> None:
        # Use subprocess for simplicity in this threaded worker context
        subprocess.Popen(
            ["ros2", "param", "set", node, name, str(value)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def get_parameters(self, names: list[str], node: str = "/imu_punch_classifier") -> dict:
        # Synchronous call via subprocess to get current params
        # Handle case where node isn't ready yet
        results = {}
        for name in names:
            try:
                out = subprocess.check_output(
                    ["ros2", "param", "get", node, name],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,  # Suppress "Node not found" errors
                    timeout=2.0
                )
                # Output format e.g.: "Boolean value is: true" or "Double value is: 0.5"
                # We just need to parse the value after the last colon
                val_str = out.strip().split(":")[-1].strip()
                
                # Try float
                try:
                    results[name] = float(val_str)
                except ValueError:
                    # Try boolean
                    if val_str.lower() == "true":
                        results[name] = True
                    elif val_str.lower() == "false":
                        results[name] = False
                    else:
                        results[name] = val_str
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Node not ready or param doesn't exist - silently skip
                pass
            except Exception:
                pass
        return results


class ImuPunchGui(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sensor Calibration & Config")
        self.resize(1100, 650)

        self.last_imu: Optional[ImuDebug] = None
        self.last_punch: Optional[ImuPunch] = None
        self._last_punch_time: Optional[float] = None
        self._calib_end: Optional[float] = None
        self._imu_proc: Optional[subprocess.Popen] = None

        self._calib_queue: list[str] = []
        self._calib_count = 0
        self._updating_ui = False

        self.ros = RosWorker()
        self.ros.imu.connect(self._on_imu)
        self.ros.punch.connect(self._on_punch)
        self.ros.status.connect(self._on_status)
        self.ros.start()

        self._build_ui()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout()

        # Left Panel: Status + 3D View + Calibration
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        title = QtWidgets.QLabel("Sensor Calibration Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(title)

        self.status_label = QtWidgets.QLabel("Status: --")
        left_layout.addWidget(self.status_label)
        
        # Profile Selector
        profile_layout = QtWidgets.QHBoxLayout()
        profile_layout.addWidget(QtWidgets.QLabel("Profile:"))
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItem("Straights Only", os.path.expanduser("~/.boxbunny/calibration_straights.json"))
        self.profile_combo.addItem("All Punches", os.path.expanduser("~/.boxbunny/calibration_all.json"))
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        profile_layout.addWidget(self.profile_combo)
        left_layout.addLayout(profile_layout)

        grid = QtWidgets.QGridLayout()
        self.axis_view = ImuAxisWidget()
        self.imu_label = QtWidgets.QLabel("IMU: --")
        self.mag_label = QtWidgets.QLabel("Magnitudes: Accel=-- | Gyro=-- (compared to thresholds)")
        self.mag_label.setStyleSheet("font-weight: bold; color: #00aaff;")
        self.direction_label = QtWidgets.QLabel("Direction: --")
        self.punch_label = QtWidgets.QLabel("Last punch: --")
        self.confidence_label = QtWidgets.QLabel("Confidence: --")
        grid.addWidget(self.axis_view, 0, 0, 5, 1)
        grid.addWidget(self.imu_label, 0, 1)
        grid.addWidget(self.mag_label, 1, 1)
        grid.addWidget(self.direction_label, 2, 1)
        grid.addWidget(self.punch_label, 3, 1)
        grid.addWidget(self.confidence_label, 4, 1)
        left_layout.addLayout(grid)

        calib_group = QtWidgets.QGroupBox("Calibration")
        calib_layout = QtWidgets.QHBoxLayout()
        
        self.calib_default_btn = QtWidgets.QPushButton("Calibrate Default (Jab/Cross)")
        self.calib_default_btn.clicked.connect(lambda: self._start_calibration_sequence("default"))
        
        self.calib_all_btn = QtWidgets.QPushButton("Calibrate All (Cycle)")
        self.calib_all_btn.clicked.connect(lambda: self._start_calibration_sequence("all"))

        self.calib_verify_btn = QtWidgets.QPushButton("Verify File")
        self.calib_verify_btn.clicked.connect(self._verify_calibration)

        self.calib_save_btn = QtWidgets.QPushButton("Save Calibration")
        self.calib_save_btn.clicked.connect(self._force_save_calibration)

        self.calib_reset_btn = QtWidgets.QPushButton("Reset")
        self.calib_reset_btn.clicked.connect(self._reset_calibration)

        calib_layout.addWidget(self.calib_default_btn)
        calib_layout.addWidget(self.calib_all_btn)
        calib_layout.addWidget(self.calib_verify_btn)
        calib_layout.addWidget(self.calib_save_btn)
        calib_layout.addWidget(self.calib_reset_btn)
        calib_layout.addStretch(1)
        calib_group.setLayout(calib_layout)
        left_layout.addWidget(calib_group)

        self.help_label = QtWidgets.QLabel(
            "Tip: Click Calibrate. Wait for prompt. Punch. Repeat when prompted."
        )
        self.help_label.setStyleSheet("color: #666;")
        left_layout.addWidget(self.help_label)
        left_layout.addStretch(1)
        
        # Right Panel: Testing Log
        right_panel = QtWidgets.QGroupBox("Punch Log (Testing)")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setStyleSheet("font-family: monospace;")
        clear_btn = QtWidgets.QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_list.clear)
        right_layout.addWidget(self.log_list)
        right_layout.addWidget(clear_btn)
        
        # Sensitivity settings
        sens_group = self._build_sensitivity_ui()
        right_layout.insertWidget(0, sens_group)
        
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)
        self.setLayout(layout)
        
        # Trigger initial profile load
        QtCore.QTimer.singleShot(1000, self._on_profile_changed)



    def _build_sensitivity_ui(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("IMU Sensitivity Settings")
        layout = QtWidgets.QFormLayout()
        
        self.accel_spin = QtWidgets.QDoubleSpinBox()
        self.accel_spin.setRange(1.0, 50.0)
        self.accel_spin.setSingleStep(0.5)
        self.accel_spin.setValue(12.0)
        self.accel_spin.valueChanged.connect(lambda v: self._on_sensitivity_changed("accel_threshold", v))
        
        self.gyro_spin = QtWidgets.QDoubleSpinBox()
        self.gyro_spin.setRange(0.1, 20.0)
        self.gyro_spin.setSingleStep(0.1)
        self.gyro_spin.setValue(0.5)
        self.gyro_spin.valueChanged.connect(lambda v: self._on_sensitivity_changed("gyro_threshold", v))

        layout.addRow("Accel Threshold:", self.accel_spin)
        layout.addRow("Gyro Threshold:", self.gyro_spin)
        group.setLayout(layout)
        return group

    def _on_sensitivity_changed(self, name: str, value: float) -> None:
        if self._updating_ui:
            return
        self.ros.set_parameter_value(name, str(value))

    def _on_profile_changed(self) -> None:
        path = self.profile_combo.currentData()
        name = self.profile_combo.currentText()
        self.ros.set_calibration_path(path)
        self.status_label.setText(f"Status: Switched to profile '{name}'")
        
        # Sync settings after a short delay to allow backend to load file
        QtCore.QTimer.singleShot(500, self._sync_settings)
        
    def _sync_settings(self) -> None:
        self._updating_ui = True
        try:
            # Sync IMU Params
            params = self.ros.get_parameters(["accel_threshold", "gyro_threshold"])
            if "accel_threshold" in params:
                self.accel_spin.setValue(params["accel_threshold"])
            if "gyro_threshold" in params:
                self.gyro_spin.setValue(params["gyro_threshold"])
        finally:
            self._updating_ui = False

    def _verify_calibration(self) -> None:
        path = self.profile_combo.currentData()
        ts = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{ts}] -- VERIFYING '{os.path.basename(path)}' --")
        
        if not os.path.exists(path):
            self.log_list.addItem("  FILE NOT FOUND (Needs calibration)")
            self.log_list.scrollToBottom()
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if not data:
                self.log_list.addItem("  FILE EMPTY")
            
            punch_count = 0
            for key, val in data.items():
                # Skip settings, only show punch types
                if key == "settings":
                    continue
                if isinstance(val, dict) and ("peak_accel" in val or "peak_gyro" in val):
                    accel = val.get("peak_accel", 0.0)
                    gyro = val.get("peak_gyro", 0.0)
                    self.log_list.addItem(f"  {key.upper()}: Accel={accel:.2f}, Gyro={gyro:.2f}")
                    punch_count += 1
            
            if punch_count == 0:
                self.log_list.addItem("  No punch calibrations found")
            else:
                self.log_list.addItem(f"  Loaded {punch_count} punch type(s)")
                self.log_list.addItem("  TEST MODE: Punch now and watch for detection...")
                
        except Exception as e:
            self.log_list.addItem(f"  ERROR: {e}")
            
        self.log_list.scrollToBottom()

    def _on_imu(self, msg: ImuDebug) -> None:
        self.last_imu = msg

    def _on_punch(self, msg: ImuPunch) -> None:
        self.last_punch = msg
        self._last_punch_time = time.time()
        
        # Add to log
        ts = time.strftime("%H:%M:%S")
        method = f" [{msg.method}]" if msg.method == "calibration_complete" else ""
        entry = f"[{ts}] {msg.punch_type.upper():<10} Conf: {msg.confidence:.2f}{method}"
        item = QtWidgets.QListWidgetItem(entry)
        
        if msg.method == "calibration_complete":
            item.setForeground(QtGui.QColor("#4DFF88")) # Green for calibration
        elif msg.confidence < 0.5:
             item.setForeground(QtGui.QColor("#FF6B6B")) # Red for low confidence
             
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()
        
        if msg.method == "calibration_complete":
            self._handle_calibration_step_complete()

    def _on_status(self, message: str) -> None:
        if not self._calib_queue: # Only update status from worker if not in sequence logic
            self.status_label.setText(f"Status: {message}")

    def _start_calibration_sequence(self, mode: str) -> None:
        if mode == "default":
            self._calib_queue = ["straight"]
        elif mode == "all":
            self._calib_queue = ["straight", "hook", "uppercut"]
        else:
            return

        self._calib_count = 0
        self.calib_default_btn.setEnabled(False)
        self.calib_all_btn.setEnabled(False)
        self._trigger_next_calibration_step()

    def _trigger_next_calibration_step(self) -> None:
        if not self._calib_queue:
            self._finish_calibration()
            return
            
        punch_type = self._calib_queue[0]
        self.status_label.setText(f"Status: Waiting for {punch_type} ({self._calib_count + 1}/3)... PUNCH NOW!")
        self.ros.request_calibration(punch_type, -1.0) # -1.0 = wait for trigger

    def _handle_calibration_step_complete(self) -> None:
        if not self._calib_queue:
            return

        self._calib_count += 1
        
        if self._calib_count < 3:
            self._trigger_next_calibration_step()
        else:
            # Done with this type
            self._calib_queue.pop(0)
            self._calib_count = 0
            if self._calib_queue:
                self._trigger_next_calibration_step()
            else:
                self._finish_calibration()

    def _finish_calibration(self) -> None:
        self.status_label.setText("Status: Calibration sequence finished!")
        self.calib_default_btn.setEnabled(True)
        self.calib_all_btn.setEnabled(True)
        self._calib_queue = []

    def _force_save_calibration(self) -> None:
        """Force save current calibration to file by triggering a ROS service."""
        # Use subprocess to call ros2 service that triggers save
        profile = self.profile_combo.currentText()
        if profile == "Straights Only":
            path = os.path.expanduser("~/.boxbunny/calibration_straights.json")
        else:
            path = os.path.expanduser("~/.boxbunny/imu_calibration.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a minimal calibration file if none exists
        if not os.path.exists(path):
            import json
            default_data = {
                "straight": {"peak_accel": 15.0, "peak_gyro": 2.0},
                "settings": {"accel_threshold": 12.5, "gyro_threshold": 2.5}
            }
            with open(path, "w") as f:
                json.dump(default_data, f, indent=2)
            self.status_label.setText(f"Status: Created default calibration at {path}")
        else:
            self.status_label.setText(f"Status: Calibration file exists at {path}")
        
        self.log_list.addItem(f"Save check: {path}")
        self.log_list.scrollToBottom()

    def _reset_calibration(self) -> None:
        """Reset the current calibration sequence."""
        self._calib_queue = []
        self._calib_count = 0
        self.calib_default_btn.setEnabled(True)
        self.calib_all_btn.setEnabled(True)
        self.status_label.setText("Status: Calibration reset. Ready.")
        self.log_list.addItem("-- CALIBRATION RESET --")
        self.log_list.scrollToBottom()

    def _direction_from_imu(self, imu: ImuDebug) -> str:
        ax, ay, az = imu.ax, imu.ay, imu.az
        mags = [abs(ax), abs(ay), abs(az)]
        max_mag = max(mags)
        if max_mag < 1.5:
            return "idle"
        axis = mags.index(max_mag)
        if axis == 0:
            return "right" if ax > 0 else "left"
        if axis == 2:
            return "up" if az > 0 else "down"
        return "straight" if ay > 0 else "back"

    def _refresh(self) -> None:
        if self.last_imu is not None:
            imu = self.last_imu
            self.imu_label.setText(
                f"IMU ax={imu.ax:.2f} ay={imu.ay:.2f} az={imu.az:.2f} | gx={imu.gx:.2f} gy={imu.gy:.2f} gz={imu.gz:.2f}"
            )
            # Show motion detection values (ay is gravity-corrected: ay_motion = ay + 9.8)
            ay_motion = abs(imu.ay + 9.8)  # Gravity on Y-axis
            self.mag_label.setText(f"Motion: ay={ay_motion:.2f} | peak_g={max(abs(imu.gx), abs(imu.gy), abs(imu.gz)):.2f}")
            self.direction_label.setText(f"Direction: {self._direction_from_imu(imu)}")
            self.axis_view.set_vector(imu.ax, imu.ay, imu.az)
        if self.last_punch is not None:
            punch = self.last_punch
            method_str = f" ({punch.method})" if punch.method else ""
            self.punch_label.setText(f"Last punch: {punch.punch_type or 'unknown'}{method_str}")
            self.confidence_label.setText(
                f"Confidence: {punch.confidence:.2f} (accel={punch.peak_accel:.2f}, gyro={punch.peak_gyro:.2f})"
            )

    def closeEvent(self, event) -> None:
        if self.ros.isRunning():
            rclpy.shutdown()
            self.ros.wait(1000)
        if self._imu_proc is not None:
            try:
                self._imu_proc.terminate()
            except Exception:
                pass
        event.accept()

    def start_imu_launch(self) -> None:
        if self._imu_proc is not None:
            return
        self._imu_proc = subprocess.Popen(
            ["ros2", "launch", "boxbunny_bringup", "imu_only.launch.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_theme(app)
    gui = ImuPunchGui()
    if os.environ.get("BOXBUNNY_IMU_AUTO_LAUNCH", "1") != "0":
        gui.start_imu_launch()
    gui.show()
    sys.exit(app.exec())


class ImuAxisWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._vec = (0.0, 0.0, 0.0)
        self.setMinimumSize(240, 200)

    def set_vector(self, ax: float, ay: float, az: float) -> None:
        self._vec = (ax, ay, az)
        self.update()

    def _project(self, x: float, y: float, z: float) -> QtCore.QPointF:
        # Simple isometric projection for a pseudo-3D view.
        angle = math.radians(30)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        px = (x - z) * cos_a
        py = y + (x + z) * sin_a
        return QtCore.QPointF(px, py)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        center = QtCore.QPointF(rect.width() * 0.5, rect.height() * 0.6)
        scale = min(rect.width(), rect.height()) * 0.35

        # Axes
        axes = [(1, 0, 0, QtGui.QColor("#FF6B6B")),
                (0, 1, 0, QtGui.QColor("#4DFF88")),
                (0, 0, 1, QtGui.QColor("#6BA8FF"))]
        for x, y, z, color in axes:
            end = self._project(x, y, z)
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(center, center + QtCore.QPointF(end.x() * scale, -end.y() * scale))

        # Vector (normalized)
        ax, ay, az = self._vec
        mag = max(1e-4, (ax * ax + ay * ay + az * az) ** 0.5)
        nx, ny, nz = ax / mag, ay / mag, az / mag
        end = self._project(nx, ny, nz)
        painter.setPen(QtGui.QPen(QtGui.QColor("#F5C542"), 3))
        painter.drawLine(center, center + QtCore.QPointF(end.x() * scale, -end.y() * scale))


if __name__ == "__main__":
    main()
