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
        rclpy.init()
        node = ImuGuiNode(self.imu, self.punch, self.status, self._requests)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()

    def request_calibration(self, punch_type: str, duration_s: float) -> None:
        self._requests.put((punch_type, duration_s))


class ImuPunchGui(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IMU Punch Calibration")
        self.resize(860, 520)

        self.last_imu: Optional[ImuDebug] = None
        self.last_punch: Optional[ImuPunch] = None
        self._last_punch_time: Optional[float] = None
        self._calib_end: Optional[float] = None
        self._imu_proc: Optional[subprocess.Popen] = None

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
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("IMU Punch Viewer + Calibration")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.status_label = QtWidgets.QLabel("Status: --")
        layout.addWidget(self.status_label)

        grid = QtWidgets.QGridLayout()
        self.axis_view = ImuAxisWidget()
        self.imu_label = QtWidgets.QLabel("IMU: --")
        self.direction_label = QtWidgets.QLabel("Direction: --")
        self.punch_label = QtWidgets.QLabel("Last punch: --")
        self.confidence_label = QtWidgets.QLabel("Confidence: --")
        grid.addWidget(self.axis_view, 0, 0, 4, 1)
        grid.addWidget(self.imu_label, 0, 1)
        grid.addWidget(self.direction_label, 1, 1)
        grid.addWidget(self.punch_label, 2, 1)
        grid.addWidget(self.confidence_label, 3, 1)
        layout.addLayout(grid)

        calib_group = QtWidgets.QGroupBox("Calibration")
        calib_layout = QtWidgets.QHBoxLayout()
        self.punch_type_box = QtWidgets.QComboBox()
        for label, value in PUNCH_TYPE_CHOICES:
            self.punch_type_box.addItem(label, value)
        self.calib_btn = QtWidgets.QPushButton("Calibrate (3 hits)")
        self.calib_btn.clicked.connect(self._start_calibration)
        calib_layout.addWidget(QtWidgets.QLabel("Punch type:"))
        calib_layout.addWidget(self.punch_type_box)
        calib_layout.addStretch(1)
        calib_layout.addWidget(self.calib_btn)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        self.help_label = QtWidgets.QLabel(
            "Tip: Click Calibrate and hit the pad 3 times during the countdown."
        )
        self.help_label.setStyleSheet("color: #666;")
        layout.addWidget(self.help_label)

        layout.addStretch(1)
        self.setLayout(layout)

    def _on_imu(self, msg: ImuDebug) -> None:
        self.last_imu = msg

    def _on_punch(self, msg: ImuPunch) -> None:
        self.last_punch = msg
        self._last_punch_time = time.time()

    def _on_status(self, message: str) -> None:
        self.status_label.setText(f"Status: {message}")

    def _start_calibration(self) -> None:
        punch_type = str(self.punch_type_box.currentData())
        duration_s = 3.5
        self._calib_end = time.time() + duration_s
        self.status_label.setText("Status: Get ready... punch 3 times now")
        self.ros.request_calibration(punch_type, duration_s)
        self.calib_btn.setEnabled(False)

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
            self.direction_label.setText(f"Direction: {self._direction_from_imu(imu)}")
            self.axis_view.set_vector(imu.ax, imu.ay, imu.az)
        if self.last_punch is not None:
            punch = self.last_punch
            self.punch_label.setText(f"Last punch: {punch.punch_type or 'unknown'}")
            self.confidence_label.setText(
                f"Confidence: {punch.confidence:.2f} (accel={punch.peak_accel:.2f}, gyro={punch.peak_gyro:.2f})"
            )
        if self._calib_end is not None:
            remaining = self._calib_end - time.time()
            if remaining <= 0:
                self._calib_end = None
                self.calib_btn.setEnabled(True)
                self.status_label.setText("Status: Calibration finished. Saved to calibration file.")
            else:
                self.status_label.setText(f"Status: Calibrating... {remaining:.1f}s left (hit 3 times)")

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
