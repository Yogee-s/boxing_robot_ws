import json
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from rclpy.parameter_client import AsyncParametersClient
from rclpy.parameter import Parameter
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from boxbunny_msgs.msg import GloveDetections, PunchEvent, ImuDebug, TrashTalk
from boxbunny_msgs.srv import StartStopDrill, GenerateLLM

from PySide6 import QtCore, QtGui, QtWidgets


class RosInterface(Node):
    def __init__(self) -> None:
        super().__init__("boxbunny_gui")
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.declare_parameter("punch_topic", "punch_events")
        self.declare_parameter("color_topic", "/camera/color/image_raw")

        self.last_image = None
        self.last_color_image = None
        self.last_detections: Optional[GloveDetections] = None
        self.last_punch: Optional[PunchEvent] = None
        self.last_imu: Optional[ImuDebug] = None
        self.drill_state = "idle"
        self.drill_summary = {}
        self.drill_countdown = 0
        self.trash_talk = ""
        self.last_punch_stamp = None
        self.punch_counter = 0

        punch_topic = self.get_parameter("punch_topic").value
        color_topic = self.get_parameter("color_topic").value

        self.debug_sub = self.create_subscription(Image, "glove_debug_image", self._on_image, 5)
        self.color_sub = self.create_subscription(Image, color_topic, self._on_color_image, 5)
        self.det_sub = self.create_subscription(GloveDetections, "glove_detections", self._on_detections, 5)
        self.punch_sub = self.create_subscription(PunchEvent, punch_topic, self._on_punch, 5)
        self.imu_sub = self.create_subscription(ImuDebug, "imu/debug", self._on_imu, 5)
        self.state_sub = self.create_subscription(String, "drill_state", self._on_state, 5)
        self.summary_sub = self.create_subscription(String, "drill_summary", self._on_summary, 5)
        self.countdown_sub = self.create_subscription(Int32, "drill_countdown", self._on_countdown, 5)
        self.trash_sub = self.create_subscription(TrashTalk, "trash_talk", self._on_trash, 5)

        self.start_stop_client = self.create_client(StartStopDrill, "start_stop_drill")
        self.llm_client = self.create_client(GenerateLLM, "llm/generate")
        self.tracker_param_client = AsyncParametersClient(self, "realsense_glove_tracker")
        self.drill_param_client = AsyncParametersClient(self, "reaction_drill_manager")

    def _on_image(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.last_image = img
        except Exception:
            pass

    def _on_color_image(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.last_color_image = img
        except Exception:
            pass

    def _on_detections(self, msg: GloveDetections) -> None:
        with self.lock:
            self.last_detections = msg

    def _on_punch(self, msg: PunchEvent) -> None:
        with self.lock:
            self.last_punch = msg
            self.last_punch_stamp = (msg.stamp.sec, msg.stamp.nanosec)
            self.punch_counter += 1

    def _on_imu(self, msg: ImuDebug) -> None:
        with self.lock:
            self.last_imu = msg

    def _on_state(self, msg: String) -> None:
        with self.lock:
            self.drill_state = msg.data

    def _on_summary(self, msg: String) -> None:
        try:
            with self.lock:
                self.drill_summary = json.loads(msg.data)
        except Exception:
            pass

    def _on_countdown(self, msg: Int32) -> None:
        with self.lock:
            self.drill_countdown = int(msg.data)

    def _on_trash(self, msg: TrashTalk) -> None:
        with self.lock:
            self.trash_talk = msg.text


class RosSpinThread(QtCore.QThread):
    def __init__(self, node: RosInterface) -> None:
        super().__init__()
        self.node = node

    def run(self) -> None:
        rclpy.spin(self.node)


class BoxBunnyGui(QtWidgets.QMainWindow):
    def __init__(self, ros: RosInterface) -> None:
        super().__init__()
        self.ros = ros
        self.setWindowTitle("BoxBunny Trainer")
        self.resize(1024, 720)

        self._frame_buffer = deque(maxlen=180)
        self._last_punch_counter = 0
        self._replay_frames = []
        self._replay_index = 0

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        self.reaction_tab = QtWidgets.QWidget()
        self._setup_reaction_tab()
        tabs.addTab(self.reaction_tab, "Reaction Drill")

        self.punch_tab = QtWidgets.QWidget()
        self._setup_punch_tab()
        tabs.addTab(self.punch_tab, "Punch Detection")

        self.calib_tab = QtWidgets.QWidget()
        self._setup_calibration_tab()
        tabs.addTab(self.calib_tab, "Calibration")

        self.llm_tab = QtWidgets.QWidget()
        self._setup_llm_tab()
        tabs.addTab(self.llm_tab, "LLM")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(50)

        self.replay_timer = QtCore.QTimer()
        self.replay_timer.timeout.connect(self._play_replay)

    def _setup_reaction_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()

        self.cue_panel = QtWidgets.QFrame()
        self.cue_panel.setFixedHeight(180)
        self.cue_panel.setStyleSheet("background-color: #444;")

        self.state_label = QtWidgets.QLabel("State: idle")
        self.countdown_label = QtWidgets.QLabel("Countdown: --")

        self.start_btn = QtWidgets.QPushButton("Start Reaction Test")
        self.stop_btn = QtWidgets.QPushButton("Stop Drill")
        self.start_btn.clicked.connect(self._start_drill)
        self.stop_btn.clicked.connect(self._stop_drill)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)

        self.last_reaction_label = QtWidgets.QLabel("Last reaction: --")
        self.summary_label = QtWidgets.QLabel("Summary: --")
        self.trash_label = QtWidgets.QLabel("Coach: --")

        self.reaction_preview = QtWidgets.QLabel()
        self.reaction_preview.setFixedSize(360, 240)
        self.reaction_preview.setStyleSheet("background-color: #222;")

        replay_row = QtWidgets.QHBoxLayout()
        self.replay_btn = QtWidgets.QPushButton("Replay Last")
        self.replay_btn.clicked.connect(self._start_replay)
        self.replay_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.replay_speed.setRange(5, 30)
        self.replay_speed.setValue(12)
        replay_row.addWidget(self.replay_btn)
        replay_row.addWidget(QtWidgets.QLabel("Slow-mo FPS"))
        replay_row.addWidget(self.replay_speed)

        layout.addWidget(self.cue_panel)
        layout.addWidget(self.state_label)
        layout.addWidget(self.countdown_label)
        layout.addLayout(button_row)
        layout.addWidget(self.last_reaction_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.trash_label)
        layout.addWidget(self.reaction_preview)
        layout.addLayout(replay_row)

        self.reaction_tab.setLayout(layout)

    def _setup_punch_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout()

        self.punch_preview = QtWidgets.QLabel()
        self.punch_preview.setFixedSize(720, 480)
        self.punch_preview.setStyleSheet("background-color: #222;")

        imu_layout = QtWidgets.QVBoxLayout()
        self.imu_label = QtWidgets.QLabel("IMU: --")
        self.punch_label = QtWidgets.QLabel("Last punch: --")
        imu_layout.addWidget(self.imu_label)
        imu_layout.addWidget(self.punch_label)
        imu_layout.addStretch(1)

        layout.addWidget(self.punch_preview)
        layout.addLayout(imu_layout)
        self.punch_tab.setLayout(layout)

    def _setup_calibration_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.calib_status = QtWidgets.QLabel("Adjust HSV ranges and Apply")

        self.green_sliders = self._create_hsv_group("Green (Left)")
        self.red1_sliders = self._create_hsv_group("Red (Right) - Range 1")
        self.red2_sliders = self._create_hsv_group("Red (Right) - Range 2")

        self.apply_btn = QtWidgets.QPushButton("Apply to Tracker")
        self.save_btn = QtWidgets.QPushButton("Save to YAML")
        self.apply_btn.clicked.connect(self._apply_hsv)
        self.save_btn.clicked.connect(self._save_yaml)

        layout.addWidget(self.calib_status)
        layout.addWidget(self.green_sliders["group"])
        layout.addWidget(self.red1_sliders["group"])
        layout.addWidget(self.red2_sliders["group"])
        layout.addWidget(self.apply_btn)
        layout.addWidget(self.save_btn)

        self.calib_tab.setLayout(layout)

    def _setup_llm_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.llm_prompt = QtWidgets.QLineEdit()
        self.llm_prompt.setPlaceholderText("Type a prompt for the coach...")
        self.llm_mode = QtWidgets.QComboBox()
        self.llm_mode.addItems(["coach", "encourage", "trash", "analysis"])
        self.llm_send = QtWidgets.QPushButton("Send")
        self.llm_send.clicked.connect(self._send_llm_prompt)
        self.llm_response = QtWidgets.QTextEdit()
        self.llm_response.setReadOnly(True)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.llm_prompt)
        row.addWidget(self.llm_mode)
        row.addWidget(self.llm_send)

        layout.addLayout(row)
        layout.addWidget(self.llm_response)
        self.llm_tab.setLayout(layout)

    def _create_hsv_group(self, title: str):
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout()
        labels = ["H", "S", "V"]
        sliders_low = []
        sliders_high = []

        for i, label in enumerate(labels):
            low = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            high = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            low.setRange(0, 255)
            high.setRange(0, 255)
            low.setValue(0)
            high.setValue(255)
            grid.addWidget(QtWidgets.QLabel(f"{label} Low"), i, 0)
            grid.addWidget(low, i, 1)
            grid.addWidget(QtWidgets.QLabel(f"{label} High"), i, 2)
            grid.addWidget(high, i, 3)
            sliders_low.append(low)
            sliders_high.append(high)

        group.setLayout(grid)
        return {"group": group, "low": sliders_low, "high": sliders_high}

    def _apply_hsv(self) -> None:
        params = [
            ("hsv_green_lower", self._slider_values(self.green_sliders["low"])),
            ("hsv_green_upper", self._slider_values(self.green_sliders["high"])),
            ("hsv_red_lower1", self._slider_values(self.red1_sliders["low"])),
            ("hsv_red_upper1", self._slider_values(self.red1_sliders["high"])),
            ("hsv_red_lower2", self._slider_values(self.red2_sliders["low"])),
            ("hsv_red_upper2", self._slider_values(self.red2_sliders["high"])),
        ]

        ros_params = [Parameter(name, Parameter.Type.INTEGER_ARRAY, value) for name, value in params]
        future = self.ros.tracker_param_client.set_parameters(ros_params)
        future.add_done_callback(lambda _: None)
        self.calib_status.setText("Applied HSV parameters to tracker")

    def _save_yaml(self) -> None:
        import yaml

        config = {
            "realsense_glove_tracker": {
                "ros__parameters": {
                    "hsv_green_lower": self._slider_values(self.green_sliders["low"]),
                    "hsv_green_upper": self._slider_values(self.green_sliders["high"]),
                    "hsv_red_lower1": self._slider_values(self.red1_sliders["low"]),
                    "hsv_red_upper1": self._slider_values(self.red1_sliders["high"]),
                    "hsv_red_lower2": self._slider_values(self.red2_sliders["low"]),
                    "hsv_red_upper2": self._slider_values(self.red2_sliders["high"]),
                }
            }
        }
        path = os.path.expanduser("~/boxbunny_hsv.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        self.calib_status.setText(f"Saved YAML to {path}")

    def _slider_values(self, sliders):
        return [int(s.value()) for s in sliders]

    def _start_drill(self) -> None:
        if not self.ros.start_stop_client.service_is_ready():
            self.ros.get_logger().warn("start_stop_drill service not ready")
            return
        req = StartStopDrill.Request()
        req.start = True
        req.num_trials = 5
        self.ros.start_stop_client.call_async(req)

    def _stop_drill(self) -> None:
        if not self.ros.start_stop_client.service_is_ready():
            return
        req = StartStopDrill.Request()
        req.start = False
        req.num_trials = 0
        self.ros.start_stop_client.call_async(req)

    def _send_llm_prompt(self) -> None:
        if not self.ros.llm_client.service_is_ready():
            self.llm_response.setPlainText("LLM service not ready")
            return
        prompt = self.llm_prompt.text().strip()
        if not prompt:
            return
        req = GenerateLLM.Request()
        req.prompt = prompt
        req.mode = self.llm_mode.currentText()
        req.context = "gui"
        future = self.ros.llm_client.call_async(req)
        future.add_done_callback(self._on_llm_response)

    def _on_llm_response(self, future) -> None:
        try:
            response = future.result()
            self.llm_response.setPlainText(response.response)
        except Exception as exc:
            self.llm_response.setPlainText(f"LLM error: {exc}")

    def _update_ui(self) -> None:
        with self.ros.lock:
            state = self.ros.drill_state
            summary = self.ros.drill_summary
            trash = self.ros.trash_talk
            imu = self.ros.last_imu
            punch = self.ros.last_punch
            img = self.ros.last_image
            color_img = self.ros.last_color_image
            countdown = self.ros.drill_countdown
            punch_counter = self.ros.punch_counter

        if state == "cue":
            self.cue_panel.setStyleSheet("background-color: #2ecc71;")
        elif state in {"waiting", "countdown", "baseline"}:
            self.cue_panel.setStyleSheet("background-color: #f1c40f;")
        else:
            self.cue_panel.setStyleSheet("background-color: #444;")

        self.state_label.setText(f"State: {state}")
        if state == "countdown":
            self.countdown_label.setText(f"Countdown: {countdown}")
        elif state == "baseline":
            self.countdown_label.setText("Countdown: baseline capture")
        else:
            self.countdown_label.setText("Countdown: --")

        last_rt = summary.get("last_reaction_time_s") if isinstance(summary, dict) else None
        mean_rt = summary.get("mean_reaction_time_s") if isinstance(summary, dict) else None
        best_rt = summary.get("best_reaction_time_s") if isinstance(summary, dict) else None
        baseline_v = summary.get("baseline_velocity_mps") if isinstance(summary, dict) else None
        self.last_reaction_label.setText(f"Last reaction: {last_rt if last_rt is not None else '--'}")
        self.summary_label.setText(
            f\"Summary: mean {mean_rt if mean_rt is not None else '--'} | best {best_rt if best_rt is not None else '--'} | baseline v {baseline_v if baseline_v is not None else '--'}\"
        )
        self.trash_label.setText(f"Coach: {trash or '--'}")

        if imu:
            self.imu_label.setText(
                f"IMU ax={imu.ax:.2f} ay={imu.ay:.2f} az={imu.az:.2f} | gx={imu.gx:.2f} gy={imu.gy:.2f} gz={imu.gz:.2f}"
            )
        if punch:
            self.punch_label.setText(
                f"Last punch: {punch.glove} {punch.punch_type or 'unknown'} v={punch.approach_velocity_mps:.2f} d={punch.distance_m:.2f}"
            )

        if img is not None:
            qimg = self._to_qimage(img)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.punch_preview.setPixmap(pix.scaled(self.punch_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        if color_img is not None:
            now = time.time()
            self._frame_buffer.append((now, color_img.copy()))
            qimg2 = self._to_qimage(color_img)
            pix2 = QtGui.QPixmap.fromImage(qimg2)
            self.reaction_preview.setPixmap(
                pix2.scaled(self.reaction_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            )

        if punch_counter != self._last_punch_counter:
            self._last_punch_counter = punch_counter
            self._capture_replay_clip()

    def _capture_replay_clip(self) -> None:
        if not self._frame_buffer:
            return
        now = time.time()
        clip = [frame for ts, frame in self._frame_buffer if now - ts <= 2.0]
        self._replay_frames = clip
        self._replay_index = 0

    def _start_replay(self) -> None:
        if not self._replay_frames:
            return
        fps = max(5, int(self.replay_speed.value()))
        interval_ms = int(1000 / fps)
        self.replay_timer.start(interval_ms)

    def _play_replay(self) -> None:
        if self._replay_index >= len(self._replay_frames):
            self.replay_timer.stop()
            return
        frame = self._replay_frames[self._replay_index]
        self._replay_index += 1
        qimg = self._to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.reaction_preview.setPixmap(
            pix.scaled(self.reaction_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        )

    def _to_qimage(self, img):
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(rgb.data, w, h, QtGui.QImage.Format.Format_RGB888)


def main() -> None:
    rclpy.init()
    ros_node = RosInterface()

    app = QtWidgets.QApplication([])
    gui = BoxBunnyGui(ros_node)
    gui.show()

    ros_thread = RosSpinThread(ros_node)
    ros_thread.start()

    app.exec()

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
