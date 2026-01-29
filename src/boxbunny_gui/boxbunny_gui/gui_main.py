import json
import os
import threading
import time
from collections import deque
from typing import Optional, List

import cv2
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String, Int32, Bool, Float32
from std_srvs.srv import SetBool, Trigger
from std_srvs.srv import SetBool
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from boxbunny_msgs.msg import GloveDetections, PunchEvent, ImuDebug, TrashTalk, ActionPrediction, DrillProgress
from boxbunny_msgs.srv import StartStopDrill, GenerateLLM, StartDrill

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QUrl

# Optional multimedia imports (video replay feature)
try:
    from PySide6.QtMultimedia import QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False
    QMediaPlayer = None
    QVideoWidget = None


# ============================================================================
# BUTTON STYLES (Centralized for consistency)
# ============================================================================

class ButtonStyle:
    """Centralized button style management for consistent appearance."""

    @staticmethod
    def _create_style(font_size, padding, min_width, min_height, bg_color, 
                     hover_color, pressed_color, border_radius=12):
        """Internal helper to generate button stylesheet."""
        return f"""
            QPushButton {{
                font-size: {font_size}px;
                font-weight: 600;
                padding: {padding}px;
                min-width: {min_width}px;
                min-height: {min_height}px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {bg_color}, stop:1 {pressed_color});
                color: white;
                border: none;
                border-radius: {border_radius}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {hover_color}, stop:1 {bg_color});
            }}
            QPushButton:pressed {{
                background: {pressed_color};
            }}
        """

    # Numpad buttons - Large touch-friendly
    NUMPAD = _create_style.__func__(
        font_size=40, padding=30, min_width=100, min_height=80,
        bg_color="#2196F3", hover_color="#42A5F5", pressed_color="#1565C0",
        border_radius=16
    )

    # Start button - Teal accent
    START = _create_style.__func__(
        font_size=18, padding=16, min_width=150, min_height=50,
        bg_color="#26d0ce", hover_color="#3ae0de", pressed_color="#1a7f7e",
    )

    # Large countdown style
    COUNTDOWN_LABEL = """
        QLabel {
            font-size: 120px;
            font-weight: bold;
            color: #26d0ce;
            background: transparent;
            border: none;
        }
    """


# ============================================================================
# CHECKBOX PROGRESS WIDGET
# ============================================================================

class CheckboxProgressWidget(QtWidgets.QWidget):
    """Visual progress tracker with checkbox indicators (☐ → ✓)."""
    
    def __init__(self, count: int = 3, parent=None):
        super().__init__(parent)
        self.count = count
        self.current = 0
        self.checkboxes = []
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(30)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        for i in range(count):
            checkbox = QtWidgets.QLabel("☐")
            checkbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            checkbox.setStyleSheet("""
                font-size: 64px;
                color: #484f58;
                min-width: 80px;
                background: transparent;
                border: none;
            """)
            layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)
    
    def tick(self, index: int = None):
        """Tick the checkbox at the given index (or next if None)."""
        if index is None:
            index = self.current
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].setText("✓")
            self.checkboxes[index].setStyleSheet("""
                font-size: 64px;
                color: #26d0ce;
                min-width: 80px;
                font-weight: bold;
                background: transparent;
                border: none;
            """)
            self.current = index + 1
    
    def reset(self):
        """Reset all checkboxes to empty."""
        self.current = 0
        for checkbox in self.checkboxes:
            checkbox.setText("☐")
            checkbox.setStyleSheet("""
                font-size: 64px;
                color: #484f58;
                min-width: 80px;
                background: transparent;
                border: none;
            """)


# ============================================================================
# COUNTDOWN SPLASH PAGE
# ============================================================================

class CountdownSplashPage(QtWidgets.QWidget):
    """Dedicated countdown splash screen before drills."""
    
    countdown_finished = QtCore.Signal()
    
    def __init__(self, title: str = "Get Ready!", parent=None):
        super().__init__(parent)
        self.countdown_value = 3
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_countdown)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Title
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: #e6edf3;
            background: transparent;
            border: none;
        """)
        
        # Large countdown number
        self.countdown_label = QtWidgets.QLabel("3")
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet(ButtonStyle.COUNTDOWN_LABEL)
        
        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 18px;
            color: #8b949e;
            background: transparent;
            border: none;
        """)
        
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
    
    def start(self, seconds: int = 3):
        """Start the countdown."""
        self.countdown_value = seconds
        self.countdown_label.setText(str(seconds))
        self.timer.start(1000)
    
    def _update_countdown(self):
        """Update countdown display."""
        if self.countdown_value > 1:
            self.countdown_value -= 1
            self.countdown_label.setText(str(self.countdown_value))
        else:
            self.timer.stop()
            self.countdown_label.setText("GO!")
            self.countdown_label.setStyleSheet("""
                font-size: 120px;
                font-weight: bold;
                color: #ff4757;
                background: transparent;
                border: none;
            """)
            # Brief delay then emit signal
            QtCore.QTimer.singleShot(500, self.countdown_finished.emit)
    
    def set_status(self, text: str):
        """Update the status label."""
        self.status_label.setText(text)


# ============================================================================
# NUMPAD WIDGET
# ============================================================================

class NumpadWidget(QtWidgets.QWidget):
    """Touch-friendly numpad (1-6) for quick selections."""
    
    button_pressed = QtCore.Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(15)
        
        # Create 2x3 grid of buttons (1-6)
        for i in range(6):
            btn = QtWidgets.QPushButton(str(i + 1))
            btn.setStyleSheet(ButtonStyle.NUMPAD)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, num=i+1: self.button_pressed.emit(num))
            row = i // 3
            col = i % 3
            layout.addWidget(btn, row, col)


# ============================================================================
# VIDEO REPLAY PAGE
# ============================================================================

class VideoReplayPage(QtWidgets.QWidget):
    """Video playback page for reviewing training sessions."""
    
    back_requested = QtCore.Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 40)
        layout.setSpacing(20)
        
        # Header row with back button and title
        header_row = QtWidgets.QHBoxLayout()
        
        self.back_btn = QtWidgets.QPushButton("← BACK")
        self.back_btn.setProperty("class", "back-btn")
        self.back_btn.clicked.connect(self.back_requested.emit)
        header_row.addWidget(self.back_btn)
        
        title = QtWidgets.QLabel("Video Replay")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        header_row.addWidget(title)
        header_row.addStretch()
        
        layout.addLayout(header_row)
        
        # Video widget container
        self.video_container = QtWidgets.QFrame()
        self.video_container.setStyleSheet("""
            QFrame {
                background: #0d1117;
                border: 2px solid #30363d;
                border-radius: 12px;
            }
        """)
        video_layout = QtWidgets.QVBoxLayout(self.video_container)
        
        # Placeholder (always created)
        self.placeholder = QtWidgets.QLabel("No video loaded")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("""
            font-size: 24px;
            color: #6e7681;
            min-height: 300px;
            background: transparent;
            border: none;
        """)
        
        # Video widget (only if multimedia available)
        self.video_widget = None
        self.media_player = None
        
        if HAS_MULTIMEDIA:
            self.video_widget = QVideoWidget()
            self.media_player = QMediaPlayer()
            self.media_player.setVideoOutput(self.video_widget)
            self.media_player.mediaStatusChanged.connect(self._on_media_status)
            video_layout.addWidget(self.video_widget)
            self.video_widget.hide()
        else:
            self.placeholder.setText("Video replay unavailable\n(Qt6 Multimedia not installed)")
        
        video_layout.addWidget(self.placeholder)
        layout.addWidget(self.video_container, 1)
        
        # Playback controls (only if multimedia available)
        if HAS_MULTIMEDIA:
            controls = QtWidgets.QHBoxLayout()
            controls.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            self.play_btn = QtWidgets.QPushButton("▶ Play")
            self.play_btn.clicked.connect(self._toggle_play)
            self.play_btn.setStyleSheet(ButtonStyle.START)
            
            self.stop_btn = QtWidgets.QPushButton("⏹ Stop")
            self.stop_btn.clicked.connect(self._stop)
            
            controls.addWidget(self.play_btn)
            controls.addWidget(self.stop_btn)
            
            layout.addLayout(controls)
    
    def load_video(self, path: str):
        """Load a video file for playback."""
        if not HAS_MULTIMEDIA:
            self.placeholder.setText("Video replay unavailable\n(Qt6 Multimedia not installed)")
            return
            
        import os
        if path and os.path.exists(path):
            self.media_player.setSource(QUrl.fromLocalFile(path))
            self.video_widget.show()
            self.placeholder.hide()
            self.media_player.play()
        else:
            self.video_widget.hide()
            self.placeholder.show()
            self.placeholder.setText(f"Video not found:\n{path}" if path else "No video loaded")
    
    def _toggle_play(self):
        if not HAS_MULTIMEDIA or not self.media_player:
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸ Pause")
    
    def _stop(self):
        if not HAS_MULTIMEDIA or not self.media_player:
            return
        self.media_player.stop()
        self.play_btn.setText("▶ Play")
    
    def _on_media_status(self, status):
        if HAS_MULTIMEDIA and status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.play_btn.setText("▶ Play")


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
        self.shadow_drill_client = self.create_client(StartDrill, "start_drill")
        self.defence_drill_client = self.create_client(StartDrill, "start_defence_drill")
        self.imu_input_client = self.create_client(SetBool, "imu_input_selector/enable")
        self.tracker_param_client = self.create_client(SetParameters, "realsense_glove_tracker/set_parameters")
        self.drill_param_client = self.create_client(SetParameters, "reaction_drill_manager/set_parameters")
        
        # Action prediction
        self.last_action: Optional[ActionPrediction] = None
        self.drill_progress: Optional[DrillProgress] = None
        self.imu_input_enabled = False
        
        self.action_sub = self.create_subscription(ActionPrediction, "action_prediction", self._on_action, 5)
        self.progress_sub = self.create_subscription(DrillProgress, "drill_progress", self._on_progress, 5)
        self.imu_enabled_sub = self.create_subscription(Bool, "imu_input_enabled", self._on_imu_enabled, 5)
        self.height_sub = self.create_subscription(Float32, "/player_height", self._on_height, 5)
        
        # New Services
        self.mode_client = self.create_client(SetBool, "action_predictor/set_simple_mode")
        self.height_trigger_client = self.create_client(Trigger, "action_predictor/calibrate_height")

    def _on_height(self, msg: Float32) -> None:
        pass # Optional: could update a status label somewhere
    
    def _on_action(self, msg: ActionPrediction) -> None:
        with self.lock:
            self.last_action = msg
    
    def _on_progress(self, msg: DrillProgress) -> None:
        with self.lock:
            self.drill_progress = msg
    
    def _on_imu_enabled(self, msg: Bool) -> None:
        with self.lock:
            self.imu_input_enabled = msg.data

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
        self.setMinimumSize(600, 450)
        self.resize(700, 500)
        
        self.ros = ros
        self._frame_buffer = deque(maxlen=180)
        self._last_punch_counter = 0
        self._replay_frames = []
        self._replay_index = 0

        self._apply_styles()
        
        # Main Layout container
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        self.header = QtWidgets.QLabel("BOXBUNNY TRAINER")
        self.header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.header.setObjectName("header")
        main_layout.addWidget(self.header)

        # Navigation Stack
        self.stack = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stack)

        # Initialize screens
        self.home_screen = QtWidgets.QWidget()
        self.reaction_tab = QtWidgets.QWidget()
        self.shadow_tab = QtWidgets.QWidget()
        self.defence_tab = QtWidgets.QWidget()
        self.punch_tab = QtWidgets.QWidget()
        self.llm_tab = QtWidgets.QWidget()
        self.calib_tab = QtWidgets.QWidget()
        
        # New enhanced pages
        self.shadow_countdown = CountdownSplashPage("Shadow Sparring")
        self.defence_countdown = CountdownSplashPage("Defence Drill")
        self.video_replay = VideoReplayPage()

        # Add to stack
        self.stack.addWidget(self.home_screen)       # 0
        self.stack.addWidget(self.reaction_tab)      # 1
        self.stack.addWidget(self.shadow_tab)        # 2
        self.stack.addWidget(self.defence_tab)       # 3
        self.stack.addWidget(self.punch_tab)         # 4
        self.stack.addWidget(self.llm_tab)           # 5
        self.stack.addWidget(self.calib_tab)         # 6
        self.stack.addWidget(self.shadow_countdown)  # 7
        self.stack.addWidget(self.defence_countdown) # 8
        self.stack.addWidget(self.video_replay)      # 9
        
        # Connect new page signals
        self.shadow_countdown.countdown_finished.connect(self._on_shadow_countdown_done)
        self.defence_countdown.countdown_finished.connect(self._on_defence_countdown_done)
        self.video_replay.back_requested.connect(lambda: self.stack.setCurrentWidget(self.home_screen))

        # Setup screens
        self._setup_home_screen()
        self._setup_reaction_tab()
        self._setup_shadow_tab()
        self._setup_defence_tab()
        self._setup_punch_tab()
        self._setup_llm_tab()
        self._setup_calibration_tab()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(50)

        self.replay_timer = QtCore.QTimer()
        self.replay_timer.timeout.connect(self._play_replay)

    def _apply_styles(self):
        self.setStyleSheet("""
            /* ===== MODERN DARK THEME ===== */
            
            /* Main Window - Deep charcoal gradient */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0d1117, stop:0.5 #161b22, stop:1 #0d1117);
            }
            
            /* Base typography */
            QLabel {
                color: #e6edf3;
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 16px;
                background: transparent;
                border: none;
            }
            
            /* Header - Sleek with accent glow - Compact */
            QLabel#header {
                font-size: 28px;
                font-weight: 700;
                letter-spacing: 2px;
                color: #ff4757;
                padding: 16px 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(22, 27, 34, 0.95), stop:1 rgba(13, 17, 23, 0.9));
                border-bottom: 1px solid rgba(255, 71, 87, 0.3);
                border-radius: 0;
            }
            
            /* Cards - Glassmorphism effect */
            QFrame, QGroupBox {
                background-color: rgba(22, 27, 34, 0.8);
                border-radius: 16px;
                border: 1px solid rgba(48, 54, 61, 0.8);
            }
            
            QGroupBox {
                margin-top: 16px;
                padding-top: 24px;
                font-weight: 600;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 8px 16px;
                color: #ff4757;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 1px;
            }
            
            /* Primary Buttons - Vibrant red gradient */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff4757, stop:1 #c0392b);
                color: #ffffff;
                border: none;
                padding: 14px 28px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 12px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6b7a, stop:1 #ff4757);
            }
            
            QPushButton:pressed {
                background: #a93226;
                padding-top: 16px;
                padding-bottom: 12px;
            }
            
            QPushButton:disabled {
                background: #30363d;
                color: #6e7681;
            }
            
            /* Form Inputs - Dark with subtle borders */
            QLineEdit, QComboBox, QSpinBox {
                padding: 12px 16px;
                border-radius: 10px;
                border: 2px solid #30363d;
                background: #0d1117;
                color: #e6edf3;
                font-size: 16px;
                selection-background-color: #ff4757;
            }
            
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 2px solid #ff4757;
            }
            
            QComboBox::drop-down {
                border: none;
                padding-right: 16px;
            }
            
            QComboBox QAbstractItemView {
                background: #161b22;
                color: #e6edf3;
                selection-background-color: #ff4757;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            
            /* Menu Buttons - Large prominent cards */
            QPushButton[class="menu-btn"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 71, 87, 0.9), stop:1 rgba(192, 57, 43, 0.9));
                color: #ffffff;
                font-size: 24px;
                font-weight: 700;
                padding: 32px 40px;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                text-align: left;
            }
            
            QPushButton[class="menu-btn"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 122, 0.95), stop:1 rgba(255, 71, 87, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            
            /* Back Button - Subtle ghost style */
            QPushButton[class="back-btn"] {
                background: transparent;
                color: #8b949e;
                border: 1px solid #30363d;
                font-size: 14px;
                font-weight: 500;
                padding: 10px 20px;
                max-width: 100px;
                border-radius: 8px;
            }
            
            QPushButton[class="back-btn"]:hover {
                background: rgba(255, 255, 255, 0.05);
                color: #e6edf3;
                border: 1px solid #8b949e;
            }
            
            /* Slider styling */
            QSlider::groove:horizontal {
                height: 6px;
                background: #30363d;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #ff4757;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #ff6b7a;
            }
            
            /* Text area */
            QTextEdit {
                background: #0d1117;
                color: #e6edf3;
                border: 2px solid #30363d;
                border-radius: 12px;
                padding: 12px;
                font-size: 15px;
            }
            
            QTextEdit:focus {
                border: 2px solid #ff4757;
            }
            
            /* Checkbox */
            QCheckBox {
                color: #e6edf3;
                font-size: 15px;
                spacing: 10px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #30363d;
                background: #0d1117;
            }
            
            QCheckBox::indicator:checked {
                background: #ff4757;
                border-color: #ff4757;
            }
            
            /* Scrollbar */
            QScrollBar:vertical {
                background: #0d1117;
                width: 10px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background: #30363d;
                border-radius: 5px;
                min-height: 30px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
        """)

    def _setup_home_screen(self) -> None:
        # Outer layout to center everything
        outer_layout = QtWidgets.QVBoxLayout()
        outer_layout.setContentsMargins(30, 10, 30, 10)  # Reduced vertical margins
        
        # Centered container with max width
        container = QtWidgets.QWidget()
        container.setMaximumWidth(500)
        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(0, 0, 0, 0)

        button_height = 65  # Reduced from 70
        btn_reaction = self._create_menu_btn("🎯 REACTION DRILL", "Test your reflexes", self.reaction_tab)
        btn_shadow = self._create_menu_btn("🥊 SHADOW SPARRING", "Practice combos", self.shadow_tab) 
        btn_defence = self._create_menu_btn("🛡️ DEFENCE DRILL", "Block incoming attacks", self.defence_tab)
        
        for btn in [btn_reaction, btn_shadow, btn_defence]:
            btn.setFixedHeight(button_height)
        
        main_layout.addWidget(btn_reaction)
        main_layout.addWidget(btn_shadow)
        main_layout.addWidget(btn_defence)
        
        main_layout.addSpacing(10)

        # Advanced Toggle button
        self.advanced_btn = QtWidgets.QPushButton("⚙️ Advanced Options ▾")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background: rgba(22, 27, 34, 0.6);
                color: #8b949e;
                border: 1px solid #30363d;
                font-size: 13px;
                padding: 8px 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: rgba(48, 54, 61, 0.6);
                color: #e6edf3;
            }
            QPushButton:checked {
                background: rgba(255, 71, 87, 0.1);
                color: #ff4757;
                border-color: rgba(255, 71, 87, 0.3);
            }
        """)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        main_layout.addWidget(self.advanced_btn)

        # Advanced Panel (Hidden by default)
        self.advanced_panel = QtWidgets.QWidget()
        adv_layout = QtWidgets.QHBoxLayout(self.advanced_panel)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(8)
        
        btn_punch = self._create_menu_btn("📊 Stats", "", self.punch_tab)
        btn_llm = self._create_menu_btn("💬 AI", "", self.llm_tab)
        btn_calib = self._create_menu_btn("⚙️ Calib", "", self.calib_tab)
        
        for btn in [btn_punch, btn_llm, btn_calib]:
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #30363d, stop:1 #21262d);
                    color: #e6edf3;
                    font-size: 13px;
                    padding: 8px;
                    border-radius: 8px;
                    border: 1px solid #484f58;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #484f58, stop:1 #30363d);
                    border-color: #8b949e;
                }
            """)

        adv_layout.addWidget(btn_punch)
        adv_layout.addWidget(btn_llm)
        adv_layout.addWidget(btn_calib)

        # Mode Toggle
        self.mode_btn = QtWidgets.QPushButton("Mode: AI Model")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setStyleSheet("background-color: #21262d; border: 1px solid #484f58;")
        self.mode_btn.clicked.connect(self._toggle_action_mode)
        adv_layout.addWidget(self.mode_btn)

        # Height Calib
        self.height_btn = QtWidgets.QPushButton("📏 Calib Height")
        self.height_btn.setStyleSheet("background-color: #21262d; border: 1px solid #484f58;")
        self.height_btn.clicked.connect(self._start_height_calibration)
        adv_layout.addWidget(self.height_btn)
        
        self.advanced_panel.setVisible(False)
        main_layout.addWidget(self.advanced_panel)

        # Layout alignment - Centered again but with compacted elements
        outer_layout.addStretch(1)
        outer_layout.addWidget(container, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        outer_layout.addStretch(1)

        self.home_screen.setLayout(outer_layout)

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_panel.setVisible(checked)
        self.advanced_btn.setText("⚙️ Advanced Options ▴" if checked else "⚙️ Advanced Options ▾")

    def _create_menu_btn(self, title, subtitle, target_widget):
        text = f"{title}\n{subtitle}" if subtitle else title
        btn = QtWidgets.QPushButton(text)
        btn.setProperty("class", "menu-btn")
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(target_widget))
        return btn

    def _add_back_btn(self, layout):
        btn = QtWidgets.QPushButton("← BACK")
        btn.setProperty("class", "back-btn")
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.home_screen))
        layout.addWidget(btn)
        return btn

    def _setup_reaction_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self._add_back_btn(layout)

        self.cue_panel = QtWidgets.QFrame()
        self.cue_panel.setFixedHeight(180)
        self.cue_panel.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(22, 27, 34, 0.9), stop:1 rgba(13, 17, 23, 0.9));
            border-radius: 16px;
            border: 1px solid rgba(48, 54, 61, 0.8);
        """)
        
        cue_layout = QtWidgets.QVBoxLayout(self.cue_panel)
        self.state_label = QtWidgets.QLabel("IDLE")
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("font-size: 48px; font-weight: bold; border: none; background: transparent;")
        
        self.countdown_label = QtWidgets.QLabel("--")
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 28px; color: #f0b429; border: none; background: transparent;")
        
        cue_layout.addWidget(self.state_label)
        cue_layout.addWidget(self.countdown_label)

        self.start_btn = QtWidgets.QPushButton("START DRILL")
        self.stop_btn = QtWidgets.QPushButton("STOP")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #484f58, stop:1 #30363d);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6e7681, stop:1 #484f58);
            }
        """)
        self.start_btn.clicked.connect(self._start_drill)
        self.stop_btn.clicked.connect(self._stop_drill)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)

        self.last_reaction_label = QtWidgets.QLabel("Last reaction: --")
        self.last_reaction_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.summary_label = QtWidgets.QLabel("Summary: --")
        self.summary_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.summary_label.setStyleSheet("color: #8b949e; font-size: 15px;")
        
        self.trash_label = QtWidgets.QLabel("--")
        self.trash_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.trash_label.setStyleSheet("font-style: italic; color: #f0b429; font-size: 18px; font-weight: 600;")

        self.reaction_preview = QtWidgets.QLabel()
        self.reaction_preview.setFixedSize(480, 320)
        self.reaction_preview.setStyleSheet("""
            background-color: #0d1117;
            border: 2px solid #30363d;
            border-radius: 12px;
        """)
        self.reaction_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        replay_row = QtWidgets.QHBoxLayout()
        self.replay_btn = QtWidgets.QPushButton("Replay Last")
        self.replay_btn.clicked.connect(self._start_replay)
        self.replay_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.replay_speed.setRange(5, 30)
        self.replay_speed.setValue(12)
        replay_row.addWidget(self.replay_btn)
        replay_row.addWidget(QtWidgets.QLabel("Slow-mo FPS"))
        replay_row.addWidget(self.replay_speed)
        
        # Central Content Wrapper
        content_layout = QtWidgets.QVBoxLayout()
        content_layout.addWidget(self.cue_panel)
        content_layout.addSpacing(20)
        content_layout.addLayout(button_row)
        content_layout.addSpacing(20)
        content_layout.addWidget(self.reaction_preview, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        content_layout.addLayout(replay_row)
        content_layout.addSpacing(20)
        content_layout.addWidget(self.last_reaction_label)
        content_layout.addWidget(self.summary_label)
        content_layout.addWidget(self.trash_label)

        # Center everything in the tab
        layout.addStretch(1)
        layout.addLayout(content_layout)
        layout.addStretch(1)
        
        self.reaction_tab.setLayout(layout)

    def _setup_punch_tab(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        
        # Left side with back button and preview
        left_layout = QtWidgets.QVBoxLayout()
        self._add_back_btn(left_layout)
        
        self.punch_preview = QtWidgets.QLabel()
        self.punch_preview.setFixedSize(640, 480)
        self.punch_preview.setStyleSheet("background-color: #222;")
        left_layout.addWidget(self.punch_preview)
        left_layout.addStretch(1)
        
        # Right side with IMU data
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
        self._add_back_btn(layout)
        
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
        self._add_back_btn(layout)
        
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
        
        # IMU Input Toggle
        self.imu_toggle = QtWidgets.QCheckBox("Enable IMU Input (punch to select)")
        self.imu_toggle.toggled.connect(self._toggle_imu_input)
        layout.addWidget(self.imu_toggle)
        
        self.llm_tab.setLayout(layout)
    
    def _setup_shadow_tab(self) -> None:
        """Setup shadow sparring drill tab."""
        outer_layout = QtWidgets.QVBoxLayout(self.shadow_tab)
        outer_layout.setContentsMargins(24, 16, 24, 24)
        self._add_back_btn(outer_layout)
        
        # Content container for max-width centering
        container = QtWidgets.QWidget()
        container.setMaximumWidth(800)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(20)
        
        # Drill selection card
        drill_card = QtWidgets.QFrame()
        drill_card.setStyleSheet("padding: 16px;")
        drill_layout = QtWidgets.QHBoxLayout(drill_card)
        drill_layout.setSpacing(16)
        
        combo_label = QtWidgets.QLabel("Select Combo:")
        combo_label.setStyleSheet("font-weight: 600; font-size: 15px;")
        drill_layout.addWidget(combo_label)
        
        self.shadow_combo = QtWidgets.QComboBox()
        self.shadow_combo.addItems([
            "1-1-2 Combo", "Jab-Cross-Hook", "Double Jab",
            "Cross-Hook-Cross", "Four Punch Combo", "Uppercut Combo"
        ])
        self.shadow_combo.setMinimumWidth(200)
        drill_layout.addWidget(self.shadow_combo)
        
        drill_layout.addStretch()
        
        self.shadow_start_btn = QtWidgets.QPushButton("▶  Start Drill")
        self.shadow_start_btn.clicked.connect(self._start_shadow_drill)
        self.shadow_start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #26d0ce, stop:1 #1a7f7e);
                padding: 12px 24px;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3ae0de, stop:1 #26d0ce);
            }
        """)
        drill_layout.addWidget(self.shadow_start_btn)
        
        layout.addWidget(drill_card)
        
        # Action prediction display - Large prominent card
        self.action_card = QtWidgets.QFrame()
        self.action_card.setFixedHeight(180)
        self.action_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(22, 27, 34, 0.9), stop:1 rgba(13, 17, 23, 0.9));
                border: 1px solid rgba(48, 54, 61, 0.8);
            }
        """)
        ac_layout = QtWidgets.QVBoxLayout(self.action_card)
        ac_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.action_label = QtWidgets.QLabel("READY")
        self.action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_label.setStyleSheet("""
            font-size: 48px;
            font-weight: 700;
            letter-spacing: 4px;
            color: #26d0ce;
        """)
        ac_layout.addWidget(self.action_label)
        
        self.action_conf_label = QtWidgets.QLabel("Confidence: --%")
        self.action_conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_conf_label.setStyleSheet("font-size: 18px; color: #8b949e;")
        ac_layout.addWidget(self.action_conf_label)
        
        layout.addWidget(self.action_card)
        
        # Checkbox progress indicator (visual feedback)
        self.shadow_checkbox_progress = CheckboxProgressWidget(count=5)
        layout.addWidget(self.shadow_checkbox_progress)
        
        # Progress display
        progress_group = QtWidgets.QGroupBox("Drill Progress")
        prog_layout = QtWidgets.QVBoxLayout(progress_group)
        prog_layout.setSpacing(8)
        
        self.shadow_progress_label = QtWidgets.QLabel("Step: -/-")
        self.shadow_progress_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.shadow_expected_label = QtWidgets.QLabel("Expected: --")
        self.shadow_expected_label.setStyleSheet("font-size: 16px; color: #ff4757; font-weight: 500;")
        self.shadow_elapsed_label = QtWidgets.QLabel("Elapsed: 0.0s")
        self.shadow_status_label = QtWidgets.QLabel("Status: idle")
        self.shadow_status_label.setStyleSheet("color: #8b949e;")
        
        prog_layout.addWidget(self.shadow_progress_label)
        prog_layout.addWidget(self.shadow_expected_label)
        prog_layout.addWidget(self.shadow_elapsed_label)
        prog_layout.addWidget(self.shadow_status_label)
        
        layout.addWidget(progress_group)
        
        # Sequence display
        self.shadow_sequence_label = QtWidgets.QLabel("Sequence: --")
        self.shadow_sequence_label.setWordWrap(True)
        self.shadow_sequence_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shadow_sequence_label.setStyleSheet("""
            font-size: 15px;
            color: #8b949e;
            padding: 12px;
            background: rgba(22, 27, 34, 0.5);
            border-radius: 8px;
        """)
        layout.addWidget(self.shadow_sequence_label)
        
        layout.addStretch(1)
        
        outer_layout.addWidget(container, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    
    def _setup_defence_tab(self) -> None:
        """Setup defence drill tab with numpad and block indicators."""
        outer_layout = QtWidgets.QVBoxLayout(self.defence_tab)
        outer_layout.setContentsMargins(24, 16, 24, 24)
        self._add_back_btn(outer_layout)
        
        # Container for centering
        container = QtWidgets.QWidget()
        container.setMaximumWidth(800)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(20)
        
        # Defence drill selection card
        drill_card = QtWidgets.QFrame()
        drill_card.setStyleSheet("padding: 16px;")
        drill_row = QtWidgets.QHBoxLayout(drill_card)
        drill_row.setSpacing(16)
        
        mode_label = QtWidgets.QLabel("Defence Mode:")
        mode_label.setStyleSheet("font-weight: 600; font-size: 15px;")
        drill_row.addWidget(mode_label)
        
        # REMOVED: Specific block area selection (Head/Body/etc)
        # Just generic defence now
        
        attacks_label = QtWidgets.QLabel("Attacks:")
        attacks_label.setStyleSheet("font-weight: 600; font-size: 15px;")
        drill_row.addWidget(attacks_label)
        
        self.defence_count_spin = QtWidgets.QSpinBox()
        self.defence_count_spin.setRange(5, 30)
        self.defence_count_spin.setValue(10)
        drill_row.addWidget(self.defence_count_spin)
        
        self.defence_start_btn = QtWidgets.QPushButton("▶  Start Drill")
        self.defence_start_btn.clicked.connect(self._start_defence_drill)
        self.defence_start_btn.setStyleSheet(ButtonStyle.START)
        drill_row.addWidget(self.defence_start_btn)
        
        layout.addWidget(drill_card)
        
        # Incoming Attack indicator - Large feedback panel
        self.block_indicator = QtWidgets.QFrame()
        self.block_indicator.setFixedHeight(180)
        self.block_indicator.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(22, 27, 34, 0.9), stop:1 rgba(13, 17, 23, 0.9));
            border-radius: 16px;
            border: 1px solid rgba(48, 54, 61, 0.8);
        """)
        bi_layout = QtWidgets.QVBoxLayout(self.block_indicator)
        bi_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.defence_action_label = QtWidgets.QLabel("Waiting to start...")
        self.defence_action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_action_label.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #8b949e;
            border: none;
            background: transparent;
        """)
        bi_layout.addWidget(self.defence_action_label)
        
        self.defence_sub_label = QtWidgets.QLabel("(Robot will strike - DODGE!)")
        self.defence_sub_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_sub_label.setStyleSheet("font-size: 18px; color: #666;")
        bi_layout.addWidget(self.defence_sub_label)
        
        layout.addWidget(self.block_indicator)
        
        # Checkbox progress for blocks completed
        self.defence_checkbox_progress = CheckboxProgressWidget(count=5)
        layout.addWidget(self.defence_checkbox_progress)
        
        # Progress display
        progress_group = QtWidgets.QGroupBox("Progress")
        prog_layout = QtWidgets.QVBoxLayout(progress_group)
        prog_layout.setSpacing(8)
        
        self.defence_progress_label = QtWidgets.QLabel("Dodges: 0/0")
        self.defence_progress_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.defence_elapsed_label = QtWidgets.QLabel("Elapsed: 0.0s")
        self.defence_status_label = QtWidgets.QLabel("Status: idle")
        self.defence_status_label.setStyleSheet("color: #8b949e;")
        
        prog_layout.addWidget(self.defence_progress_label)
        prog_layout.addWidget(self.defence_elapsed_label)
        prog_layout.addWidget(self.defence_status_label)
        
        layout.addWidget(progress_group)
        layout.addStretch(1)
        
        outer_layout.addWidget(container, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

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
        
        if not self.ros.tracker_param_client.service_is_ready():
            self.calib_status.setText("Tracker param service not ready")
            return

        req = SetParameters.Request()
        req.parameters = [p.to_parameter_msg() for p in ros_params]
        future = self.ros.tracker_param_client.call_async(req)
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
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(38, 208, 206, 0.9), stop:1 rgba(26, 127, 126, 0.9));
                border-radius: 16px;
                border: 1px solid #26d0ce;
            """)
            self.state_label.setText("PUNCH NOW!")
            self.state_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #fff; border: none;")
        elif state == "waiting":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(240, 180, 41, 0.9), stop:1 rgba(180, 130, 30, 0.9));
                border-radius: 16px;
                border: 1px solid #f0b429;
            """)
            self.state_label.setText("GET READY...")
        elif state == "countdown":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 71, 87, 0.8), stop:1 rgba(192, 57, 43, 0.8));
                border-radius: 16px;
                border: 1px solid #ff4757;
            """)
            self.state_label.setText("STEADY")
        elif state == "baseline":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(88, 166, 255, 0.8), stop:1 rgba(56, 139, 253, 0.8));
                border-radius: 16px;
                border: 1px solid #58a6ff;
            """)
            self.state_label.setText("STAY STILL")
        else:
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(22, 27, 34, 0.9), stop:1 rgba(13, 17, 23, 0.9));
                border-radius: 16px;
                border: 1px solid rgba(48, 54, 61, 0.8);
            """)
            self.state_label.setText("IDLE")

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
            f"Summary: mean {mean_rt if mean_rt is not None else '--'} | best {best_rt if best_rt is not None else '--'} | baseline v {baseline_v if baseline_v is not None else '--'}"
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
        
        # Update new drill tabs
        self._update_shadow_ui()
        self._update_defence_ui()

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
    
    def _start_shadow_drill(self) -> None:
        """Start shadow sparring drill."""
        if not self.ros.shadow_drill_client.service_is_ready():
            self.shadow_status_label.setText("Status: service not ready")
            return
        req = StartDrill.Request()
        req.drill_type = "shadow_sparring"
        req.drill_name = self.shadow_combo.currentText()
        req.repetitions = 1
        self.ros.shadow_drill_client.call_async(req)
        self.shadow_status_label.setText("Status: starting...")
    
    def _start_defence_drill(self) -> None:
        """Start defence drill."""
        if not self.ros.defence_drill_client.service_is_ready():
            self.defence_status_label.setText("Status: service not ready")
            return
        req = StartDrill.Request()
        req.drill_type = "defence"
        req.drill_name = self.defence_combo.currentText()
        req.repetitions = self.defence_count_spin.value()
        self.ros.defence_drill_client.call_async(req)
        self.defence_status_label.setText("Status: starting...")
    
    def _toggle_imu_input(self, enabled: bool) -> None:
        """Toggle IMU input for menu selection."""
        if not self.ros.imu_input_client.service_is_ready():
            return
        req = SetBool.Request()
        req.data = enabled
        self.ros.imu_input_client.call_async(req)
    
    def _update_shadow_ui(self) -> None:
        """Update shadow sparring tab UI."""
        with self.ros.lock:
            action = self.ros.last_action
            progress = self.ros.drill_progress
        
        # Action prediction display
        if action is not None:
            self.action_label.setText(f"{action.action_label.upper()}")
            self.action_conf_label.setText(f"Confidence: {action.confidence * 100:.0f}%")
            if action.confidence > 0.7:
                 self.action_label.setStyleSheet("""
                    font-size: 48px;
                    font-weight: 700;
                    letter-spacing: 4px;
                    color: #26d0ce;
                 """)
            else:
                 self.action_label.setStyleSheet("""
                    font-size: 48px;
                    font-weight: 700;
                    letter-spacing: 4px;
                    color: #ff4757;
                 """)
        
        # Drill progress
        if progress is not None and progress.drill_name:
            current_step = progress.current_step
            total_steps = progress.total_steps
            
            self.shadow_progress_label.setText(f"Step: {current_step}/{total_steps}")
            if current_step < len(progress.expected_actions):
                expected = progress.expected_actions[current_step]
                self.shadow_expected_label.setText(f"Expected: {expected.upper()}")
            self.shadow_elapsed_label.setText(f"Elapsed: {progress.elapsed_time_s:.1f}s")
            self.shadow_status_label.setText(f"Status: {progress.status}")
            self.shadow_sequence_label.setText(
                f"Sequence: {' → '.join(progress.expected_actions)}")
            
            # Update checkbox progress - tick checkboxes for completed steps
            if not hasattr(self, '_last_shadow_step'):
                self._last_shadow_step = 0
            if current_step > self._last_shadow_step:
                # Tick checkboxes for newly completed steps
                for i in range(self._last_shadow_step, min(current_step, 5)):
                    self.shadow_checkbox_progress.tick(i)
            self._last_shadow_step = current_step
            
            # Reset checkboxes when drill completes or restarts
            if progress.status == 'complete' or progress.status == 'idle':
                self._last_shadow_step = 0
                self.shadow_checkbox_progress.reset()
    
    def _update_defence_ui(self) -> None:
        """Update defence drill tab UI."""
        with self.ros.lock:
            action = self.ros.last_action
            progress = self.ros.drill_progress
        
        # Block detection
        if action is not None:
            if action.action_label == 'block' and action.confidence > 0.5:
                self.block_indicator.setStyleSheet("""
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(38, 208, 206, 0.8), stop:1 rgba(26, 127, 126, 0.8));
                    border-radius: 16px;
                    border: 1px solid #26d0ce;
                """)
                self.defence_action_label.setText("BLOCK DETECTED!")
            else:
                self.block_indicator.setStyleSheet("""
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(22, 27, 34, 0.9), stop:1 rgba(13, 17, 23, 0.9));
                    border-radius: 16px;
                    border: 1px solid rgba(48, 54, 61, 0.8);
                """)
                self.defence_action_label.setText(f"Detected: {action.action_label}")
        
        # Progress
        if progress is not None and 'Defence' in progress.drill_name:
            successful = sum(progress.step_completed) if progress.step_completed else 0
            self.defence_progress_label.setText(
                f"Blocks: {successful}/{progress.total_steps}")
            self.defence_elapsed_label.setText(f"Elapsed: {progress.elapsed_time_s:.1f}s")
            self.defence_status_label.setText(f"Status: {progress.status}")
            
            # Update checkbox progress based on completed steps
            if hasattr(self, '_last_defence_successful') and successful > self._last_defence_successful:
                for i in range(self._last_defence_successful, min(successful, 5)):
                    self.defence_checkbox_progress.tick(i)
            self._last_defence_successful = successful
    
    def _on_shadow_countdown_done(self) -> None:
        """Handle shadow countdown completion - start the actual drill."""
        self.stack.setCurrentWidget(self.shadow_tab)
        self._start_shadow_drill()
    
    def _on_defence_countdown_done(self) -> None:
        """Handle defence countdown completion - start the actual drill."""
        self.stack.setCurrentWidget(self.defence_tab)
        self._start_defence_drill()
    
    def _on_blocking_zone_selected(self, zone: int) -> None:
        """Handle numpad button press for blocking zone selection."""
        # Update the action label to show selected zone
        zone_names = {
            1: "HEAD LEFT", 2: "HEAD CENTER", 3: "HEAD RIGHT",
            4: "BODY LEFT", 5: "BODY CENTER", 6: "BODY RIGHT"
        }
        zone_name = zone_names.get(zone, f"Zone {zone}")
        
        # Flash the block indicator
        self.block_indicator.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(33, 150, 243, 0.8), stop:1 rgba(21, 101, 192, 0.8));
            border-radius: 16px;
            border: 1px solid #2196F3;
        """)
        self.defence_action_label.setText(f"Selected: {zone_name}")
        self.defence_action_label.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: #fff;
            border: none;
            background: transparent;
        """)
        
        # Reset style after brief delay
        QtCore.QTimer.singleShot(800, lambda: self.block_indicator.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(22, 27, 34, 0.9), stop:1 rgba(13, 17, 23, 0.9));
            border-radius: 16px;
            border: 1px solid rgba(48, 54, 61, 0.8);
        """))
    
    def _show_video_replay(self, video_path: str = None) -> None:
        """Navigate to video replay page and load video."""
        self.video_replay.load_video(video_path)
        self.stack.setCurrentWidget(self.video_replay)


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


    def _toggle_action_mode(self) -> None:
        is_simple = self.mode_btn.isChecked()
        self.mode_btn.setText("Mode: Simple/Color" if is_simple else "Mode: AI Model")
        self.mode_btn.setStyleSheet(
            "background-color: #ff4757; color: white;" if is_simple else "background-color: #21262d; border: 1px solid #484f58;"
        )
        
        if self.ros.mode_client.service_is_ready():
            req = SetBool.Request()
            req.data = is_simple
            self.ros.mode_client.call_async(req)
        else:
            print("Mode service not ready")

    def _start_height_calibration(self) -> None:
        # Use countdown splash for the 3s wait
        self.stack.setCurrentWidget(self.shadow_countdown)
        self.shadow_countdown.set_status("Stand Straight for Height Calibration...")
        self.shadow_countdown.countdown_finished.disconnect()
        self.shadow_countdown.countdown_finished.connect(self._trigger_height_calc)
        self.shadow_countdown.start(3)

    def _trigger_height_calc(self) -> None:
        self.stack.setCurrentWidget(self.home_screen)
        if self.ros.height_trigger_client.service_is_ready():
             self.ros.height_trigger_client.call_async(Trigger.Request())
             QtWidgets.QMessageBox.information(self, "Height", "Calibration request sent! Check logs/status.")
        else:
             QtWidgets.QMessageBox.warning(self, "Height", "Height service not ready")

if __name__ == "__main__":
    main()
