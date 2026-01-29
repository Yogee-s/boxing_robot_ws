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
    """Visual progress tracker with numbered step indicators."""
    
    def __init__(self, count: int = 3, parent=None):
        super().__init__(parent)
        self.count = count
        self.current = 0
        self.checkboxes = []
        
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setSpacing(4)
        outer_layout.setContentsMargins(0, 8, 0, 0)
        
        # Title label
        title = QtWidgets.QLabel("PROGRESS")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 10px; font-weight: 700; color: #555; letter-spacing: 1px;")
        outer_layout.addWidget(title)
        
        # Checkboxes row
        checkbox_row = QtWidgets.QHBoxLayout()
        checkbox_row.setSpacing(8)
        checkbox_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        for i in range(count):
            checkbox = QtWidgets.QLabel(f"{i+1}")
            checkbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            checkbox.setStyleSheet("""
                font-size: 16px;
                font-weight: 700;
                color: #484f58;
                min-width: 32px;
                min-height: 32px;
                background: #1a1a1a;
                border: 2px solid #333;
                border-radius: 6px;
            """)
            checkbox_row.addWidget(checkbox)
            self.checkboxes.append(checkbox)
        
        outer_layout.addLayout(checkbox_row)
    
    def tick(self, index: int = None):
        """Tick the checkbox at the given index (or next if None)."""
        if index is None:
            index = self.current
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].setText("✓")
            self.checkboxes[index].setStyleSheet("""
                font-size: 16px;
                font-weight: 700;
                color: #000;
                min-width: 32px;
                min-height: 32px;
                background: #26d0ce;
                border: 2px solid #26d0ce;
                border-radius: 6px;
            """)
            self.current = index + 1
    
    def reset(self):
        """Reset all checkboxes to empty."""
        self.current = 0
        for i, checkbox in enumerate(self.checkboxes):
            checkbox.setText(f"{i+1}")
            checkbox.setStyleSheet("""
                font-size: 16px;
                font-weight: 700;
                color: #484f58;
                min-width: 32px;
                min-height: 32px;
                background: #1a1a1a;
                border: 2px solid #333;
                border-radius: 6px;
            """)


# ============================================================================
# STARTUP LOADING SCREEN
# ============================================================================

class StartupLoadingScreen(QtWidgets.QWidget):
    """Loading screen that waits for camera and LLM to be ready."""
    
    ready = QtCore.Signal()
    
    def __init__(self, ros_interface, parent=None):
        super().__init__(parent)
        self.ros = ros_interface
        self.camera_ready = False
        self.llm_ready = False
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)
        
        # Title - smaller for 7" screen
        title = QtWidgets.QLabel("🥊 BOXBUNNY")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: 800;
            color: #ff8c00;
            background: transparent;
            letter-spacing: 3px;
        """)
        layout.addWidget(title)
        
        # Loading spinner/status
        self.status_label = QtWidgets.QLabel("⏳ Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #888888;
            background: transparent;
        """)
        layout.addWidget(self.status_label)
        
        # Status items - compact
        status_frame = QtWidgets.QFrame()
        status_frame.setMaximumWidth(280)
        status_layout = QtWidgets.QVBoxLayout(status_frame)
        status_layout.setSpacing(8)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_status = QtWidgets.QLabel("⏳ Camera: Connecting...")
        self.camera_status.setStyleSheet("font-size: 13px; color: #666666; padding: 4px;")
        status_layout.addWidget(self.camera_status)
        
        self.llm_status = QtWidgets.QLabel("⏳ AI Coach: Connecting...")
        self.llm_status.setStyleSheet("font-size: 13px; color: #666666; padding: 4px;")
        status_layout.addWidget(self.llm_status)
        
        layout.addWidget(status_frame, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Skip button (appears after timeout) - more prominent
        self.skip_btn = QtWidgets.QPushButton("Skip & Continue →")
        self.skip_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 140, 0, 0.2);
                color: #ff8c00;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 24px;
                border: 2px solid #ff8c00;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: rgba(255, 140, 0, 0.4);
            }
        """)
        self.skip_btn.clicked.connect(self._skip)
        self.skip_btn.hide()
        layout.addWidget(self.skip_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Check timer
        self.check_timer = QtCore.QTimer()
        self.check_timer.timeout.connect(self._check_services)
        
        # Timeout timer - show skip button after 2 seconds (faster for touchscreen UX)
        self.timeout_timer = QtCore.QTimer()
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self._show_skip)
        
    def start_checking(self):
        """Start checking for services."""
        self.check_timer.start(300)  # Check every 300ms (faster)
        self.timeout_timer.start(2000)  # Show skip after 2s (faster)
    
    def _check_services(self):
        """Check if camera and LLM are ready."""
        # Check camera - look for image data from either topic
        # live_infer_rgbd.py publishes to /glove_debug_image (last_image)
        # realsense node publishes to /camera/color/image_raw (last_color_image)
        with self.ros.lock:
            has_camera = self.ros.last_image is not None or self.ros.last_color_image is not None
        
        if has_camera and not self.camera_ready:
            self.camera_ready = True
            self.camera_status.setText("✅ Camera: Ready")
            self.camera_status.setStyleSheet("font-size: 13px; color: #00ff00; padding: 4px; font-weight: 600;")
        
        # Check LLM service
        llm_available = self.ros.llm_client.service_is_ready()
        if llm_available and not self.llm_ready:
            self.llm_ready = True
            self.llm_status.setText("✅ AI Coach: Ready")
            self.llm_status.setStyleSheet("font-size: 13px; color: #00ff00; padding: 4px; font-weight: 600;")
        
        # Update main status
        if self.camera_ready and self.llm_ready:
            self.status_label.setText("✅ All systems ready!")
            self.status_label.setStyleSheet("font-size: 14px; color: #00ff00; background: transparent; font-weight: 600;")
            self.check_timer.stop()
            self.timeout_timer.stop()
            # Small delay then signal ready
            QtCore.QTimer.singleShot(300, self.ready.emit)
        elif self.camera_ready:
            self.status_label.setText("⏳ Waiting for AI Coach...")
        elif self.llm_ready:
            self.status_label.setText("⏳ Waiting for Camera...")
    
    def _show_skip(self):
        """Show skip button after timeout."""
        self.skip_btn.show()
        self.status_label.setText("Tap Skip to continue...")
    
    def _skip(self):
        """Skip waiting and proceed."""
        self.check_timer.stop()
        self.timeout_timer.stop()
        self.ready.emit()


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
    # Target display: 7-inch HDMI touchscreen (1024x600)
    SCREEN_WIDTH = 1024
    SCREEN_HEIGHT = 600
    
    def __init__(self, ros: RosInterface) -> None:
        super().__init__()
        self.ros = ros
        self.setWindowTitle("BoxBunny Trainer")
        
        # Fixed size for 7-inch touchscreen (1024x600)
        self.setFixedSize(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.setMinimumSize(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        
        self._frame_buffer = deque(maxlen=180)
        self._last_punch_counter = 0
        self._replay_frames = []
        self._replay_index = 0
        self._initialized = False
        self._camera_received = False

        self._apply_styles()
        
        # Main Layout container
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header - responsive height
        self.header = QtWidgets.QLabel("BOXBUNNY TRAINER")
        self.header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.header.setObjectName("header")
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        main_layout.addWidget(self.header)

        # Navigation Stack
        self.stack = QtWidgets.QStackedWidget()
        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.stack)
        
        # ===== STARTUP LOADING SCREEN (Index 0) =====
        self.startup_screen = StartupLoadingScreen(ros)
        self.startup_screen.ready.connect(self._on_startup_complete)
        self.stack.addWidget(self.startup_screen)  # 0

        # Initialize main screens
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

        # Add to stack (indexes shifted by 1 due to startup screen)
        self.stack.addWidget(self.home_screen)       # 1
        self.stack.addWidget(self.reaction_tab)      # 2
        self.stack.addWidget(self.shadow_tab)        # 3
        self.stack.addWidget(self.defence_tab)       # 4
        self.stack.addWidget(self.punch_tab)         # 5
        self.stack.addWidget(self.llm_tab)           # 6
        self.stack.addWidget(self.calib_tab)         # 7
        self.stack.addWidget(self.shadow_countdown)  # 8
        self.stack.addWidget(self.defence_countdown) # 9
        self.stack.addWidget(self.video_replay)      # 10
        
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

        # Start on loading screen
        self.stack.setCurrentWidget(self.startup_screen)
        
        # Start checking for services after a brief delay
        QtCore.QTimer.singleShot(500, self.startup_screen.start_checking)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(50)

        self.replay_timer = QtCore.QTimer()
        self.replay_timer.timeout.connect(self._play_replay)
    
    def _on_startup_complete(self):
        """Called when startup loading is complete."""
        self._initialized = True
        self._camera_received = True  # Mark camera as received since startup confirmed it
        # Update video status to show ready
        if hasattr(self, 'video_status_label'):
            self.video_status_label.setText("📹 LIVE ●")
            self.video_status_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #00ff00;")
        self.stack.setCurrentWidget(self.home_screen)

    def _apply_styles(self):
        self.setStyleSheet("""
            /* ===== ORANGE & BLACK BOXING THEME ===== */
            
            /* Main Window - Pure black with subtle gradient */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0a0a0a, stop:0.5 #121212, stop:1 #0a0a0a);
            }
            
            /* Base typography */
            QLabel {
                color: #f0f0f0;
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 16px;
                background: transparent;
                border: none;
            }
            
            /* Header - Bold orange accent */
            QLabel#header {
                font-size: 32px;
                font-weight: 800;
                letter-spacing: 4px;
                color: #ff8c00;
                padding: 20px 24px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(20, 20, 20, 0.98), stop:1 rgba(10, 10, 10, 0.95));
                border-bottom: 3px solid #ff8c00;
                border-radius: 0;
            }
            
            /* Cards - Dark with orange accents */
            QFrame, QGroupBox {
                background-color: rgba(18, 18, 18, 0.9);
                border-radius: 16px;
                border: 1px solid rgba(255, 140, 0, 0.2);
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
                color: #ff8c00;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            
            /* Primary Buttons - Bold orange gradient */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff8c00, stop:1 #cc7000);
                color: #000000;
                border: none;
                padding: 16px 32px;
                font-size: 17px;
                font-weight: 700;
                border-radius: 14px;
                min-height: 24px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffa333, stop:1 #ff8c00);
            }
            
            QPushButton:pressed {
                background: #b36200;
                padding-top: 18px;
                padding-bottom: 14px;
            }
            
            QPushButton:disabled {
                background: #2a2a2a;
                color: #555555;
            }
            
            /* Form Inputs - Dark with orange accents */
            QLineEdit, QComboBox, QSpinBox {
                padding: 12px 16px;
                border-radius: 10px;
                border: 2px solid #333333;
                background: #1a1a1a;
                color: #f0f0f0;
                font-size: 16px;
                selection-background-color: #ff8c00;
            }
            
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 2px solid #ff8c00;
            }
            
            QComboBox::drop-down {
                border: none;
                padding-right: 16px;
            }
            
            QComboBox QAbstractItemView {
                background: #1a1a1a;
                color: #f0f0f0;
                selection-background-color: #ff8c00;
                border: 1px solid #333333;
                border-radius: 8px;
            }
            
            /* Menu Buttons - Large bold orange cards */
            QPushButton[class="menu-btn"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff8c00, stop:1 #e67300);
                color: #000000;
                font-size: 22px;
                font-weight: 800;
                padding: 28px 36px;
                border-radius: 18px;
                border: 2px solid rgba(255, 255, 255, 0.15);
                text-align: left;
                letter-spacing: 1px;
            }
            
            QPushButton[class="menu-btn"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffa333, stop:1 #ff8c00);
                border: 2px solid rgba(255, 255, 255, 0.4);
                color: #000000;
            }
            
            /* Back Button - Subtle ghost style */
            QPushButton[class="back-btn"] {
                background: transparent;
                color: #888888;
                border: 1px solid #333333;
                font-size: 14px;
                font-weight: 500;
                padding: 10px 20px;
                max-width: 100px;
                border-radius: 8px;
            }
            
            QPushButton[class="back-btn"]:hover {
                background: rgba(255, 140, 0, 0.1);
                color: #ff8c00;
                border: 1px solid #ff8c00;
            }
            
            /* Slider styling */
            QSlider::groove:horizontal {
                height: 6px;
                background: #333333;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #ff8c00;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #ffa333;
            }
            
            /* Text area */
            QTextEdit {
                background: #1a1a1a;
                color: #f0f0f0;
                border: 2px solid #333333;
                border-radius: 12px;
                padding: 12px;
                font-size: 15px;
            }
            
            QTextEdit:focus {
                border: 2px solid #ff8c00;
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
        """Clean, aesthetic home screen for 7\" touchscreen (1024x600)."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(40, 20, 40, 16)
        layout.setSpacing(12)
        
        # Center everything vertically
        layout.addStretch(1)
        
        # === MAIN DRILL BUTTONS (centered) ===
        drills_container = QtWidgets.QWidget()
        drills_container.setFixedWidth(500)
        drills_layout = QtWidgets.QVBoxLayout(drills_container)
        drills_layout.setSpacing(10)
        drills_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_reaction = self._create_menu_btn_centered("🎯  REACTION", self.reaction_tab)
        btn_shadow = self._create_menu_btn_centered("🥊  SHADOW", self.shadow_tab)
        btn_defence = self._create_menu_btn_centered("🛡️  DEFENCE", self.defence_tab)
        
        for btn in [btn_reaction, btn_shadow, btn_defence]:
            btn.setFixedHeight(65)
            drills_layout.addWidget(btn)
        
        layout.addWidget(drills_container, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        layout.addSpacing(12)
        
        # === QUICK ACCESS ROW (horizontal) ===
        quick_container = QtWidgets.QWidget()
        quick_container.setFixedWidth(500)
        quick_row = QtWidgets.QHBoxLayout(quick_container)
        quick_row.setSpacing(10)
        quick_row.setContentsMargins(0, 0, 0, 0)
        
        btn_stats = self._create_quick_btn("📊", "STATS", self.punch_tab)
        btn_llm = self._create_quick_btn("💬", "COACH", self.llm_tab)
        btn_calib = self._create_quick_btn("⚙️", "SETUP", self.calib_tab)
        
        for btn in [btn_stats, btn_llm, btn_calib]:
            quick_row.addWidget(btn, stretch=1)
        
        layout.addWidget(quick_container, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        layout.addSpacing(8)
        
        # === ADVANCED TOGGLE ===
        adv_container = QtWidgets.QWidget()
        adv_container.setFixedWidth(500)
        adv_layout = QtWidgets.QVBoxLayout(adv_container)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(8)
        
        self.advanced_btn = QtWidgets.QPushButton("⚗️ ADVANCED ▾")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setFixedHeight(30)
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #555555;
                border: 1px solid #333333;
                font-size: 11px;
                font-weight: 600;
                border-radius: 6px;
            }
            QPushButton:hover { color: #ff8c00; border-color: #ff8c00; }
            QPushButton:checked { color: #ff8c00; border-color: #ff8c00; background: rgba(255,140,0,0.1); }
        """)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        adv_layout.addWidget(self.advanced_btn)

        # Advanced Panel (Hidden by default)
        self.advanced_panel = QtWidgets.QFrame()
        self.advanced_panel.setStyleSheet("background: rgba(25,25,25,0.95); border-radius: 8px; border: 1px solid #333;")
        adv_panel_layout = QtWidgets.QVBoxLayout(self.advanced_panel)
        adv_panel_layout.setContentsMargins(12, 10, 12, 10)
        adv_panel_layout.setSpacing(8)
        
        # Detection mode row
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(12)
        mode_label = QtWidgets.QLabel("Detection:")
        mode_label.setStyleSheet("font-size: 11px; color: #888; font-weight: 600;")
        mode_row.addWidget(mode_label)
        
        self.color_mode_radio = QtWidgets.QRadioButton("Color")
        self.color_mode_radio.setChecked(True)
        self.color_mode_radio.setStyleSheet("font-size: 11px; color: #ff8c00;")
        self.color_mode_radio.toggled.connect(self._on_detection_mode_changed)
        mode_row.addWidget(self.color_mode_radio)
        
        self.action_mode_radio = QtWidgets.QRadioButton("AI Model")
        self.action_mode_radio.setStyleSheet("font-size: 11px; color: #888;")
        mode_row.addWidget(self.action_mode_radio)
        
        mode_row.addSpacing(20)
        
        self.imu_toggle = QtWidgets.QCheckBox("IMU Input")
        self.imu_toggle.setStyleSheet("font-size: 11px; color: #888;")
        self.imu_toggle.toggled.connect(self._toggle_imu_input)
        mode_row.addWidget(self.imu_toggle)
        
        mode_row.addStretch()
        
        self.height_btn = QtWidgets.QPushButton("📏 Height")
        self.height_btn.setFixedSize(70, 24)
        self.height_btn.setStyleSheet("background: #2a2a2a; color: #ff8c00; font-size: 10px; border-radius: 4px;")
        self.height_btn.clicked.connect(self._start_height_calibration)
        mode_row.addWidget(self.height_btn)
        
        self.imu_calib_btn = QtWidgets.QPushButton("🔧 IMU")
        self.imu_calib_btn.setFixedSize(60, 24)
        self.imu_calib_btn.setStyleSheet("background: #2a2a2a; color: #555; font-size: 10px; border-radius: 4px;")
        self.imu_calib_btn.clicked.connect(self._open_imu_calibration)
        self.imu_calib_btn.setEnabled(False)
        mode_row.addWidget(self.imu_calib_btn)
        
        adv_panel_layout.addLayout(mode_row)
        
        self.advanced_panel.setVisible(False)
        adv_layout.addWidget(self.advanced_panel)
        
        layout.addWidget(adv_container, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch(1)
        
        # Status indicator at bottom
        self.status_indicator = QtWidgets.QLabel("● Ready")
        self.status_indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_indicator.setStyleSheet("font-size: 12px; color: #00cc00; font-weight: 600;")
        layout.addWidget(self.status_indicator)
        
        self.home_screen.setLayout(layout)
    
    def _create_menu_btn_centered(self, title: str, target_widget):
        """Create a centered menu button."""
        btn = QtWidgets.QPushButton(title)
        btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff8c00, stop:1 #e67300);
                color: #000000;
                font-size: 20px;
                font-weight: 800;
                padding: 16px 36px;
                border-radius: 14px;
                border: 2px solid rgba(255, 255, 255, 0.15);
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffa333, stop:1 #ff8c00);
                border: 2px solid rgba(255, 255, 255, 0.4);
            }
        """)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(target_widget))
        return btn
    
    def _create_quick_btn(self, icon: str, title: str, target_widget):
        """Create a quick access button with icon and title."""
        btn = QtWidgets.QPushButton(f"{icon}\n{title}")
        btn.setFixedHeight(55)
        btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #151515);
                color: #ff8c00;
                font-size: 11px;
                font-weight: 700;
                padding: 8px;
                border-radius: 10px;
                border: 1px solid #333333;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1e1e1e);
                border-color: #ff8c00;
            }
        """)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(target_widget))
        return btn

    def _on_detection_mode_changed(self) -> None:
        """Handle detection mode radio button change."""
        is_action = self.action_mode_radio.isChecked()
        
        # Update status indicator
        if is_action:
            self.status_indicator.setText("● AI Mode")
            self.status_indicator.setStyleSheet("font-size: 11px; color: #f0b429; padding: 4px;")
        else:
            self.status_indicator.setText("● Color Mode")
            self.status_indicator.setStyleSheet("font-size: 11px; color: #26d0ce; padding: 4px;")
        
        # Send mode change to backend
        if self.ros.mode_client.service_is_ready():
            req = SetBool.Request()
            req.data = not is_action  # simple_mode = True for color tracking
            self.ros.mode_client.call_async(req)
    
    def _open_imu_calibration(self) -> None:
        """Open the IMU calibration GUI."""
        import subprocess
        imu_gui_path = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/src/boxbunny_imu/boxbunny_imu/imu_punch_gui.py"
        subprocess.Popen(['python3', imu_gui_path])

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_panel.setVisible(checked)
        self.advanced_btn.setText("⚗️ Advanced ▴" if checked else "⚗️ Advanced ▾")
    
    def _toggle_imu_input(self, enabled: bool) -> None:
        """Toggle IMU input for punch detection."""
        self.imu_calib_btn.setEnabled(enabled)
        if not self.ros.imu_input_client.service_is_ready():
            return
        req = SetBool.Request()
        req.data = enabled
        self.ros.imu_input_client.call_async(req)

    def _create_menu_btn(self, title, subtitle, target_widget):
        text = f"{title}\n{subtitle}" if subtitle else title
        btn = QtWidgets.QPushButton(text)
        btn.setProperty("class", "menu-btn")
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(target_widget))
        return btn

    def _add_back_btn(self, layout):
        btn = QtWidgets.QPushButton("← BACK")
        btn.setProperty("class", "back-btn")
        btn.setFixedHeight(28)
        btn.setFixedWidth(80)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ff8c00;
                font-size: 11px;
                font-weight: 600;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QPushButton:hover { border-color: #ff8c00; }
        """)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.home_screen))
        layout.addWidget(btn)
        return btn

    def _setup_reaction_tab(self) -> None:
        """Reaction drill - clean aesthetic layout for 7" touchscreen."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        self._add_back_btn(layout)
        
        # Main content area - horizontal split
        main_content = QtWidgets.QHBoxLayout()
        main_content.setSpacing(16)
        
        # === LEFT COLUMN: Camera Feed ===
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)
        
        # Video container with header
        video_frame = QtWidgets.QFrame()
        video_frame.setFixedWidth(380)
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 2px solid #222;
                border-radius: 10px;
            }
        """)
        video_inner = QtWidgets.QVBoxLayout(video_frame)
        video_inner.setContentsMargins(6, 6, 6, 6)
        video_inner.setSpacing(4)
        
        # Video header row
        video_header = QtWidgets.QHBoxLayout()
        self.video_status_label = QtWidgets.QLabel("📹 LIVE")
        self.video_status_label.setStyleSheet("font-size: 11px; font-weight: 700; color: #00cc00;")
        video_header.addWidget(self.video_status_label)
        video_header.addStretch()
        self.replay_btn = QtWidgets.QPushButton("🔄 Replay")
        self.replay_btn.setFixedHeight(22)
        self.replay_btn.setStyleSheet("background: #222; color: #ff8c00; border-radius: 4px; font-size: 10px; padding: 2px 8px;")
        self.replay_btn.clicked.connect(self._start_replay)
        video_header.addWidget(self.replay_btn)
        video_inner.addLayout(video_header)
        
        # Video preview
        self.reaction_preview = QtWidgets.QLabel()
        self.reaction_preview.setFixedSize(366, 275)
        self.reaction_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.reaction_preview.setText("⏳ Connecting...")
        self.reaction_preview.setStyleSheet("""
            background: #000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #555;
            font-size: 13px;
        """)
        video_inner.addWidget(self.reaction_preview)
        
        self.replay_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.replay_speed.setRange(5, 30)
        self.replay_speed.setValue(12)
        self.replay_speed.setVisible(False)
        
        left_col.addWidget(video_frame)
        left_col.addStretch()
        main_content.addLayout(left_col)
        
        # === RIGHT COLUMN: Controls ===
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(10)
        
        # Cue Panel - prominent status display
        self.cue_panel = QtWidgets.QFrame()
        self.cue_panel.setFixedHeight(120)
        self.cue_panel.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a1a1a, stop:1 #0d0d0d);
            border-radius: 12px;
            border: 2px solid #333;
        """)
        cue_layout = QtWidgets.QVBoxLayout(self.cue_panel)
        cue_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.state_label = QtWidgets.QLabel("READY")
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("font-size: 42px; font-weight: 800; color: #ff8c00; background: transparent;")
        
        self.countdown_label = QtWidgets.QLabel("Press START to begin")
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 12px; color: #666; background: transparent;")
        
        cue_layout.addWidget(self.state_label)
        cue_layout.addWidget(self.countdown_label)
        right_col.addWidget(self.cue_panel)
        
        # Control Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.start_btn = QtWidgets.QPushButton("▶  START")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000;
                font-size: 16px;
                font-weight: 700;
                border-radius: 10px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        self.start_btn.clicked.connect(self._start_drill)
        
        self.stop_btn = QtWidgets.QPushButton("⬛  STOP")
        self.stop_btn.setFixedHeight(50)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #888;
                font-size: 16px;
                font-weight: 700;
                border-radius: 10px;
                border: 1px solid #333;
            }
            QPushButton:hover { background: #333; color: #fff; }
        """)
        self.stop_btn.clicked.connect(self._stop_drill)
        
        btn_row.addWidget(self.start_btn, stretch=1)
        btn_row.addWidget(self.stop_btn, stretch=1)
        right_col.addLayout(btn_row)
        
        # Results Panel
        results_frame = QtWidgets.QFrame()
        results_frame.setStyleSheet("""
            background: #151515;
            border-radius: 8px;
            border: 1px solid #282828;
        """)
        results_inner = QtWidgets.QHBoxLayout(results_frame)
        results_inner.setContentsMargins(16, 12, 16, 12)
        results_inner.setSpacing(20)
        
        # Last
        last_col = QtWidgets.QVBoxLayout()
        last_title = QtWidgets.QLabel("LAST")
        last_title.setStyleSheet("font-size: 10px; color: #666;")
        self.last_reaction_label = QtWidgets.QLabel("--")
        self.last_reaction_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f0f0;")
        last_col.addWidget(last_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        last_col.addWidget(self.last_reaction_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        results_inner.addLayout(last_col)
        
        # Best
        best_col = QtWidgets.QVBoxLayout()
        best_title = QtWidgets.QLabel("🏆 BEST")
        best_title.setStyleSheet("font-size: 10px; color: #ff8c00;")
        self.best_attempt_label = QtWidgets.QLabel("--")
        self.best_attempt_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #ff8c00;")
        best_col.addWidget(best_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        best_col.addWidget(self.best_attempt_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        results_inner.addLayout(best_col)
        
        # Mean
        mean_col = QtWidgets.QVBoxLayout()
        mean_title = QtWidgets.QLabel("AVG")
        mean_title.setStyleSheet("font-size: 10px; color: #666;")
        self.summary_label = QtWidgets.QLabel("--")
        self.summary_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #888;")
        mean_col.addWidget(mean_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        mean_col.addWidget(self.summary_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        results_inner.addLayout(mean_col)
        
        right_col.addWidget(results_frame)
        right_col.addStretch()
        
        main_content.addLayout(right_col, stretch=1)
        layout.addLayout(main_content, stretch=1)
        
        # === BOTTOM: Coach Bar ===
        coach_bar = QtWidgets.QFrame()
        coach_bar.setFixedHeight(50)
        coach_bar.setStyleSheet("""
            background: rgba(255, 140, 0, 0.06);
            border-radius: 8px;
            border: 1px solid rgba(255, 140, 0, 0.2);
        """)
        coach_layout = QtWidgets.QHBoxLayout(coach_bar)
        coach_layout.setContentsMargins(14, 0, 14, 0)
        coach_layout.setSpacing(12)
        
        self.trash_label = QtWidgets.QLabel("💬 Ready when you are!")
        self.trash_label.setStyleSheet("font-size: 13px; color: #ff8c00; font-weight: 500;")
        coach_layout.addWidget(self.trash_label, stretch=1)
        
        # Coach quick buttons
        for text, mode in [("💡 Tip", "tip"), ("🔥 Hype", "hype"), ("😤 Taunt", "taunt")]:
            btn = QtWidgets.QPushButton(text)
            btn.setFixedSize(65, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 140, 0, 0.15);
                    color: #ff8c00;
                    border: 1px solid rgba(255, 140, 0, 0.3);
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                }
                QPushButton:hover { background: rgba(255, 140, 0, 0.25); }
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode: self._quick_coach_action(m))
            coach_layout.addWidget(btn)
        
        layout.addWidget(coach_bar)
        self.reaction_tab.setLayout(layout)
        
        # Initialize tracking
        self._reaction_attempts = []
        self._best_attempt_index = -1
        self._best_attempt_frames = []
        self.attempt_labels = []

    def _setup_punch_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)
        self._add_back_btn(layout)
        
        # Header - compact
        header = QtWidgets.QLabel("📊 PUNCH DETECTION")
        header.setStyleSheet("font-size: 16px; font-weight: 700; color: #ff8c00;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Main content - horizontal layout
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(10)
        
        # LEFT - Live Video Feed (compact)
        video_frame = QtWidgets.QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 8px;
            }
        """)
        video_layout = QtWidgets.QVBoxLayout(video_frame)
        video_layout.setContentsMargins(6, 6, 6, 6)
        video_layout.setSpacing(4)
        
        video_header = QtWidgets.QLabel("📹 GLOVE TRACKING")
        video_header.setStyleSheet("font-size: 11px; font-weight: 700; color: #ff8c00;")
        video_layout.addWidget(video_header)
        
        self.punch_preview = QtWidgets.QLabel()
        self.punch_preview.setFixedSize(380, 285)  # Fits 7" screen
        self.punch_preview.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
        """)
        self.punch_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.punch_preview.setText("⏳ Camera...")
        video_layout.addWidget(self.punch_preview)
        
        content.addWidget(video_frame, stretch=2)
        
        # RIGHT - Stats Panel (compact)
        stats_panel = QtWidgets.QFrame()
        stats_panel.setFixedWidth(260)
        stats_panel.setStyleSheet("""
            QFrame {
                background: rgba(18, 18, 18, 0.9);
                border: 1px solid #333333;
                border-radius: 8px;
            }
        """)
        stats_layout = QtWidgets.QVBoxLayout(stats_panel)
        stats_layout.setContentsMargins(10, 8, 10, 8)
        stats_layout.setSpacing(8)
        
        stats_header = QtWidgets.QLabel("⚡ LAST PUNCH")
        stats_header.setStyleSheet("font-size: 12px; font-weight: 700; color: #ff8c00;")
        stats_layout.addWidget(stats_header)
        
        self.punch_label = QtWidgets.QLabel("Waiting...")
        self.punch_label.setWordWrap(True)
        self.punch_label.setStyleSheet("""
            font-size: 13px;
            color: #f0f0f0;
            padding: 8px;
            background: #1a1a1a;
            border-radius: 6px;
            border: 1px solid #333333;
        """)
        stats_layout.addWidget(self.punch_label)
        
        # IMU Data
        imu_header = QtWidgets.QLabel("📡 IMU")
        imu_header.setStyleSheet("font-size: 12px; font-weight: 700; color: #ff8c00;")
        stats_layout.addWidget(imu_header)
        
        self.imu_label = QtWidgets.QLabel("IMU: Disabled")
        self.imu_label.setWordWrap(True)
        self.imu_label.setStyleSheet("""
            font-size: 11px;
            color: #888888;
            padding: 6px;
            background: #1a1a1a;
            border-radius: 6px;
            border: 1px solid #333333;
        """)
        self.imu_label.setVisible(True)
        stats_layout.addWidget(self.imu_label)
        
        # Punch counter - compact
        counter_frame = QtWidgets.QFrame()
        counter_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 140, 0, 0.1);
                border: 1px solid rgba(255, 140, 0, 0.3);
                border-radius: 8px;
            }
        """)
        counter_layout = QtWidgets.QVBoxLayout(counter_frame)
        counter_layout.setContentsMargins(8, 6, 8, 6)
        counter_layout.setSpacing(2)
        
        self.punch_counter_label = QtWidgets.QLabel("TOTAL PUNCHES")
        self.punch_counter_label.setStyleSheet("font-size: 10px; color: #ff8c00; font-weight: 600;")
        self.punch_counter_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        counter_layout.addWidget(self.punch_counter_label)
        
        self.punch_count_display = QtWidgets.QLabel("0")
        self.punch_count_display.setStyleSheet("font-size: 36px; font-weight: 800; color: #ff8c00;")
        self.punch_count_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        counter_layout.addWidget(self.punch_count_display)
        
        stats_layout.addWidget(counter_frame)
        stats_layout.addStretch()
        
        content.addWidget(stats_panel)
        
        layout.addLayout(content, stretch=1)
        self.punch_tab.setLayout(layout)

    def _setup_calibration_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)
        self._add_back_btn(layout)
        
        header = QtWidgets.QLabel("🎯 HSV CALIBRATION")
        header.setStyleSheet("font-size: 16px; font-weight: 700; color: #ff8c00;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        self.calib_status = QtWidgets.QLabel("Adjust HSV and Apply")
        self.calib_status.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self.calib_status)

        self.green_sliders = self._create_hsv_group("Green (Left)")
        self.red1_sliders = self._create_hsv_group("Red (Right) - Range 1")
        self.red2_sliders = self._create_hsv_group("Red (Right) - Range 2")

        # Compact button row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        
        self.apply_btn = QtWidgets.QPushButton("✓ Apply")
        self.apply_btn.setFixedHeight(36)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000;
                font-weight: 700;
                font-size: 12px;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        self.save_btn = QtWidgets.QPushButton("💾 Save YAML")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #ff8c00;
                font-weight: 600;
                font-size: 12px;
                border: 1px solid #ff8c00;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover { background: #333; }
        """)
        self.apply_btn.clicked.connect(self._apply_hsv)
        self.save_btn.clicked.connect(self._save_yaml)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()

        layout.addWidget(self.green_sliders["group"])
        layout.addWidget(self.red1_sliders["group"])
        layout.addWidget(self.red2_sliders["group"])
        layout.addLayout(btn_row)
        layout.addStretch()

        self.calib_tab.setLayout(layout)

    def _setup_llm_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(8)
        self._add_back_btn(layout)
        
        # Header
        header = QtWidgets.QLabel("💬 AI COACH")
        header.setStyleSheet("font-size: 16px; font-weight: 800; color: #ff8c00;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Main content: Chat area (left) + Quick buttons (right column)
        content_row = QtWidgets.QHBoxLayout()
        content_row.setSpacing(10)
        
        # Left side - Chat response area
        self.llm_response = QtWidgets.QTextEdit()
        self.llm_response.setReadOnly(True)
        self.llm_response.setPlaceholderText("Ask your AI coach...")
        self.llm_response.setStyleSheet("""
            QTextEdit {
                background: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
                color: #f0f0f0;
            }
        """)
        content_row.addWidget(self.llm_response, stretch=2)
        
        # Right side - 3 Quick buttons in a vertical column
        quick_col = QtWidgets.QVBoxLayout()
        quick_col.setSpacing(8)
        
        quick_btns = [
            ("🔥 MOTIVATE", "Give me intense motivation to push through!"),
            ("💡 TIP", "Give me one practical boxing tip"),
            ("😤 TRASH TALK", "Hit me with competitive trash talk!"),
        ]
        
        for text, prompt in quick_btns:
            btn = QtWidgets.QPushButton(text)
            btn.setFixedSize(100, 50)
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2a2a2a, stop:1 #1a1a1a);
                    color: #ff8c00;
                    border: 2px solid #ff8c00;
                    border-radius: 10px;
                    font-size: 11px;
                    font-weight: 700;
                }
                QPushButton:hover { 
                    background: #ff8c00;
                    color: #000000;
                }
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, p=prompt: self._quick_llm_prompt(p))
            quick_col.addWidget(btn)
        
        quick_col.addStretch()
        content_row.addLayout(quick_col)
        
        layout.addLayout(content_row, stretch=1)
        
        # Input row at bottom
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(8)
        
        self.llm_prompt = QtWidgets.QLineEdit()
        self.llm_prompt.setPlaceholderText("Type your question...")
        self.llm_prompt.setFixedHeight(38)
        self.llm_prompt.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                font-size: 12px;
                border-radius: 8px;
                border: 1px solid #333333;
                background: #1a1a1a;
            }
            QLineEdit:focus { border-color: #ff8c00; }
        """)
        self.llm_prompt.returnPressed.connect(self._send_llm_prompt)
        input_row.addWidget(self.llm_prompt, stretch=1)
        
        # Hidden mode selector (default to coach)
        self.llm_mode = QtWidgets.QComboBox()
        self.llm_mode.addItems(["coach", "encourage", "trash", "analysis"])
        self.llm_mode.hide()
        
        self.llm_send = QtWidgets.QPushButton("Send 📤")
        self.llm_send.setFixedSize(70, 38)
        self.llm_send.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 12px;
                font-weight: 700;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        self.llm_send.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.llm_send.clicked.connect(self._send_llm_prompt)
        input_row.addWidget(self.llm_send)
        
        layout.addLayout(input_row)
        
        self.llm_tab.setLayout(layout)
    
    def _quick_llm_prompt(self, prompt: str):
        """Send a quick pre-defined prompt to LLM."""
        self.llm_prompt.setText(prompt)
        self._send_llm_prompt()
    
    def _quick_coach_action(self, mode: str):
        """Quick coach action from reaction drill page - shows result in trash_label."""
        prompts = {
            "tip": "Give me one short boxing tip in 10 words or less",
            "hype": "Give me short intense motivation in 10 words or less", 
            "taunt": "Give me playful trash talk in 10 words or less",
        }
        prompt = prompts.get(mode, "Give me a quick boxing tip")
        
        # Show loading state
        self.trash_label.setText("💬 Coach: Thinking...")
        
        # Send LLM request asynchronously
        def do_request():
            if not self.ros.llm_client.service_is_ready():
                return "Coach is warming up... try again!"
            req = GenerateLLM.Request()
            req.mode = "coach"
            req.prompt = prompt
            future = self.ros.llm_client.call_async(req)
            rclpy.spin_until_future_complete(self.ros, future, timeout_sec=8.0)
            if future.result() is not None:
                return future.result().response
            return "No response - coach is busy!"
        
        # Run in thread to not block UI
        def run_and_update():
            response = do_request()
            # Update UI from main thread
            QtCore.QMetaObject.invokeMethod(
                self.trash_label, "setText",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, f"💬 Coach: {response}")
            )
        
        threading.Thread(target=run_and_update, daemon=True).start()
    
    def _setup_shadow_tab(self) -> None:
        """Setup shadow sparring drill tab - with camera feed."""
        outer_layout = QtWidgets.QVBoxLayout(self.shadow_tab)
        outer_layout.setContentsMargins(12, 8, 12, 8)
        outer_layout.setSpacing(6)
        self._add_back_btn(outer_layout)
        
        # Content - horizontal layout
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(12)
        
        # === LEFT: Camera Feed ===
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)
        
        video_frame = QtWidgets.QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 2px solid #222;
                border-radius: 10px;
            }
        """)
        video_inner = QtWidgets.QVBoxLayout(video_frame)
        video_inner.setContentsMargins(6, 6, 6, 6)
        video_inner.setSpacing(4)
        
        video_header = QtWidgets.QHBoxLayout()
        shadow_video_title = QtWidgets.QLabel("🥊 SHADOW SPARRING")
        shadow_video_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #ff8c00;")
        video_header.addWidget(shadow_video_title)
        video_header.addStretch()
        self.shadow_video_status = QtWidgets.QLabel("● LIVE")
        self.shadow_video_status.setStyleSheet("font-size: 10px; font-weight: 700; color: #00cc00;")
        video_header.addWidget(self.shadow_video_status)
        video_inner.addLayout(video_header)
        
        self.shadow_preview = QtWidgets.QLabel()
        self.shadow_preview.setFixedSize(380, 260)
        self.shadow_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shadow_preview.setText("⏳ Connecting...")
        self.shadow_preview.setStyleSheet("""
            background: #000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #555;
            font-size: 13px;
        """)
        video_inner.addWidget(self.shadow_preview)
        
        left_col.addWidget(video_frame)
        left_col.addStretch()
        content.addLayout(left_col)
        
        # === RIGHT: Controls & Action Display ===
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(8)
        
        # Action prediction card - prominent display
        self.action_card = QtWidgets.QFrame()
        self.action_card.setFixedHeight(100)
        self.action_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0d0d0d);
                border: 2px solid #333;
                border-radius: 12px;
            }
        """)
        ac_layout = QtWidgets.QVBoxLayout(self.action_card)
        ac_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ac_layout.setContentsMargins(10, 8, 10, 8)
        
        self.action_label = QtWidgets.QLabel("READY")
        self.action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_label.setStyleSheet("font-size: 36px; font-weight: 800; color: #ff8c00; background: transparent;")
        ac_layout.addWidget(self.action_label)
        
        self.action_conf_label = QtWidgets.QLabel("Confidence: --%")
        self.action_conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_conf_label.setStyleSheet("font-size: 11px; color: #888; background: transparent;")
        ac_layout.addWidget(self.action_conf_label)
        
        right_col.addWidget(self.action_card)
        
        # Drill controls row
        controls_frame = QtWidgets.QFrame()
        controls_frame.setStyleSheet("background: #151515; border-radius: 8px; border: 1px solid #282828;")
        controls_inner = QtWidgets.QHBoxLayout(controls_frame)
        controls_inner.setContentsMargins(12, 10, 12, 10)
        controls_inner.setSpacing(10)
        
        combo_label = QtWidgets.QLabel("Combo:")
        combo_label.setStyleSheet("font-weight: 700; font-size: 12px; color: #ff8c00;")
        controls_inner.addWidget(combo_label)
        
        self.shadow_combo = QtWidgets.QComboBox()
        self.shadow_combo.addItems([
            "1-1-2", "Jab-Cross-Hook", "Double Jab",
            "Cross-Hook-Cross", "4 Punch", "Uppercut"
        ])
        self.shadow_combo.setFixedWidth(130)
        self.shadow_combo.setStyleSheet("font-size: 12px; padding: 6px;")
        controls_inner.addWidget(self.shadow_combo)
        
        self.shadow_start_btn = QtWidgets.QPushButton("▶ START")
        self.shadow_start_btn.setFixedSize(85, 34)
        self.shadow_start_btn.clicked.connect(self._start_shadow_drill)
        self.shadow_start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 12px;
                font-weight: 700;
                border-radius: 6px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        controls_inner.addWidget(self.shadow_start_btn)
        
        right_col.addWidget(controls_frame)
        
        # Progress info
        progress_frame = QtWidgets.QFrame()
        progress_frame.setStyleSheet("background: #151515; border-radius: 8px; border: 1px solid #282828;")
        prog_layout = QtWidgets.QGridLayout(progress_frame)
        prog_layout.setSpacing(6)
        prog_layout.setContentsMargins(12, 10, 12, 10)
        
        self.shadow_progress_label = QtWidgets.QLabel("Step: -/-")
        self.shadow_progress_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #ff8c00;")
        self.shadow_expected_label = QtWidgets.QLabel("Next: --")
        self.shadow_expected_label.setStyleSheet("font-size: 12px; color: #ffa333;")
        self.shadow_elapsed_label = QtWidgets.QLabel("Time: 0.0s")
        self.shadow_elapsed_label.setStyleSheet("font-size: 12px; color: #888;")
        self.shadow_status_label = QtWidgets.QLabel("Status: idle")
        self.shadow_status_label.setStyleSheet("font-size: 12px; color: #666;")
        
        prog_layout.addWidget(self.shadow_progress_label, 0, 0)
        prog_layout.addWidget(self.shadow_expected_label, 0, 1)
        prog_layout.addWidget(self.shadow_elapsed_label, 1, 0)
        prog_layout.addWidget(self.shadow_status_label, 1, 1)
        
        right_col.addWidget(progress_frame)
        
        # Sequence display
        self.shadow_sequence_label = QtWidgets.QLabel("Sequence: --")
        self.shadow_sequence_label.setWordWrap(True)
        self.shadow_sequence_label.setStyleSheet("""
            font-size: 12px;
            color: #8b949e;
            padding: 10px;
            background: #151515;
            border-radius: 8px;
            border: 1px solid #282828;
        """)
        right_col.addWidget(self.shadow_sequence_label)
        
        # Checkbox progress indicator
        self.shadow_checkbox_progress = CheckboxProgressWidget(count=5)
        right_col.addWidget(self.shadow_checkbox_progress)
        
        right_col.addStretch()
        content.addLayout(right_col, stretch=1)
        
        outer_layout.addLayout(content, stretch=1)

    
    def _setup_defence_tab(self) -> None:
        """Setup defence drill tab - with camera feed."""
        outer_layout = QtWidgets.QVBoxLayout(self.defence_tab)
        outer_layout.setContentsMargins(12, 8, 12, 8)
        outer_layout.setSpacing(6)
        self._add_back_btn(outer_layout)
        
        # Content - horizontal layout
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(12)
        
        # === LEFT: Camera Feed ===
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)
        
        video_frame = QtWidgets.QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 2px solid #222;
                border-radius: 10px;
            }
        """)
        video_inner = QtWidgets.QVBoxLayout(video_frame)
        video_inner.setContentsMargins(6, 6, 6, 6)
        video_inner.setSpacing(4)
        
        video_header = QtWidgets.QHBoxLayout()
        defence_video_title = QtWidgets.QLabel("🛡️ DEFENCE DRILL")
        defence_video_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #ff8c00;")
        video_header.addWidget(defence_video_title)
        video_header.addStretch()
        self.defence_video_status = QtWidgets.QLabel("● LIVE")
        self.defence_video_status.setStyleSheet("font-size: 10px; font-weight: 700; color: #00cc00;")
        video_header.addWidget(self.defence_video_status)
        video_inner.addLayout(video_header)
        
        self.defence_preview = QtWidgets.QLabel()
        self.defence_preview.setFixedSize(380, 260)
        self.defence_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_preview.setText("⏳ Connecting...")
        self.defence_preview.setStyleSheet("""
            background: #000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #555;
            font-size: 13px;
        """)
        video_inner.addWidget(self.defence_preview)
        
        left_col.addWidget(video_frame)
        left_col.addStretch()
        content.addLayout(left_col)
        
        # === RIGHT: Controls & Block Indicator ===
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(8)
        
        # Block indicator - prominent display
        self.block_indicator = QtWidgets.QFrame()
        self.block_indicator.setFixedHeight(100)
        self.block_indicator.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0d0d0d);
                border: 2px solid #333;
                border-radius: 12px;
            }
        """)
        bi_layout = QtWidgets.QVBoxLayout(self.block_indicator)
        bi_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        bi_layout.setContentsMargins(10, 8, 10, 8)
        
        self.defence_action_label = QtWidgets.QLabel("READY")
        self.defence_action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_action_label.setStyleSheet("font-size: 36px; font-weight: 800; color: #888; background: transparent;")
        bi_layout.addWidget(self.defence_action_label)
        
        self.defence_sub_label = QtWidgets.QLabel("Dodge incoming attacks!")
        self.defence_sub_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_sub_label.setStyleSheet("font-size: 11px; color: #555; background: transparent;")
        bi_layout.addWidget(self.defence_sub_label)
        
        right_col.addWidget(self.block_indicator)
        
        # Drill controls row
        controls_frame = QtWidgets.QFrame()
        controls_frame.setStyleSheet("background: #151515; border-radius: 8px; border: 1px solid #282828;")
        controls_inner = QtWidgets.QHBoxLayout(controls_frame)
        controls_inner.setContentsMargins(12, 10, 12, 10)
        controls_inner.setSpacing(10)
        
        attacks_label = QtWidgets.QLabel("Attacks:")
        attacks_label.setStyleSheet("font-weight: 700; font-size: 12px; color: #ff8c00;")
        controls_inner.addWidget(attacks_label)
        
        self.defence_count_spin = QtWidgets.QSpinBox()
        self.defence_count_spin.setRange(5, 30)
        self.defence_count_spin.setValue(10)
        self.defence_count_spin.setFixedWidth(70)
        self.defence_count_spin.setStyleSheet("font-size: 12px; padding: 6px;")
        controls_inner.addWidget(self.defence_count_spin)
        
        self.defence_start_btn = QtWidgets.QPushButton("▶ START")
        self.defence_start_btn.setFixedSize(85, 34)
        self.defence_start_btn.clicked.connect(self._start_defence_drill)
        self.defence_start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 12px;
                font-weight: 700;
                border-radius: 6px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        controls_inner.addWidget(self.defence_start_btn)
        
        right_col.addWidget(controls_frame)
        
        # Progress info
        progress_frame = QtWidgets.QFrame()
        progress_frame.setStyleSheet("background: #151515; border-radius: 8px; border: 1px solid #282828;")
        prog_layout = QtWidgets.QGridLayout(progress_frame)
        prog_layout.setSpacing(6)
        prog_layout.setContentsMargins(12, 10, 12, 10)
        
        self.defence_progress_label = QtWidgets.QLabel("Dodges: 0/0")
        self.defence_progress_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #ff8c00;")
        self.defence_elapsed_label = QtWidgets.QLabel("Time: 0.0s")
        self.defence_elapsed_label.setStyleSheet("font-size: 12px; color: #888;")
        self.defence_status_label = QtWidgets.QLabel("Status: idle")
        self.defence_status_label.setStyleSheet("font-size: 12px; color: #666;")
        
        prog_layout.addWidget(self.defence_progress_label, 0, 0)
        prog_layout.addWidget(self.defence_elapsed_label, 0, 1)
        prog_layout.addWidget(self.defence_status_label, 1, 0, 1, 2)
        
        right_col.addWidget(progress_frame)
        
        # Checkbox progress indicator
        self.defence_checkbox_progress = CheckboxProgressWidget(count=5)
        right_col.addWidget(self.defence_checkbox_progress)
        
        right_col.addStretch()
        content.addLayout(right_col, stretch=1)
        
        outer_layout.addLayout(content, stretch=1)

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
            # Use debug image if raw color not available (live_infer_rgbd publishes to debug topic)
            display_img = color_img if color_img is not None else img
            countdown = self.ros.drill_countdown
            punch_counter = self.ros.punch_counter

        # Update cue panel styling based on state - ORANGE/BLACK THEME
        if state == "cue":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff8c00, stop:1 #cc7000);
                border-radius: 10px;
                border: 2px solid #ffa333;
            """)
            self.state_label.setText("⚡ PUNCH!")
            self.state_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #000000; border: none; background: transparent;")
            self.countdown_label.setStyleSheet("font-size: 11px; color: #000000; border: none; background: transparent;")
        elif state == "waiting":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 140, 0, 0.3), stop:1 rgba(200, 110, 0, 0.3));
                border-radius: 10px;
                border: 2px solid #ff8c00;
            """)
            self.state_label.setText("GET READY...")
            self.state_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #ff8c00; border: none; background: transparent;")
            self.countdown_label.setStyleSheet("font-size: 11px; color: #888888; border: none; background: transparent;")
        elif state == "countdown":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(200, 50, 50, 0.8), stop:1 rgba(150, 30, 30, 0.8));
                border-radius: 10px;
                border: 2px solid #cc3333;
            """)
            self.state_label.setText("STEADY...")
            self.state_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #ff6666; border: none; background: transparent;")
            self.countdown_label.setStyleSheet("font-size: 11px; color: #ffaaaa; border: none; background: transparent;")
        elif state == "baseline":
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(80, 80, 200, 0.6), stop:1 rgba(50, 50, 150, 0.6));
                border-radius: 10px;
                border: 2px solid #6666cc;
            """)
            self.state_label.setText("STAY STILL")
            self.state_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #9999ff; border: none; background: transparent;")
            self.countdown_label.setStyleSheet("font-size: 11px; color: #aaaaff; border: none; background: transparent;")
        else:
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(18, 18, 18, 0.95), stop:1 rgba(10, 10, 10, 0.95));
                border-radius: 10px;
                border: 2px solid #333333;
            """)
            self.state_label.setText("READY")
            self.state_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #ff8c00; border: none; background: transparent;")
            self.countdown_label.setStyleSheet("font-size: 11px; color: #888888; border: none; background: transparent;")

        if state == "countdown":
            self.countdown_label.setText(f"Countdown: {countdown}")
        elif state == "baseline":
            self.countdown_label.setText("Capturing baseline...")
        else:
            self.countdown_label.setText("Press START to begin")

        # Update attempt tracking
        last_rt = summary.get("last_reaction_time_s") if isinstance(summary, dict) else None
        mean_rt = summary.get("mean_reaction_time_s") if isinstance(summary, dict) else None
        best_rt = summary.get("best_reaction_time_s") if isinstance(summary, dict) else None
        reaction_times = summary.get("reaction_times", []) if isinstance(summary, dict) else []
        
        # Update individual attempt labels
        if hasattr(self, 'attempt_labels'):
            for i, lbl in enumerate(self.attempt_labels):
                if i < len(reaction_times):
                    rt = reaction_times[i]
                    is_best = (best_rt is not None and abs(rt - best_rt) < 0.001)
                    if is_best:
                        lbl.setText(f"Attempt {i+1}: {rt:.3f}s 🏆")
                        lbl.setStyleSheet("font-size: 16px; color: #ff8c00; font-weight: 700; padding: 4px 0;")
                    else:
                        lbl.setText(f"Attempt {i+1}: {rt:.3f}s")
                        lbl.setStyleSheet("font-size: 16px; color: #f0f0f0; padding: 4px 0;")
                else:
                    lbl.setText(f"Attempt {i+1}: --")
                    lbl.setStyleSheet("font-size: 16px; color: #555555; padding: 4px 0;")
        
        if hasattr(self, 'best_attempt_label'):
            if best_rt is not None:
                self.best_attempt_label.setText(f"{best_rt:.3f}s")
            else:
                self.best_attempt_label.setText("--")
        
        self.last_reaction_label.setText(f"{last_rt:.3f}s" if last_rt is not None else "--")
        self.summary_label.setText(f"{mean_rt:.3f}s" if mean_rt is not None else "--")
        
        # Update trash talk
        if trash:
            self.trash_label.setText(f"💬 {trash}")
        else:
            self.trash_label.setText("💬 Ready when you are!")

        # IMU display
        if imu and self.imu_input_enabled:
            self.imu_label.setText(
                f"ax={imu.ax:.2f}  ay={imu.ay:.2f}  az={imu.az:.2f}\ngx={imu.gx:.2f}  gy={imu.gy:.2f}  gz={imu.gz:.2f}"
            )
            self.imu_label.setStyleSheet("""
                font-size: 14px;
                color: #ff8c00;
                padding: 12px;
                background: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #ff8c00;
            """)
        else:
            self.imu_label.setText("IMU: Disabled (enable in Experimental Features)")
            self.imu_label.setStyleSheet("""
                font-size: 14px;
                color: #555555;
                padding: 12px;
                background: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #333333;
            """)

        # Punch info
        if punch:
            glove_emoji = "🥊" if punch.glove == "left" else "🥋"
            punch_type = punch.punch_type or "unknown"
            self.punch_label.setText(
                f"{glove_emoji} {punch.glove.upper()} - {punch_type.upper()}\n"
                f"Velocity: {punch.approach_velocity_mps:.2f} m/s\n"
                f"Distance: {punch.distance_m:.2f} m"
            )

        # Update punch counter display
        if hasattr(self, 'punch_count_display'):
            self.punch_count_display.setText(str(punch_counter))

        # Update video previews
        if img is not None:
            qimg = self._to_qimage(img)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.punch_preview.setPixmap(pix.scaled(self.punch_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        if display_img is not None:
            # First frame received - update status
            if not self._camera_received:
                self._camera_received = True
                self.video_status_label.setText("● LIVE")
                self.video_status_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #00ff00;")
            
            now = time.time()
            self._frame_buffer.append((now, display_img.copy()))
            qimg2 = self._to_qimage(display_img)
            pix2 = QtGui.QPixmap.fromImage(qimg2)
            self.reaction_preview.setPixmap(
                pix2.scaled(self.reaction_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            )
            
            # Update shadow and defence previews too
            if hasattr(self, 'shadow_preview'):
                self.shadow_preview.setPixmap(
                    pix2.scaled(self.shadow_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                )
            if hasattr(self, 'defence_preview'):
                self.defence_preview.setPixmap(
                    pix2.scaled(self.defence_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                )
        else:
            # No camera feed yet
            if not self._camera_received:
                self.video_status_label.setText("● CONNECTING...")
                self.video_status_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #ffaa00;")

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
        req.drill_name = "Random" # self.defence_combo removed
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

    def _start_height_calibration(self) -> None:
        # Use countdown splash for the 3s wait
        self.stack.setCurrentWidget(self.shadow_countdown)
        self.shadow_countdown.set_status("Stand Straight for Height Calibration...")
        try:
            self.shadow_countdown.countdown_finished.disconnect()
        except Exception:
            pass
        self.shadow_countdown.countdown_finished.connect(self._trigger_height_calc)
        self.shadow_countdown.start(3)

    def _trigger_height_calc(self) -> None:
        self.stack.setCurrentWidget(self.home_screen)
        if self.ros.height_trigger_client.service_is_ready():
             self.ros.height_trigger_client.call_async(Trigger.Request())
             QtWidgets.QMessageBox.information(self, "Height", "Calibration request sent! Check logs/status.")
        else:
             QtWidgets.QMessageBox.warning(self, "Height", "Height service not ready")

def main() -> None:
    rclpy.init()
    ros_node = RosInterface()

    app = QtWidgets.QApplication([])
    gui = BoxBunnyGui(ros_node)
    gui.show()

    ros_thread = RosSpinThread(ros_node)
    ros_thread.start()

    exit_code = app.exec()

    # Clean shutdown
    ros_thread.quit()
    ros_thread.wait()
    ros_node.destroy_node()
    rclpy.shutdown()
    
    import sys
    sys.exit(exit_code)



if __name__ == "__main__":
    main()
