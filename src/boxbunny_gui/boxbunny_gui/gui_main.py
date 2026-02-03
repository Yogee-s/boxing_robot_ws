import json
import os
import threading
import time
from collections import deque
from typing import Optional, List
import csv # Added by user instruction

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from std_msgs.msg import String, Int32, Bool, Float32
from std_srvs.srv import SetBool, Trigger
from std_srvs.srv import SetBool
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from boxbunny_msgs.msg import GloveDetections, PunchEvent, ImuDebug, TrashTalk, ActionPrediction, DrillProgress
from boxbunny_msgs.srv import StartStopDrill, GenerateLLM, StartDrill

# Fix for Qt Platform Plugin "xcb" error (OpenCV vs PySide6 conflict)
# Must set this BEFORE importing PySide6
import sys
if "boxing_ai" in sys.executable:
    # Point to Conda's Qt plugins (avoiding OpenCV's bundled Qt)
    conda_p = os.path.dirname(sys.executable)
    # Different possible locations depending on OS/Conda layout
    possible_roots = [
        os.path.join(conda_p, "../lib/qt6/plugins"),
        os.path.join(conda_p, "../plugins"),
        os.path.join(conda_p, "Library/plugins") # Windows
    ]
    for p in possible_roots:
        if os.path.exists(os.path.join(p, "platforms/libqxcb.so")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = p
            # Unset plugin path to prevent conflicting lookups
            if "QT_PLUGIN_PATH" in os.environ:
                del os.environ["QT_PLUGIN_PATH"]
            break

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
        font_size=48, padding=36, min_width=120, min_height=100,
        bg_color="#2196F3", hover_color="#42A5F5", pressed_color="#1565C0",
        border_radius=18
    )

    # Start button - Teal accent
    START = _create_style.__func__(
        font_size=22, padding=20, min_width=180, min_height=60,
        bg_color="#26d0ce", hover_color="#3ae0de", pressed_color="#1a7f7e",
    )

    # Large countdown style
    COUNTDOWN_LABEL = """
        QLabel {
            font-size: 150px;
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
    """Visual progress tracker with numbered step indicators and punch labels."""
    
    def __init__(self, count: int = 3, labels: list = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        self.count = count
        self.current = 0
        self.checkboxes = []
        self.punch_labels = labels or []
        
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setSpacing(4)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        # Checkboxes row
        checkbox_row = QtWidgets.QHBoxLayout()
        checkbox_row.setSpacing(16)
        checkbox_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        for i in range(count):
            # Container for each punch box
            punch_container = QtWidgets.QVBoxLayout()
            punch_container.setSpacing(4)
            punch_container.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            # Main checkbox
            checkbox = QtWidgets.QLabel(f"{i+1}")
            checkbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            checkbox.setFixedSize(52, 52)
            checkbox.setStyleSheet("""
                font-size: 22px;
                font-weight: 700;
                color: #484f58;
                background: #1a1a1a;
                border: 3px solid #333;
                border-radius: 10px;
            """)
            punch_container.addWidget(checkbox, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.checkboxes.append(checkbox)
            
            # Punch type label below
            label_text = self.punch_labels[i] if i < len(self.punch_labels) else ""
            punch_label = QtWidgets.QLabel(label_text.upper() if label_text else "")
            punch_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            punch_label.setStyleSheet("font-size: 10px; font-weight: 600; color: #666; background: transparent;")
            punch_container.addWidget(punch_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            
            checkbox_row.addLayout(punch_container)
        
        outer_layout.addLayout(checkbox_row)
    
    def set_labels(self, labels: list):
        """Update punch labels."""
        self.punch_labels = labels
        # Find and update label widgets
        for i, checkbox in enumerate(self.checkboxes):
            parent_layout = checkbox.parent()
            if parent_layout and i < len(labels):
                # Labels are stored as the second widget in each punch_container
                pass  # Labels are set at creation; for dynamic update, recreate widget
    
    def tick(self, index: int = None):
        """Tick the checkbox at the given index (or next if None)."""
        if index is None:
            index = self.current
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].setText("✓")
            self.checkboxes[index].setStyleSheet("""
                font-size: 22px;
                font-weight: 700;
                color: #000;
                background: #26d0ce;
                border: 3px solid #26d0ce;
                border-radius: 10px;
            """)
            self.current = index + 1
    
    def reset(self):
        """Reset all checkboxes to empty."""
        self.current = 0
        for i, checkbox in enumerate(self.checkboxes):
            checkbox.setText(f"{i+1}")
            checkbox.setStyleSheet("""
                font-size: 22px;
                font-weight: 700;
                color: #484f58;
                background: #1a1a1a;
                border: 3px solid #333;
                border-radius: 10px;
            """)
    
    def set_wrong(self, index: int):
        """Mark a checkbox as wrong (red X)."""
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].setText("✗")
            self.checkboxes[index].setStyleSheet("""
                font-size: 22px;
                font-weight: 700;
                color: #fff;
                background: #ff4757;
                border: 3px solid #ff4757;
                border-radius: 10px;
            """)
            self.current = index + 1


class ComboHistoryWidget(QtWidgets.QWidget):
    """Visual tracker for combo history (success/wrong combos)."""
    
    def __init__(self, max_count: int = 3, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        self.max_count = max_count
        self.history = []  # List of 'success' or 'wrong'
        self.boxes = []
        
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setSpacing(4)
        outer_layout.setContentsMargins(6, 0, 6, 0)
        outer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Small label
        label = QtWidgets.QLabel("HISTORY")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 10px; font-weight: 600; color: #666; background: transparent;")
        outer_layout.addWidget(label)
        
        # Boxes row
        self.boxes_row = QtWidgets.QHBoxLayout()
        self.boxes_row.setSpacing(4)
        self.boxes_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        for i in range(max_count):
            box = QtWidgets.QLabel("")
            box.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            box.setFixedSize(30, 30)
            box.setStyleSheet("""
                font-size: 14px;
                font-weight: 700;
                color: #444;
                background: #1a1a1a;
                border: 2px solid #333;
                border-radius: 4px;
            """)
            self.boxes_row.addWidget(box)
            self.boxes.append(box)
        
        outer_layout.addLayout(self.boxes_row)
    
    def add_result(self, result: str):
        """Add a combo result ('success' or 'wrong')."""
        self.history.append(result)
        # Keep only the last max_count results
        if len(self.history) > self.max_count:
            self.history = self.history[-self.max_count:]
        self._update_display()
    
    def _update_display(self):
        """Update the visual display of history."""
        for i, box in enumerate(self.boxes):
            if i < len(self.history):
                result = self.history[i]
                if result == 'success':
                    box.setText("✓")
                    box.setStyleSheet("""
                        font-size: 14px;
                        font-weight: 700;
                        color: #000;
                        background: #26d0ce;
                        border: 2px solid #26d0ce;
                        border-radius: 4px;
                    """)
                else:  # wrong
                    box.setText("✗")
                    box.setStyleSheet("""
                        font-size: 14px;
                        font-weight: 700;
                        color: #fff;
                        background: #ff4757;
                        border: 2px solid #ff4757;
                        border-radius: 4px;
                    """)
            else:
                box.setText("")
                box.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 700;
                    color: #444;
                    background: #1a1a1a;
                    border: 2px solid #333;
                    border-radius: 4px;
                """)
    
    def reset(self):
        """Clear history."""
        self.history = []
        self._update_display()


# ============================================================================
# COACH BAR WIDGET (Reusable LLM Chat Bar)
# ============================================================================

class CoachBarWidget(QtWidgets.QFrame):
    """Reusable AI Coach bar with message display and quick action buttons."""
    
    def __init__(self, ros_interface, parent=None):
        super().__init__(parent)
        self.ros = ros_interface
        
        self.setMinimumHeight(90)
        self.setStyleSheet("""
            QFrame {
                background: rgba(255, 140, 0, 0.1);
                border-radius: 12px;
                border: 2px solid rgba(255, 140, 0, 0.3);
            }
        """)

        # Connect streaming callback
        self.ros.add_stream_listener(self._on_stream_data)
        self._received_stream = False
        self._streaming_text = ""
        self._stream_target = "coach_bar"
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)
        
        # Coach icon - bigger
        icon_label = QtWidgets.QLabel()
        icon_label.setText("🤖")
        icon_label.setStyleSheet("font-size: 36px; background: transparent; border: none;")
        layout.addWidget(icon_label)
        
        # Message label - takes up available space
        self.message_label = QtWidgets.QLabel("Tap a button for coaching tips!")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("""
            font-size: 16px; 
            color: #ff8c00; 
            font-weight: 600;
            background: transparent;
            border: none;
            padding: 8px;
            line-height: 1.4;
        """)
        layout.addWidget(self.message_label, stretch=1)
        
        # Quick action buttons - bigger for touch
        self.buttons = {}
        btn_info = [
            ("TIP", "tip", "#00ee88"),
            ("HYPE", "hype", "#ffaa00"),
            ("FOCUS", "focus", "#44aaff"),
        ]
        for label_text, mode, color in btn_info:
            btn = QtWidgets.QPushButton(label_text)
            btn.setMinimumWidth(80)
            btn.setMinimumHeight(50)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1a1a1a;
                    color: {color};
                    border: 2px solid {color};
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 10px 16px;
                }}
                QPushButton:hover {{ 
                    background-color: {color};
                    color: #000000;
                }}
                QPushButton:pressed {{
                    background-color: {color};
                }}
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            btn.clicked.connect(lambda checked, m=mode: self._request_coach(m))
            layout.addWidget(btn)
            self.buttons[mode] = btn
    
    def set_message(self, text: str):
        """Set the coach message."""
        self.message_label.setText(text)
    
    def _request_coach(self, mode: str):
        """Request a coach response from LLM with varied prompts."""
        import random
        
        # IMPORTANT: These prompts must NOT reference user performance data
        # They are general advice, not feedback on specific training
        prompt_variations = {
            "tip": [
                "Give ONE specific boxing technique tip in under 15 words. No greeting.",
                "Name one defensive drill. Under 12 words.",
                "One footwork tip. Under 10 words.",
                "Best way to improve jab speed? Under 12 words.",
                "Common hook mistake and fix? Under 15 words.",
            ],
            "hype": [
                "Give me ONE LINE of intense motivation. No questions, just fire me up!",
                "Channel a boxing legend - one powerful motivational line only.",
                "Make me feel unstoppable in one sentence. Be intense!",
                "One line to get my heart pumping for training.",
                "Champion mindset quote. One powerful line.",
            ],
            "focus": [
                "One calming breath instruction. Under 12 words.",
                "A short mantra for focus. Under 8 words.",
                "Mental reset cue in one sentence.",
                "How to clear the mind before a round? One line.",
                "Visualization tip for boxers. Under 15 words.",
            ],
        }
        
        prompts = prompt_variations.get(mode, prompt_variations["tip"])
        prompt = random.choice(prompts)
        
        # Reset any previous stream and show loading state
        if hasattr(self, "_stream_timer") and self._stream_timer.isActive():
            self._stream_timer.stop()
        self._received_stream = False
        self._streaming_text = ""
        self.message_label.setText("🤔 Thinking...")
        
        # Check if service is ready
        if not self.ros.llm_client.service_is_ready():
            self.message_label.setText("⚠️ Coach not available - start the LLM service")
            return
        
        # Make the async request
        req = GenerateLLM.Request()
        req.mode = "coach"
        req.prompt = prompt
        future = self.ros.llm_client.call_async(req)
        
        # Reset stream state
        self._received_stream = False
        self._streaming_text = ""
        
        # Add callback for when response arrives
        self.ros.stream_target = self._stream_target
        future.add_done_callback(self._on_coach_response)
        
    def _on_stream_data(self, text: str):
        """Handle incoming stream token."""
        if getattr(self.ros, "stream_target", None) != self._stream_target:
            return
        # Update on main thread
        QtCore.QMetaObject.invokeMethod(
            self, "_update_stream",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text)
        )

    @QtCore.Slot(str)
    def _update_stream(self, text: str):
        """Update display with new token."""
        # Check if we need to clear the "Thinking..." message
        # We do this if it's the first token OR if the current text is still the loading message
        current_text = self.message_label.text()
        if not self._received_stream or "Thinking" in current_text:
            self._received_stream = True
            self.message_label.setText("")
            self._streaming_text = ""
        
        self._streaming_text += text
        self.message_label.setText(self._streaming_text)
        
    def _on_coach_response(self, future):
        """Handle LLM response callback - called from ROS thread."""
        try:
            result = future.result()
            if result is not None and result.response:
                response = result.response
            else:
                response = "⚠️ No response - check if Ollama is running"
        except Exception as e:
            response = f"⚠️ Error: {str(e)[:30]}"
        
        # Only use fallback display if we didn't receive a stream
        if not self._received_stream:
            # Start streaming the text character by character (fake stream for non-LLM responses)
            QtCore.QMetaObject.invokeMethod(
                self, "_start_stream",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, response)
            )
        self.ros.stream_target = None
    
    @QtCore.Slot(str)
    def _start_stream(self, text: str):
        """Start streaming text to the message label."""
        self._stream_text = text
        self._stream_index = 0
        self._current_display = ""
        
        # Create timer if needed
        if not hasattr(self, '_stream_timer'):
            self._stream_timer = QtCore.QTimer(self)
            self._stream_timer.timeout.connect(self._stream_next_chars)
        
        # Clear and start streaming - show chars quickly
        self.message_label.setText("")
        self._stream_timer.start(25)  # 25ms per chunk for fast but visible streaming
    
    def _stream_next_chars(self):
        """Add next chunk of characters to display."""
        if self._stream_index >= len(self._stream_text):
            self._stream_timer.stop()
            self.ros.stream_target = None
            return
        
        # Add 2-3 characters at a time for smoother effect
        chunk_size = 3
        end_idx = min(self._stream_index + chunk_size, len(self._stream_text))
        self._current_display += self._stream_text[self._stream_index:end_idx]
        self._stream_index = end_idx
        
        self.message_label.setText(self._current_display)


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
        self._llm_warmed = False
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Title - bigger for impact
        title = QtWidgets.QLabel("🥊 BOXBUNNY")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: 800;
            color: #ff8c00;
            background: transparent;
            letter-spacing: 4px;
        """)
        layout.addWidget(title)
        
        # Loading spinner/status
        self.status_label = QtWidgets.QLabel("⏳ Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 18px;
            color: #888888;
            background: transparent;
        """)
        layout.addWidget(self.status_label)
        
        # Status items - bigger text
        status_frame = QtWidgets.QFrame()
        status_frame.setMaximumWidth(350)
        status_layout = QtWidgets.QVBoxLayout(status_frame)
        status_layout.setSpacing(12)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_status = QtWidgets.QLabel("⏳ Camera: Connecting...")
        self.camera_status.setStyleSheet("font-size: 16px; color: #666666; padding: 6px;")
        status_layout.addWidget(self.camera_status)
        
        self.llm_status = QtWidgets.QLabel("⏳ AI Coach: Connecting...")
        self.llm_status.setStyleSheet("font-size: 16px; color: #666666; padding: 6px;")
        status_layout.addWidget(self.llm_status)
        
        layout.addWidget(status_frame, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Skip button (appears after timeout) - bigger and more prominent
        self.skip_btn = QtWidgets.QPushButton("Skip & Continue →")
        self.skip_btn.setMinimumSize(180, 50)
        self.skip_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 140, 0, 0.2);
                color: #ff8c00;
                font-size: 16px;
                font-weight: 600;
                padding: 12px 28px;
                border: 2px solid #ff8c00;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: rgba(255, 140, 0, 0.4);
            }
            QPushButton:pressed {
                background: rgba(255, 140, 0, 0.6);
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
        # pose mode publishes to /action_debug_image (last_pose_image) - CRITICAL FIX
        with self.ros.lock:
            has_camera = (self.ros.last_image is not None or 
                          self.ros.last_color_image is not None or 
                          self.ros.last_color_image_fallback is not None or
                          self.ros.last_pose_image is not None)
            
        if has_camera and not self.camera_ready:
            self.camera_ready = True
            self.camera_status.setText("✅ Camera: Ready")
            self.camera_status.setStyleSheet("font-size: 16px; color: #00ff00; padding: 6px; font-weight: 600;")
        
        # Check LLM service
        llm_available = self.ros.llm_client.service_is_ready()
        if llm_available and not self.llm_ready:
            self.llm_ready = True
            self.llm_status.setText("✅ AI Coach: Ready")
            self.llm_status.setStyleSheet("font-size: 16px; color: #00ff00; padding: 6px; font-weight: 600;")
            self._warm_llm()
        
        # Update main status
        if self.camera_ready and self.llm_ready:
            self.status_label.setText("✅ All systems ready!")
            self.status_label.setStyleSheet("font-size: 18px; color: #00ff00; background: transparent; font-weight: 600;")
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

    def _warm_llm(self) -> None:
        """Send a tiny warm-up request so the first user prompt is fast."""
        if self._llm_warmed:
            return
        if not self.ros.llm_client.service_is_ready():
            return
        self._llm_warmed = True
        req = GenerateLLM.Request()
        req.mode = "coach"
        req.prompt = "Warm-up. Reply with OK."
        try:
            future = self.ros.llm_client.call_async(req)
            future.add_done_callback(lambda _f: None)
        except Exception:
            pass


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
        layout.setSpacing(24)
        
        # Title
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 42px;
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
            font-size: 22px;
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
        self.declare_parameter("color_topic_fallback", "/camera/camera/color/image_raw")

        self.last_image = None
        self.last_color_image = None
        self.last_color_image_stamp = None
        self.last_color_image_fallback = None
        self.last_color_image_fallback_stamp = None
        self.last_pose_image = None  # For pose skeleton from action_debug_image
        self.last_detections: Optional[GloveDetections] = None
        self.last_punch: Optional[PunchEvent] = None
        self.last_imu: Optional[ImuDebug] = None
        self.last_shadow_punch: Optional[PunchEvent] = None
        self.last_shadow_punch_stamp = None
        self.shadow_punch_counter = 0
        self.drill_state = "idle"
        self.drill_summary = {}
        self.drill_countdown = 0
        self.trash_talk = ""
        self.last_punch_stamp = None
        self.punch_counter = 0
        self.robot_action_status: Optional[int] = None
        self.robot_action_status_stamp: Optional[float] = None

        punch_topic = self.get_parameter("punch_topic").value
        color_topic = self.get_parameter("color_topic").value
        color_fallback_topic = self.get_parameter("color_topic_fallback").value

        # Image topics are typically best-effort; use sensor data QoS to ensure matching.
        self.debug_sub = self.create_subscription(
            Image, "glove_debug_image", self._on_image, qos_profile_sensor_data
        )
        self.color_sub = self.create_subscription(
            Image, color_topic, self._on_color_image, qos_profile_sensor_data
        )
        if color_fallback_topic and color_fallback_topic != color_topic:
            self.color_fallback_sub = self.create_subscription(
                Image, color_fallback_topic, self._on_color_image_fallback, qos_profile_sensor_data
            )
        self.pose_sub = self.create_subscription(
            Image, "action_debug_image", self._on_pose_image, qos_profile_sensor_data
        )
        self.det_sub = self.create_subscription(GloveDetections, "glove_detections", self._on_detections, 5)
        self.punch_sub = self.create_subscription(PunchEvent, punch_topic, self._on_punch, 5)
        self.shadow_punch_sub = self.create_subscription(PunchEvent, "punch_events_raw", self._on_shadow_punch, 5)
        self.imu_sub = self.create_subscription(ImuDebug, "imu/debug", self._on_imu, 5)
        self.state_sub = self.create_subscription(String, "drill_state", self._on_state, 5)
        self.summary_sub = self.create_subscription(String, "drill_summary", self._on_summary, 5)
        self.countdown_sub = self.create_subscription(Int32, "drill_countdown", self._on_countdown, 5)
        self.trash_sub = self.create_subscription(TrashTalk, "trash_talk", self._on_trash, 5)
        self.robot_status_sub = self.create_subscription(
            String, "/robot/robot_action_status", self._on_robot_action_status, 5
        )

        self.start_stop_client = self.create_client(StartStopDrill, "start_stop_drill")
        self.llm_client = self.create_client(GenerateLLM, "llm/generate")
        self.llm_param_client = self.create_client(SetParameters, "trash_talk_node/set_parameters")
        self.shadow_drill_client = self.create_client(StartDrill, "/start_drill")
        self.shadow_stop_client = self.create_client(Trigger, "/stop_shadow_drill")
        self.defence_drill_client = self.create_client(StartDrill, "start_defence_drill")
        self.imu_input_client = self.create_client(SetBool, "imu_input_selector/enable")
        self.tracker_param_client = self.create_client(SetParameters, "realsense_glove_tracker/set_parameters")
        self.drill_param_client = self.create_client(SetParameters, "reaction_drill_manager/set_parameters")
        
        # Action prediction
        self.last_action: Optional[ActionPrediction] = None
        self.drill_progress: Optional[DrillProgress] = None
        self.imu_input_enabled = False
        self.stream_target = None
        
        self._last_processed_punch_timestamp = None
        
        self.action_sub = self.create_subscription(ActionPrediction, "action_prediction", self._on_action, 5)
        self.progress_sub = self.create_subscription(DrillProgress, "drill_progress", self._on_progress, 5)
        self.imu_enabled_sub = self.create_subscription(Bool, "imu_input_enabled", self._on_imu_enabled, 5)
        self.height_sub = self.create_subscription(Float32, "/player_height", self._on_height, 5)
        self.stream_sub = self.create_subscription(String, "llm/stream", self._on_llm_stream, 10)
        
        self.stream_callback = None
        self.stream_listeners = []
        
        # New Services
        self.mode_client = self.create_client(SetBool, "action_predictor/set_simple_mode")
        self.height_trigger_client = self.create_client(Trigger, "action_predictor/calibrate_height")
        
        # Publisher for motor commands
        self.motor_pub = self.create_publisher(String, '/robot/robot_action_trigger', 10)

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
                self.last_color_image_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception:
            pass

    def _on_color_image_fallback(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.last_color_image_fallback = img
                self.last_color_image_fallback_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception:
            pass

    def _on_pose_image(self, msg: Image) -> None:
        """Handle pose skeleton debug image from action predictor."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.last_pose_image = img
        except Exception:
            pass

    def _on_detections(self, msg: GloveDetections) -> None:
        with self.lock:
            self.last_detections = msg

    def _on_punch(self, msg: PunchEvent) -> None:
        with self.lock:
            print(f"DEBUG: GUI received punch: {msg.punch_type} at {msg.approach_velocity_mps:.2f} m/s")
            self.last_punch = msg
            self.last_punch_stamp = (msg.stamp.sec, msg.stamp.nanosec)
            self.punch_counter += 1

    def _on_shadow_punch(self, msg: PunchEvent) -> None:
        with self.lock:
            self.last_shadow_punch = msg
            self.last_shadow_punch_stamp = (msg.stamp.sec, msg.stamp.nanosec)
            if msg.is_punch:
                self.shadow_punch_counter += 1

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

    def _on_llm_stream(self, msg: String) -> None:
        if self.stream_callback:
            self.stream_callback(msg.data)
        for listener in self.stream_listeners:
            listener(msg.data)

    def add_stream_listener(self, listener) -> None:
        self.stream_listeners.append(listener)

    def _on_robot_action_status(self, msg: String) -> None:
        """Track robot action status; 0 means action complete."""
        raw = str(msg.data).strip()
        try:
            status = int(raw)
        except Exception:
            try:
                status = int(float(raw))
            except Exception:
                status = None
        with self.lock:
            self.robot_action_status = status
            self.robot_action_status_stamp = time.time()



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
        print("DEBUG: LOADING UPDATED GUI - SUCCESS/MISS TRACKER ENABLED", flush=True)
        self.ros = ros
        self.setWindowTitle("BoxBunny Trainer")
        self._is_fullscreen = False
        self._use_aspect_scaling = True
        
        # Default size for 7-inch touchscreen (1024x600), but allow resize
        self.resize(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.setMinimumSize(800, 480)
        
        self._frame_buffer = deque(maxlen=180)
        self._last_punch_counter = 0
        self._replay_frames = []
        self._replay_index = 0
        self._initialized = False
        self._camera_received = False
        self._shadow_last_preview_ts = None
        self._last_reaction_frame_ts = 0.0
        self._last_reaction_summary_key = None
        self._pending_replay_clip = None
        self._pending_replay_time = None
        self._reaction_clips = []
        self._best_reaction_clip = None
        self._replay_active = False
        self._llm_request_inflight = False

        self._apply_styles()
        
        # Main Layout container
        main_widget = QtWidgets.QWidget()
        main_widget.setStyleSheet("background: #0a0a0a;")
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header bar - integrated title with back and window controls
        self.header_frame = QtWidgets.QFrame()
        self.header_frame.setObjectName("headerFrame")
        self.header_frame.setStyleSheet("""
            QFrame#headerFrame {
                background: #0d0d0d;
                border: 3px solid #ff8c00;
                border-radius: 12px;
                margin: 8px;
                padding: 4px;
            }
        """)
        header_row = QtWidgets.QHBoxLayout(self.header_frame)
        header_row.setContentsMargins(12, 8, 12, 8)
        header_row.setSpacing(10)
        
        # Back button (left) - hidden on home screen, uses Unicode arrow
        self.header_back_btn = QtWidgets.QPushButton("◀ BACK")
        self.header_back_btn.setObjectName("headerBackBtn")
        self.header_back_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.header_back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.home_screen))
        self.header_back_btn.hide()
        header_row.addWidget(self.header_back_btn)
        
        # Spacer when back button hidden (will be sized dynamically)
        self.header_left_spacer = QtWidgets.QWidget()
        header_row.addWidget(self.header_left_spacer)
        
        # Header title - centered, expands to fill
        self.header = QtWidgets.QLabel("BOXBUNNY TRAINER")
        self.header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.header.setObjectName("header")
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        header_row.addWidget(self.header, stretch=1)
        
        # Window control button (right) - clear text label
        self.fullscreen_btn = QtWidgets.QPushButton("MAX")
        self.fullscreen_btn.setObjectName("fullscreenBtn")
        self.fullscreen_btn.setToolTip("Toggle Fullscreen (F11)")
        self.fullscreen_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        header_row.addWidget(self.fullscreen_btn)
        
        main_layout.addWidget(self.header_frame)
        self._apply_header_scale(1.0)
        
        # Store title mappings for different screens
        self._screen_titles = {}

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
        
        # Map screens to their titles
        self._screen_titles = {
            self.home_screen: "BOXBUNNY TRAINER",
            self.reaction_tab: "🎯 REACTION DRILL",
            self.shadow_tab: "🥊 SHADOW SPARRING",
            self.defence_tab: "🛡️ DEFENCE DRILL",
            self.punch_tab: "📊 PUNCH STATS",
            self.llm_tab: "💬 AI COACH",
            self.calib_tab: "⚙️ CALIBRATION",
            self.startup_screen: "BOXBUNNY TRAINER",
            self.shadow_countdown: "🥊 SHADOW SPARRING",
            self.defence_countdown: "🛡️ DEFENCE DRILL",
            self.video_replay: "🎬 VIDEO REPLAY",
        }
        
        # Connect stack change to update title
        self.stack.currentChanged.connect(self._on_screen_changed)
        
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

        self.stats_timer = QtCore.QTimer()
        self.stats_timer.timeout.connect(self._update_reaction_stats)
        self.stats_timer.start(200)

        self._attach_scaled_root(main_widget)
    
    def _on_screen_changed(self, index: int):
        """Update header title and back button when screen changes."""
        current_widget = self.stack.widget(index)
        self._update_header_for_screen(current_widget)
        self._reset_llm_outputs()
        
        # Auto-switch detection mode based on verify screen requirements
        # Reaction Drill -> Needs Pose (AI Mode)
        # Shadow/Defence -> Needs Color Tracking (Simple Mode)
        if current_widget == self.reaction_tab:
            if hasattr(self, 'action_mode_radio') and not self.action_mode_radio.isChecked():
                self.action_mode_radio.setChecked(True)
                # self._on_detection_mode_changed() # Triggered by setChecked signal?
                # Signal might not fire if not user interaction, let's force call if needed
                # But typically setChecked fires toggled.
        elif current_widget == self.shadow_tab or current_widget == self.defence_tab:
            if hasattr(self, 'color_mode_radio') and not self.color_mode_radio.isChecked():
                self.color_mode_radio.setChecked(True)
        elif current_widget == self.home_screen:
            # Revert to AI mode on home screen for general usage? Or keep last?
            # Let's default to Pose (AI) as it's the more robust default
             if hasattr(self, 'action_mode_radio') and not self.action_mode_radio.isChecked():
                self.action_mode_radio.setChecked(True)
    
    def _on_startup_complete(self):
        """Called when startup loading is complete."""
        self._initialized = True
        self._camera_received = True  # Mark camera as received since startup confirmed it
        # Update video status to show ready
        if hasattr(self, 'video_status_label'):
            self.video_status_label.setText("📹 LIVE ●")
            self.video_status_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #00ff00;")
        self.stack.setCurrentWidget(self.home_screen)

    def _reset_llm_outputs(self) -> None:
        """Clear coach/LLM outputs when switching screens."""
        if hasattr(self, 'reaction_coach_bar'):
            self.reaction_coach_bar.set_message("Tap a button for coaching tips!")
        if hasattr(self, 'shadow_coach_bar'):
            self.shadow_coach_bar.set_message("Tap a button for coaching tips!")
        if hasattr(self, 'defence_coach_bar'):
            self.defence_coach_bar.set_message("Tap a button for coaching tips!")
        if hasattr(self.ros, "stream_target"):
            self.ros.stream_target = None

    def _update_header_for_screen(self, widget):
        """Update header title and back button visibility."""
        # Show/hide back button based on screen
        is_home = (widget == self.home_screen or widget == self.startup_screen)
        self.header_back_btn.setVisible(not is_home)
        self.header_left_spacer.setVisible(is_home)  # Show spacer when back is hidden
        
        # Update title
        if widget in self._screen_titles:
            title = self._screen_titles[widget]
            self.header.setText(title)
        elif widget == self.home_screen:
            self.header.setText("BOXBUNNY TRAINER")
    
    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        if self._is_fullscreen:
            self.showNormal()
            self.fullscreen_btn.setText("MAX")
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self.fullscreen_btn.setText("EXIT")
            self._is_fullscreen = True
    
    def resizeEvent(self, event):
        """Scale the entire UI to preserve aspect ratio on resize."""
        super().resizeEvent(event)
        if not self._use_aspect_scaling or not hasattr(self, "_view"):
            w = event.size().width()
            h = event.size().height()
            scale = min(w / 800, h / 480)
            scale = max(0.6, min(scale, 2.0))
            self._apply_header_scale(scale)
            return
        self._apply_view_scale()

    def showEvent(self, event):
        """Ensure the scaled view fits correctly on first show."""
        super().showEvent(event)
        if self._use_aspect_scaling and hasattr(self, "_view"):
            self._apply_view_scale()

    def _apply_header_scale(self, scale: float) -> None:
        """Apply header sizing at a given scale factor."""
        header_h = int(80 * scale)
        header_h = max(70, min(header_h, 120))
        self.header_frame.setFixedHeight(header_h)

        title_size = int(32 * scale)
        title_size = max(24, min(title_size, 48))

        btn_size = int(18 * scale)
        btn_size = max(14, min(btn_size, 24))

        self.header.setStyleSheet(f"""
            font-size: {max(20, int(title_size * 0.9))}px;
            font-weight: 800;
            letter-spacing: {max(1, int(2 * scale))}px;
            color: #ff8c00;
            background: transparent;
            border: none;
        """)

        side_btn_w = int(140 * scale)
        side_btn_h = int(50 * scale)

        self.header_back_btn.setFixedSize(side_btn_w, side_btn_h)
        self.header_back_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: #ff8c00;
                font-size: {max(16, btn_size + 4)}px;
                font-weight: 700;
                border: 2px solid #ff8c00;
                border-radius: 8px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background: #ff8c00;
                color: #000;
            }}
        """)

        self.header_left_spacer.setFixedSize(side_btn_w, 0)

        self.fullscreen_btn.setFixedSize(side_btn_w, side_btn_h)
        self.fullscreen_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: #888;
                border: 2px solid #555;
                border-radius: 8px;
                font-size: {max(14, btn_size)}px;
                font-weight: 600;
                padding: 0px;
            }}
            QPushButton:hover {{
                color: #ff8c00;
                border-color: #ff8c00;
            }}
        """)

    def _apply_view_scale(self) -> None:
        """Scale the view to fit while preserving aspect ratio."""
        viewport = self._view.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        scale = min(viewport.width() / self.SCREEN_WIDTH,
                    viewport.height() / self.SCREEN_HEIGHT)
        transform = QtGui.QTransform()
        transform.scale(scale, scale)
        self._view.setTransform(transform)

    def _attach_scaled_root(self, root: QtWidgets.QWidget) -> None:
        """Attach the root widget, optionally scaling to preserve aspect ratio."""
        if not self._use_aspect_scaling:
            self.setCentralWidget(root)
            return

        root.setFixedSize(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        self._scene = QtWidgets.QGraphicsScene(self)
        self._proxy = self._scene.addWidget(root)
        self._scene.setSceneRect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        self._view = QtWidgets.QGraphicsView(self._scene)
        self._view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                                 QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setBackgroundBrush(QtGui.QColor("#0a0a0a"))
        self._view.setStyleSheet("background: #0a0a0a;")
        self._view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCentralWidget(self._view)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == QtCore.Qt.Key.Key_F11:
            self._toggle_fullscreen()
        elif event.key() == QtCore.Qt.Key.Key_Escape and self._is_fullscreen:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def _apply_styles(self):
        self.setStyleSheet("""
            /* ===== ORANGE & BLACK BOXING THEME ===== */
            
            /* Main Window - Pure black */
            QMainWindow {
                background: #0a0a0a;
            }
            
            /* Ensure stacked widget has no margins */
            QStackedWidget {
                background: #0a0a0a;
                border: none;
                margin: 0px;
                padding: 0px;
            }
            
            /* Base typography */
            QLabel {
                color: #eaeaea;
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 16px;
                background: transparent;
                border: none;
            }
            
            /* Header - Bold orange accent - responsive sizing */
            QLabel#header {
                font-size: 18px;
                font-weight: 800;
                letter-spacing: 2px;
                color: #ff8c00;
                padding: 4px 8px;
                background: transparent;
                border: none;
            }
            
            /* Header back button - dynamic sizing */
            QPushButton#headerBackBtn {
                background: transparent;
                color: #ff8c00;
                font-size: 12px;
                font-weight: 700;
                border: 2px solid #ff8c00;
                border-radius: 8px;
                padding: 12px 42px;
                min-width: 130px;
                margin: 4px;
            }
            QPushButton#headerBackBtn:hover {
                background: #ff8c00;
                color: #000;
            }
            
            /* Fullscreen button - dynamic sizing */
            QPushButton#fullscreenBtn {
                background: transparent;
                color: #888;
                border: 2px solid #555;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                padding: 12px 42px;
                min-width: 100px;
                margin: 4px;
            }
            QPushButton#fullscreenBtn:hover {
                color: #ff8c00;
                border-color: #ff8c00;
            }
            
            /* Cards - Dark with orange accents */
            QFrame, QGroupBox {
                background-color: rgba(22, 22, 22, 0.95);
                border-radius: 16px;
                border: 1px solid rgba(255, 140, 0, 0.22);
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
                border: 2px solid #2f2f2f;
                background: #1f1f1f;
                color: #f5f5f5;
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
                background: #1f1f1f;
                color: #f5f5f5;
                selection-background-color: #ff8c00;
                border: 1px solid #2f2f2f;
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
                background: #ff8c00;
                border-color: #ff8c00;
                image: url(data:image/x-xpm;base64,LyogWFBNICovCnN0YXRpYyBjaGFyICogY2hlY2tfeHBtW10gPSB7CiIxMiAxMiAyIDEiLAoiICBjIE5vbmUiLAoiLiBjICMwMDAwMDAiLAoiICAgICAgICAgICAgIiwKIiAgICAgICAgICAgICIsCiIgICAgICAgICAgICAiLAoiICAgICAgICAgICAuIiwKIiAgICAgICAgICAuICIsCiIgICAgICAgICAuICAiLAoiICAgICAgICAuICAgIiwKIiAuICAgICAuICAgICIsCiIgIC4gICAuICAgICAiLAoiICAgLiAuICAgICAgIiwKIiAgICAuICAgICAgICIsCiIgICAgICAgICAgICAifTsK);
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
        layout.setContentsMargins(20, 12, 20, 10)
        layout.setSpacing(8)
        
        # === MAIN CONTENT ROW (Horizontal) ===
        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(20)
        
        # Add stretch to center the left column when right is hidden
        content_row.addStretch(1)
        
        # --- LEFT COLUMN: Buttons ---
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(10)
        left_col.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Center everything vertically in left col
        left_col.addStretch(1)
        
        # Drills Buttons
        drills_container = QtWidgets.QWidget()
        drills_container.setFixedWidth(500)  # Fixed width for stability
        drills_layout = QtWidgets.QVBoxLayout(drills_container)
        drills_layout.setSpacing(10)
        drills_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_reaction = self._create_menu_btn_centered("🎯  REACTION", self.reaction_tab)
        btn_shadow = self._create_menu_btn_centered("🥊  SHADOW", self.shadow_tab)
        btn_defence = self._create_menu_btn_centered("🛡️  DEFENCE", self.defence_tab)
        
        for btn in [btn_reaction, btn_shadow, btn_defence]:
            btn.setMinimumHeight(80)
            btn.setMaximumHeight(95)
            drills_layout.addWidget(btn)
        
        left_col.addWidget(drills_container)
        
        left_col.addSpacing(14)
        
        # Quick Access Buttons
        quick_container = QtWidgets.QWidget()
        quick_container.setFixedWidth(500)
        quick_row = QtWidgets.QHBoxLayout(quick_container)
        quick_row.setSpacing(12)
        quick_row.setContentsMargins(0, 0, 0, 0)
        
        btn_stats = self._create_quick_btn("📊", "STATS", self.punch_tab)
        btn_llm = self._create_quick_btn("💬", "COACH", self.llm_tab)
        btn_calib = self._create_quick_btn("⚙️", "SETUP", self.calib_tab)
        
        for btn in [btn_stats, btn_llm, btn_calib]:
            quick_row.addWidget(btn)
        
        left_col.addWidget(quick_container)
        
        left_col.addSpacing(8)
        
        # Advanced Toggle
        self.advanced_btn = QtWidgets.QPushButton("⚗️ ADVANCED ▾")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setFixedWidth(500)
        self.advanced_btn.setMinimumHeight(42)
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #555555;
                border: 2px solid #333333;
                font-size: 14px;
                font-weight: 600;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover { color: #ff8c00; border-color: #ff8c00; }
            QPushButton:checked { color: #ff8c00; border-color: #ff8c00; background: rgba(255,140,0,0.1); }
        """)
        self.advanced_btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        left_col.addWidget(self.advanced_btn)
        
        left_col.addStretch(1)
        
        content_row.addLayout(left_col)
        
        # Spacer between columns
        self.col_spacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        content_row.addItem(self.col_spacer)
        
        # --- RIGHT COLUMN: Advanced Panel ---
        self.right_col_container = QtWidgets.QWidget()
        self.right_col_container.setFixedWidth(300)
        self.right_col_container.setVisible(False) # Hidden by default
        right_col_layout = QtWidgets.QVBoxLayout(self.right_col_container)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.advanced_panel = QtWidgets.QFrame()
        self.advanced_panel.setStyleSheet("""
            QFrame {
                background: rgba(30, 30, 30, 0.98);
                border-radius: 12px;
                border: 2px solid #ff8c00;
            }
        """)
        adv_panel_layout = QtWidgets.QVBoxLayout(self.advanced_panel)
        adv_panel_layout.setContentsMargins(16, 14, 16, 14)
        adv_panel_layout.setSpacing(12)
        
        # Detection mode section (Vertical now to fit sidebar)
        mode_label = QtWidgets.QLabel("Detection:")
        mode_label.setStyleSheet("font-size: 16px; color: #ff8c00; font-weight: 700;")
        adv_panel_layout.addWidget(mode_label)
        
        self.color_mode_radio = QtWidgets.QRadioButton("Color Mode")
        self.color_mode_radio.setChecked(True)
        self.color_mode_radio.setStyleSheet("""
            QRadioButton {
                font-size: 15px; color: #e6edf3; spacing: 8px; padding: 4px;
            }
            QRadioButton::indicator {
                width: 20px; height: 20px; border-radius: 10px; border: 2px solid #555; background: #1a1a1a;
            }
            QRadioButton::indicator:checked { border: 2px solid #ff8c00; background: #ff8c00; }
        """)
        self.color_mode_radio.toggled.connect(self._on_detection_mode_changed)
        adv_panel_layout.addWidget(self.color_mode_radio)
        
        self.action_mode_radio = QtWidgets.QRadioButton("AI Mode")
        self.action_mode_radio.setStyleSheet("""
            QRadioButton {
                font-size: 15px; color: #e6edf3; spacing: 8px; padding: 4px;
            }
            QRadioButton::indicator {
                width: 20px; height: 20px; border-radius: 10px; border: 2px solid #555; background: #1a1a1a;
            }
            QRadioButton::indicator:checked { border: 2px solid #ff8c00; background: #ff8c00; }
        """)
        adv_panel_layout.addWidget(self.action_mode_radio)
        
        # Separator
        sep = QtWidgets.QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #444;")
        adv_panel_layout.addWidget(sep)
        
        self.imu_toggle = QtWidgets.QCheckBox("Enable IMU Input")
        self.imu_toggle.setStyleSheet("""
            QCheckBox {
                font-size: 15px; color: #e6edf3; spacing: 8px; padding: 4px;
            }
            QCheckBox::indicator {
                width: 20px; height: 20px; border-radius: 4px; border: 2px solid #555; background: #1a1a1a;
            }
            QCheckBox::indicator:checked { border: 2px solid #ff8c00; background: #ff8c00; }
        """)
        self.imu_toggle.toggled.connect(self._toggle_imu_input)
        adv_panel_layout.addWidget(self.imu_toggle)
        
        adv_panel_layout.addSpacing(10)
        
        self.height_btn = QtWidgets.QPushButton("Calibrate Height")
        self.height_btn.setFixedHeight(45)
        self.height_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00; color: #000; font-size: 14px; font-weight: 700; border-radius: 8px;
            }
            QPushButton:hover { background: #ffa333; }
            QPushButton:pressed { background: #cc7000; }
        """)
        self.height_btn.clicked.connect(self._start_height_calibration)
        adv_panel_layout.addWidget(self.height_btn)
        
        right_col_layout.addWidget(self.advanced_panel)
        right_col_layout.addStretch(1) # Push to top
        
        content_row.addWidget(self.right_col_container)
        content_row.addStretch(1)
        
        layout.addLayout(content_row)
        
        # Status indicator
        self.status_indicator = QtWidgets.QLabel("● Ready")
        self.status_indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_indicator.setStyleSheet("font-size: 14px; color: #00cc00; font-weight: 600; padding: 6px;")
        layout.addWidget(self.status_indicator)
        
        self.home_screen.setLayout(layout)
    
    def _create_menu_btn_centered(self, title: str, target_widget):
        """Create a centered menu button."""
        btn = QtWidgets.QPushButton(title)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff8c00, stop:1 #e67300);
                color: #000000;
                font-size: 28px;
                font-weight: 800;
                padding: 24px 48px;
                border-radius: 18px;
                border: 2px solid rgba(255, 255, 255, 0.15);
                letter-spacing: 3px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffa333, stop:1 #ff8c00);
                border: 2px solid rgba(255, 255, 255, 0.4);
            }
            QPushButton:pressed {
                background: #cc7000;
                padding-top: 26px;
                padding-bottom: 22px;
            }
        """)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(target_widget))
        return btn
    
    def _create_quick_btn(self, icon: str, title: str, target_widget):
        """Create a quick access button with icon and title."""
        btn = QtWidgets.QPushButton(f"{icon}\n{title}")
        btn.setMinimumHeight(140)
        btn.setMinimumWidth(140)
        # Use Minimum policy to preventing shrinking below min-height
        btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #151515);
                color: #ff8c00;
                font-size: 20px;
                font-weight: 700;
                padding: 10px 20px;
                border-radius: 14px;
                border: 2px solid #333333;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1e1e1e);
                border-color: #ff8c00;
            }
            QPushButton:pressed {
                background: #151515;
                border-color: #ffa333;
            }
        """)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
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

    def _toggle_advanced(self, checked: bool) -> None:
        if hasattr(self, 'right_col_container'):
            self.right_col_container.setVisible(checked)
        self.advanced_panel.setVisible(checked)
        self.advanced_btn.setText("⚗️ Advanced ▴" if checked else "⚗️ Advanced ▾")
    
    def _toggle_imu_input(self, enabled: bool) -> None:
        """Toggle IMU input for punch detection."""
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
        """This is now a no-op - back button is in header. Kept for compatibility."""
        # Back button is now in the header row, shown/hidden via _update_header_for_screen
        pass

    def _setup_reaction_tab(self) -> None:
        """Reaction drill - clean aesthetic layout for 7" touchscreen."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)
        self._add_back_btn(layout)
        
        # Main content area - horizontal split
        main_content = QtWidgets.QHBoxLayout()
        main_content.setSpacing(10)
        
        # === LEFT COLUMN: Camera Feed ===
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(4)
        
        # Add stretch at top to center camera vertically
        left_col.addStretch(1)
        
        # Video container with header
        video_frame = QtWidgets.QFrame()
        video_frame.setFixedSize(420, 340)
        video_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 2px solid #222;
                border-radius: 8px;
            }
        """)
        video_inner = QtWidgets.QVBoxLayout(video_frame)
        video_inner.setContentsMargins(4, 4, 4, 4)
        video_inner.setSpacing(4)
        
        # Video header row
        video_header = QtWidgets.QHBoxLayout()
        self.video_status_label = QtWidgets.QLabel("📹 LIVE")
        self.video_status_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #00cc00; padding: 4px;")
        video_header.addWidget(self.video_status_label)
        video_header.addStretch()
        self.replay_btn = QtWidgets.QPushButton("🔄 Replay")
        self.replay_btn.setFixedHeight(36)
        self.replay_btn.setStyleSheet("background: #222; color: #ff8c00; border-radius: 8px; font-size: 15px; padding: 6px 14px;")
        self.replay_btn.clicked.connect(self._start_replay)
        video_header.addWidget(self.replay_btn)
        video_inner.addLayout(video_header)
        
        # Video preview
        self.reaction_preview = QtWidgets.QLabel()
        self.reaction_preview.setFixedSize(400, 300)
        self.reaction_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.reaction_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.reaction_preview.setText("⏳ Connecting...")
        self.reaction_preview.setStyleSheet("""
            background: #000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #555;
            font-size: 13px;
        """)
        self.reaction_preview.setScaledContents(False)
        video_inner.addWidget(self.reaction_preview, stretch=1)
        
        self.replay_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.replay_speed.setRange(5, 30)
        self.replay_speed.setValue(12)
        self.replay_speed.setVisible(False)
        
        left_col.addWidget(video_frame)
        left_col.addStretch(1)
        main_content.addLayout(left_col)
        
        # === RIGHT COLUMN: Controls ===
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(10)
        
        # Add stretch at top to center content vertically
        right_col.addStretch(1)
        
        # Cue Panel - prominent status display
        self.cue_panel = QtWidgets.QFrame()
        self.cue_panel.setMinimumHeight(96)
        self.cue_panel.setMaximumHeight(120)
        self.cue_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.cue_panel.setStyleSheet("""
            background: transparent;
            border: none;
        """)
        cue_layout = QtWidgets.QVBoxLayout(self.cue_panel)
        cue_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cue_layout.setContentsMargins(8, 16, 8, 12)
        cue_layout.setSpacing(4)
        
        self.state_label = QtWidgets.QLabel("READY")
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("font-size: 34px; font-weight: 800; color: #ff8c00; background: transparent;")
        
        self.countdown_label = QtWidgets.QLabel("Throw a punch on cue")
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 13px; color: #ffa333; background: transparent;")
        
        cue_layout.addWidget(self.state_label)
        cue_layout.addWidget(self.countdown_label)
        right_col.addWidget(self.cue_panel)
        
        # Control Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.start_btn = QtWidgets.QPushButton("▶  START")
        self.start_btn.setMinimumHeight(56)
        self.start_btn.setMinimumWidth(120)
        self.start_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000;
                font-size: 18px;
                font-weight: 700;
                border-radius: 10px;
                padding: 14px 20px;
            }
            QPushButton:hover { background: #ffa333; }
            QPushButton:pressed { background: #cc7000; }
        """)
        self.start_btn.clicked.connect(self._start_drill)
        
        self.stop_btn = QtWidgets.QPushButton("⬛  STOP")
        self.stop_btn.setMinimumHeight(56)
        self.stop_btn.setMinimumWidth(120)
        self.stop_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #888;
                font-size: 18px;
                font-weight: 700;
                border-radius: 10px;
                border: 1px solid #333;
                padding: 14px 20px;
            }
            QPushButton:hover { background: #333; color: #fff; }
            QPushButton:pressed { background: #222; }
        """)
        self.stop_btn.clicked.connect(self._stop_drill)
        
        btn_row.addWidget(self.start_btn, stretch=1)
        btn_row.addWidget(self.stop_btn, stretch=1)
        right_col.addLayout(btn_row)
        
        # Individual Attempt Timings - show all 3 attempts
        attempts_frame = QtWidgets.QFrame()
        attempts_frame.setStyleSheet("""
            background: #151515;
            border-radius: 6px;
            border: 1px solid #282828;
        """)
        attempts_layout = QtWidgets.QHBoxLayout(attempts_frame)
        attempts_layout.setContentsMargins(10, 8, 10, 8)
        attempts_layout.setSpacing(6)
        
        # Create 3 attempt displays
        self.attempt_labels = []
        for i in range(3):
            attempt_col = QtWidgets.QVBoxLayout()
            attempt_col.setSpacing(2)
            
            title = QtWidgets.QLabel(f"#{i+1}")
            title.setStyleSheet("font-size: 15px; color: #666; font-weight: 600;")
            title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            time_label = QtWidgets.QLabel("--")
            time_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #555;")
            time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            time_label.setMinimumWidth(75)
            
            attempt_col.addWidget(title)
            attempt_col.addWidget(time_label)
            attempts_layout.addLayout(attempt_col, stretch=1)
            self.attempt_labels.append(time_label)
        
        # Separator
        sep = QtWidgets.QFrame()
        sep.setFixedWidth(2)
        sep.setStyleSheet("background: #333;")
        attempts_layout.addWidget(sep)
        
        # Best time display
        best_col = QtWidgets.QVBoxLayout()
        best_col.setSpacing(2)
        best_title = QtWidgets.QLabel("🏆 BEST")
        best_title.setStyleSheet("font-size: 15px; color: #ff8c00; font-weight: 600;")
        best_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.best_attempt_label = QtWidgets.QLabel("--")
        self.best_attempt_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #ff8c00;")
        self.best_attempt_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.best_attempt_label.setMinimumWidth(75)
        best_col.addWidget(best_title)
        best_col.addWidget(self.best_attempt_label)
        attempts_layout.addLayout(best_col, stretch=1)
        
        right_col.addWidget(attempts_frame)
        
        # Keep legacy labels for compatibility
        self.last_reaction_label = QtWidgets.QLabel("--")
        self.last_reaction_label.hide()
        self.summary_label = QtWidgets.QLabel("--")
        self.summary_label.hide()
        
        # === Compact Stats Row ===
        stats_row = QtWidgets.QFrame()
        stats_row.setStyleSheet("""
            QFrame {
                background: #151515;
                border-radius: 8px;
                border: 1px solid #282828;
            }
        """)
        stats_inner = QtWidgets.QHBoxLayout(stats_row)
        stats_inner.setContentsMargins(14, 10, 14, 10)
        stats_inner.setSpacing(20)
        
        self.total_attempts_label = QtWidgets.QLabel("Attempts: 0")
        self.total_attempts_label.setStyleSheet("font-size: 16px; color: #888;")
        self.avg_reaction_label = QtWidgets.QLabel("Avg: --")
        self.avg_reaction_label.setStyleSheet("font-size: 16px; color: #888;")
        self.session_best_label = QtWidgets.QLabel("Best: --")
        self.session_best_label.setStyleSheet("font-size: 16px; color: #26d0ce; font-weight: 600;")
        
        stats_inner.addWidget(self.total_attempts_label)
        stats_inner.addWidget(self.avg_reaction_label)
        stats_inner.addWidget(self.session_best_label)
        stats_inner.addStretch()
        
        right_col.addWidget(stats_row)
        
        # Add stretch at bottom to center content vertically
        right_col.addStretch(1)
        
        main_content.addLayout(right_col, stretch=1)
        layout.addLayout(main_content, stretch=1)
        
        # === BOTTOM: Coach Bar - tall and prominent ===
        self.reaction_coach_bar = CoachBarWidget(self.ros)
        self.reaction_coach_bar.setMinimumHeight(100)
        self.reaction_coach_bar.setMaximumHeight(140)
        layout.addWidget(self.reaction_coach_bar)
        
        # Keep trash_label reference for backward compatibility
        self.trash_label = self.reaction_coach_bar.message_label
        
        self.reaction_tab.setLayout(layout)
        
        # Initialize tracking
        self._reaction_attempts = []
        self._best_attempt_index = -1
        self._best_attempt_frames = []
        # self.attempt_labels is already populated above

    def _setup_punch_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(6)
        self._add_back_btn(layout)
        
        # Header - compact
        header = QtWidgets.QLabel("📊 PUNCH DETECTION")
        header.setStyleSheet("font-size: 18px; font-weight: 700; color: #ff8c00; padding: 6px;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Main content - horizontal layout
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(8)
        
        # LEFT - Live Video Feed (compact)
        video_frame = QtWidgets.QFrame()
        video_frame.setMinimumWidth(340)
        video_frame.setMaximumWidth(450)
        video_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 8px;
            }
        """)
        video_layout = QtWidgets.QVBoxLayout(video_frame)
        video_layout.setContentsMargins(4, 4, 4, 4)
        video_layout.setSpacing(4)
        
        video_header = QtWidgets.QLabel("📹 GLOVE TRACKING")
        video_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #ff8c00; padding: 4px;")
        video_layout.addWidget(video_header)
        
        self.punch_preview = QtWidgets.QLabel()
        self.punch_preview.setMinimumSize(320, 240)
        self.punch_preview.setMaximumSize(420, 320)
        self.punch_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.punch_preview.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
        """)
        self.punch_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.punch_preview.setText("⏳ Camera...")
        video_layout.addWidget(self.punch_preview, stretch=1)
        
        content.addWidget(video_frame, stretch=2)
        
        # RIGHT - Stats Panel (compact)
        stats_panel = QtWidgets.QFrame()
        stats_panel.setMinimumWidth(220)
        stats_panel.setMaximumWidth(280)
        stats_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        stats_panel.setStyleSheet("""
            QFrame {
                background: rgba(18, 18, 18, 0.9);
                border: 1px solid #333333;
                border-radius: 8px;
            }
        """)
        stats_layout = QtWidgets.QVBoxLayout(stats_panel)
        stats_layout.setContentsMargins(8, 6, 8, 6)
        stats_layout.setSpacing(6)
        
        stats_header = QtWidgets.QLabel("⚡ LAST PUNCH")
        stats_header.setStyleSheet("font-size: 15px; font-weight: 700; color: #ff8c00; padding: 4px;")
        stats_layout.addWidget(stats_header)
        
        self.punch_label = QtWidgets.QLabel("Waiting...")
        self.punch_label.setWordWrap(True)
        self.punch_label.setStyleSheet("""
            font-size: 15px;
            color: #f0f0f0;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 8px;
            border: 1px solid #333333;
        """)
        stats_layout.addWidget(self.punch_label)
        
        # IMU Data
        imu_header = QtWidgets.QLabel("📡 IMU")
        imu_header.setStyleSheet("font-size: 15px; font-weight: 700; color: #ff8c00; padding: 4px;")
        stats_layout.addWidget(imu_header)
        
        self.imu_label = QtWidgets.QLabel("IMU: Disabled")
        self.imu_label.setWordWrap(True)
        self.imu_label.setStyleSheet("""
            font-size: 14px;
            color: #888888;
            padding: 8px;
            background: #1a1a1a;
            border-radius: 8px;
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
                border-radius: 6px;
            }
        """)
        counter_layout = QtWidgets.QVBoxLayout(counter_frame)
        counter_layout.setContentsMargins(10, 8, 10, 8)
        counter_layout.setSpacing(2)
        
        self.punch_counter_label = QtWidgets.QLabel("TOTAL PUNCHES")
        self.punch_counter_label.setStyleSheet("font-size: 13px; color: #ff8c00; font-weight: 600;")
        self.punch_counter_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        counter_layout.addWidget(self.punch_counter_label)
        
        self.punch_count_display = QtWidgets.QLabel("0")
        self.punch_count_display.setStyleSheet("font-size: 42px; font-weight: 800; color: #ff8c00;")
        self.punch_count_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        counter_layout.addWidget(self.punch_count_display)
        
        stats_layout.addWidget(counter_frame)
        stats_layout.addStretch()
        
        content.addWidget(stats_panel)
        
        layout.addLayout(content, stretch=1)
        self.punch_tab.setLayout(layout)

    def _setup_calibration_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(6)
        self._add_back_btn(layout)
        
        header = QtWidgets.QLabel("🎯 HSV CALIBRATION")
        header.setStyleSheet("font-size: 18px; font-weight: 700; color: #ff8c00; padding: 6px;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        self.calib_status = QtWidgets.QLabel("Adjust HSV and Apply")
        self.calib_status.setStyleSheet("font-size: 13px; color: #cfcfcf; padding: 4px;")
        layout.addWidget(self.calib_status)

        self.green_sliders = self._create_hsv_group("Green (Left)")
        self.red1_sliders = self._create_hsv_group("Red (Right) - Range 1")
        self.red2_sliders = self._create_hsv_group("Red (Right) - Range 2")

        # Compact button row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        
        self.apply_btn = QtWidgets.QPushButton("✓ Apply")
        self.apply_btn.setFixedHeight(42)
        self.apply_btn.setMinimumWidth(100)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000;
                font-weight: 700;
                font-size: 14px;
                border-radius: 8px;
                padding: 6px 20px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        self.save_btn = QtWidgets.QPushButton("💾 Save YAML")
        self.save_btn.setFixedHeight(42)
        self.save_btn.setMinimumWidth(110)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #ff8c00;
                font-weight: 600;
                font-size: 14px;
                border: 2px solid #ff8c00;
                border-radius: 8px;
                padding: 6px 20px;
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
        
        # Header handled by main app bar; avoid duplicate title here.
        
        # Main content area with stacked layout for dynamic transitions
        self.llm_content_stack = QtWidgets.QStackedWidget()
        
        # === PAGE 1: Initial prompt selection (big buttons) ===
        self.llm_prompt_page = QtWidgets.QWidget()
        prompt_layout = QtWidgets.QVBoxLayout(self.llm_prompt_page)
        prompt_layout.setContentsMargins(20, 16, 20, 16)
        prompt_layout.setSpacing(16)
        
        prompt_header = QtWidgets.QLabel("TAP TO GET COACH ADVICE")
        prompt_header.setStyleSheet("font-size: 17px; color: #888; font-weight: 600; padding: 6px;")
        prompt_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        prompt_layout.addWidget(prompt_header)
        
        prompt_layout.addStretch()
        
        # Big prompt buttons in a centered grid
        btn_grid = QtWidgets.QHBoxLayout()
        btn_grid.setSpacing(14)
        btn_grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Store prompt variations for varied responses
        self._llm_prompt_variations = {
            "motivate": [
                "Fire me up for my training session! Give me that championship mindset in 2-3 powerful sentences.",
                "I need pre-fight energy right now! Channel the intensity of a title bout. Be specific and inspiring.",
                "Motivate me like a legendary corner man. Make me feel unstoppable. Keep it punchy and powerful.",
                "Give me that warrior spirit in 2-3 sentences. I'm about to push my limits.",
                "Light that fire in my soul. Make me believe I can take on anyone. Be bold and direct.",
            ],
            "tip": [
                "Give me one specific boxing technique tip I can practice right now. Be actionable and explain why it works.",
                "What separates good boxers from great ones? Give me one key insight with a practical drill.",
                "Share a defensive technique that could save me in a tough round. Be specific about body positioning.",
                "How do pros generate knockout power? Give me one technique tip with the physics behind it.",
                "What's the most underrated boxing skill? Tell me how to develop it in my next training session.",
            ],
            "focus": [
                "Help me get into the zone. Give me a mental technique for laser focus before training.",
                "How do champions stay calm under pressure? Share a mindfulness tip for fighters.",
                "I need to clear my head and focus. Guide me with a calming breathing technique.",
                "What visualization technique helps boxers perform at their peak? Give me something I can use now.",
                "Give me a quick breathing exercise to center myself. Walk me through it step by step.",
            ],
        }
        
        self.llm_quick_btns = []
        quick_prompts = [
            ("💪", "MOTIVATE", "motivate", "#ff6b00"),
            ("💡", "TIP", "tip", "#00cc66"),
            ("🎯", "FOCUS", "focus", "#44aaff"),
        ]
        
        for emoji, label, prompt, color in quick_prompts:
            # Single tall button with emoji + label stacked inside
            btn = QtWidgets.QPushButton(f"{emoji}\n{label}")
            btn.setMinimumSize(120, 120)
            btn.setMaximumSize(150, 150)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2a2a2a, stop:1 #1a1a1a);
                    color: {color};
                    border: 3px solid {color};
                    border-radius: 18px;
                    font-size: 18px;
                    font-weight: 700;
                    padding: 12px;
                }}
                QPushButton:hover {{ 
                    background: {color};
                    color: #000000;
                }}
                QPushButton:pressed {{
                    background: {color};
                    border-color: #fff;
                }}
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            btn.clicked.connect(lambda checked, p=prompt: self._quick_llm_prompt(p))
            
            self.llm_quick_btns.append(btn)
            btn_grid.addWidget(btn)
        
        prompt_layout.addLayout(btn_grid)
        prompt_layout.addStretch()
        
        self.llm_content_stack.addWidget(self.llm_prompt_page)
        
        # === PAGE 2: Loading state ===
        self.llm_loading_page = QtWidgets.QWidget()
        loading_layout = QtWidgets.QVBoxLayout(self.llm_loading_page)
        loading_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.llm_loading_label = QtWidgets.QLabel("🤔 Coach is thinking...")
        self.llm_loading_label.setStyleSheet("font-size: 22px; color: #ff8c00; font-weight: 600; padding: 10px;")
        self.llm_loading_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.llm_loading_label)
        
        self.llm_content_stack.addWidget(self.llm_loading_page)
        
        # === PAGE 3: Response display with small side buttons ===
        self.llm_response_page = QtWidgets.QWidget()
        response_layout = QtWidgets.QHBoxLayout(self.llm_response_page)
        response_layout.setContentsMargins(10, 10, 10, 10)
        response_layout.setSpacing(12)
        
        # Main response area
        self.llm_response = QtWidgets.QTextEdit()
        self.llm_response.setReadOnly(True)
        self.llm_response.setStyleSheet("""
            QTextEdit {
                background: #1a1a1a;
                border: 2px solid #ff8c00;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                line-height: 1.5;
                color: #f0f0f0;
            }
        """)
        response_layout.addWidget(self.llm_response, stretch=1)
        
        # Small side buttons column
        self.llm_side_btns = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(self.llm_side_btns)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)
        
        self.llm_small_btns = []
        for emoji, label, prompt, color in quick_prompts:
            btn = QtWidgets.QPushButton(f"{emoji}")
            btn.setMinimumSize(48, 48)
            btn.setMaximumSize(56, 56)
            btn.setToolTip(label)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(255, 140, 0, 0.15);
                    color: {color};
                    border: 2px solid {color};
                    border-radius: 12px;
                    font-size: 20px;
                }}
                QPushButton:hover {{ 
                    background: {color};
                    color: #000;
                }}
                QPushButton:pressed {{
                    background: {color};
                }}
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, p=prompt: self._quick_llm_prompt(p))
            side_layout.addWidget(btn)
            self.llm_small_btns.append(btn)
        
        side_layout.addStretch()
        
        # "New" button to go back to prompt selection
        new_btn = QtWidgets.QPushButton("↺")
        new_btn.setMinimumSize(48, 48)
        new_btn.setMaximumSize(56, 56)
        new_btn.setToolTip("Start New")
        new_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 100, 100, 0.3);
                color: #888;
                border: 2px solid #555;
                border-radius: 12px;
                font-size: 20px;
            }
            QPushButton:hover { 
                background: #555;
                color: #fff;
                border-color: #888;
            }
            QPushButton:pressed {
                background: #444;
            }
        """)
        new_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        new_btn.clicked.connect(self._reset_llm_view)
        side_layout.addWidget(new_btn)
        
        response_layout.addWidget(self.llm_side_btns)
        
        self.llm_content_stack.addWidget(self.llm_response_page)
        
        # Start on prompt page
        self.llm_content_stack.setCurrentWidget(self.llm_prompt_page)
        
        layout.addWidget(self.llm_content_stack, stretch=1)
        
        # Input row at bottom (always visible)
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(6)
        
        self.llm_prompt = QtWidgets.QLineEdit()
        self.llm_prompt.setPlaceholderText("")
        self.llm_prompt.setMinimumHeight(44)
        self.llm_prompt.setStyleSheet("""
            QLineEdit {
                padding: 10px 14px;
                font-size: 14px;
                border-radius: 10px;
                border: 2px solid #333333;
                background: #1a1a1a;
                color: #f0f0f0;
            }
            QLineEdit:focus { border-color: #ff8c00; }
        """)
        self.llm_prompt.returnPressed.connect(self._send_llm_prompt)
        input_row.addWidget(self.llm_prompt, stretch=1)
        
        # Hidden mode selector (default to coach)
        self.llm_mode = QtWidgets.QComboBox()
        self.llm_mode.addItems(["coach", "encourage", "focus", "analysis"])
        self.llm_mode.hide()
        
        self.llm_send = QtWidgets.QPushButton("ENTER")
        self.llm_send.setMinimumSize(80, 44)
        self.llm_send.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 14px;
                font-weight: 800;
                border-radius: 10px;
                border: none;
                padding: 10px 16px;
            }
            QPushButton:hover { background: #ffa333; }
            QPushButton:pressed { background: #cc7000; }
        """)
        self.llm_send.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.llm_send.clicked.connect(self._send_llm_prompt)
        input_row.addWidget(self.llm_send)
        
        layout.addLayout(input_row)

        # Tunable LLM params (compact controls)
        params_row = QtWidgets.QHBoxLayout()
        params_row.setSpacing(8)
        params_row.setContentsMargins(0, 6, 0, 2)

        self.llm_temp_label = QtWidgets.QLabel("Temp: 0.70")
        self.llm_temp_label.setStyleSheet("font-size: 13px; color: #bbb;")
        params_row.addWidget(self.llm_temp_label)
        self.llm_temp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.llm_temp_slider.setRange(10, 120)
        self.llm_temp_slider.setValue(70)
        self.llm_temp_slider.setFixedWidth(110)
        self.llm_temp_slider.valueChanged.connect(self._on_llm_params_changed)
        params_row.addWidget(self.llm_temp_slider)

        self.llm_tokens_label = QtWidgets.QLabel("Max tokens: 32")
        self.llm_tokens_label.setStyleSheet("font-size: 13px; color: #bbb;")
        params_row.addWidget(self.llm_tokens_label)
        self.llm_tokens_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.llm_tokens_slider.setRange(8, 256)
        self.llm_tokens_slider.setValue(32)
        self.llm_tokens_slider.setFixedWidth(120)
        self.llm_tokens_slider.valueChanged.connect(self._on_llm_params_changed)
        params_row.addWidget(self.llm_tokens_slider)

        self.llm_ctx_label = QtWidgets.QLabel("Context: 512")
        self.llm_ctx_label.setStyleSheet("font-size: 13px; color: #bbb;")
        params_row.addWidget(self.llm_ctx_label)
        self.llm_ctx_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.llm_ctx_slider.setRange(128, 4096)
        self.llm_ctx_slider.setValue(512)
        self.llm_ctx_slider.setFixedWidth(130)
        self.llm_ctx_slider.valueChanged.connect(self._on_llm_params_changed)
        params_row.addWidget(self.llm_ctx_slider)

        self.llm_threads_label = QtWidgets.QLabel("Threads: 4")
        self.llm_threads_label.setStyleSheet("font-size: 13px; color: #bbb;")
        params_row.addWidget(self.llm_threads_label)
        self.llm_threads_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.llm_threads_slider.setRange(1, max(2, os.cpu_count() or 4))
        self.llm_threads_slider.setValue(min(4, max(2, os.cpu_count() or 4)))
        self.llm_threads_slider.setFixedWidth(90)
        self.llm_threads_slider.valueChanged.connect(self._on_llm_params_changed)
        params_row.addWidget(self.llm_threads_slider)

        self.llm_batch_label = QtWidgets.QLabel("Batch: 128")
        self.llm_batch_label.setStyleSheet("font-size: 13px; color: #bbb;")
        params_row.addWidget(self.llm_batch_label)
        self.llm_batch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.llm_batch_slider.setRange(16, 512)
        self.llm_batch_slider.setValue(128)
        self.llm_batch_slider.setFixedWidth(100)
        self.llm_batch_slider.valueChanged.connect(self._on_llm_params_changed)
        params_row.addWidget(self.llm_batch_slider)

        self.llm_apply_params_btn = QtWidgets.QPushButton("APPLY")
        self.llm_apply_params_btn.setMinimumSize(72, 34)
        self.llm_apply_params_btn.setStyleSheet("""
            QPushButton {
                background: #1f1f1f;
                color: #ff8c00;
                border: 1px solid #ff8c00;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                padding: 4px 10px;
            }
            QPushButton:hover { background: #2a2a2a; }
            QPushButton:pressed { background: #1a1a1a; }
        """)
        self.llm_apply_params_btn.clicked.connect(self._apply_llm_params)
        params_row.addWidget(self.llm_apply_params_btn)

        self.llm_defaults_btn = QtWidgets.QPushButton("DEFAULTS")
        self.llm_defaults_btn.setMinimumSize(84, 34)
        self.llm_defaults_btn.setStyleSheet("""
            QPushButton {
                background: #1f1f1f;
                color: #bbb;
                border: 1px solid #444;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                padding: 4px 10px;
            }
            QPushButton:hover { background: #2a2a2a; color: #fff; }
            QPushButton:pressed { background: #1a1a1a; }
        """)
        self.llm_defaults_btn.clicked.connect(self._reset_llm_params)
        params_row.addWidget(self.llm_defaults_btn)

        self.llm_params_status = QtWidgets.QLabel("")
        self.llm_params_status.setStyleSheet("font-size: 12px; color: #888;")
        params_row.addWidget(self.llm_params_status)
        params_row.addStretch()

        layout.addLayout(params_row)

        toggles_row = QtWidgets.QHBoxLayout()
        toggles_row.setSpacing(10)
        toggle_style = """
            QCheckBox {
                font-size: 13px; color: #f0f0f0; spacing: 8px; padding: 2px;
            }
            QCheckBox::indicator {
                width: 18px; height: 18px; border-radius: 4px; border: 2px solid #666; background: #1a1a1a;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #ff8c00;
                background: #ff8c00;
                image: url(data:image/x-xpm;base64,LyogWFBNICovCnN0YXRpYyBjaGFyICogY2hlY2tfeHBtW10gPSB7CiIxMiAxMiAyIDEiLAoiICBjIE5vbmUiLAoiLiBjICMwMDAwMDAiLAoiICAgICAgICAgICAgIiwKIiAgICAgICAgICAgICIsCiIgICAgICAgICAgICAiLAoiICAgICAgICAgICAuIiwKIiAgICAgICAgICAuICIsCiIgICAgICAgICAuICAiLAoiICAgICAgICAuICAgIiwKIiAuICAgICAuICAgICIsCiIgIC4gICAuICAgICAiLAoiICAgLiAuICAgICAgIiwKIiAgICAuICAgICAgICIsCiIgICAgICAgICAgICAifTsK);
            }
        """
        self.llm_use_llm_toggle = QtWidgets.QCheckBox("Use LLM")
        self.llm_use_llm_toggle.setChecked(True)
        self.llm_use_llm_toggle.setStyleSheet(toggle_style)
        self.llm_use_llm_toggle.toggled.connect(self._on_llm_params_changed)
        toggles_row.addWidget(self.llm_use_llm_toggle)

        self.llm_use_stats_toggle = QtWidgets.QCheckBox("Use Stats Context")
        self.llm_use_stats_toggle.setChecked(True)
        self.llm_use_stats_toggle.setStyleSheet(toggle_style)
        self.llm_use_stats_toggle.toggled.connect(self._on_llm_params_changed)
        toggles_row.addWidget(self.llm_use_stats_toggle)

        self.llm_singlish_toggle = QtWidgets.QCheckBox("Singlish")
        self.llm_singlish_toggle.setChecked(False)
        self.llm_singlish_toggle.setStyleSheet(toggle_style)
        self.llm_singlish_toggle.toggled.connect(self._on_llm_params_changed)
        toggles_row.addWidget(self.llm_singlish_toggle)

        self.llm_advice_toggle = QtWidgets.QCheckBox("Advice")
        self.llm_advice_toggle.setChecked(False)
        self.llm_advice_toggle.setStyleSheet(toggle_style)
        self.llm_advice_toggle.toggled.connect(self._on_llm_params_changed)
        toggles_row.addWidget(self.llm_advice_toggle)

        self.llm_memory_toggle = QtWidgets.QCheckBox("Remember")
        self.llm_memory_toggle.setChecked(False)
        self.llm_memory_toggle.setStyleSheet(toggle_style)
        self.llm_memory_toggle.toggled.connect(self._on_llm_params_changed)
        toggles_row.addWidget(self.llm_memory_toggle)

        self.llm_history_label = QtWidgets.QLabel("History: 4")
        self.llm_history_label.setStyleSheet("font-size: 13px; color: #f0f0f0;")
        toggles_row.addWidget(self.llm_history_label)
        self.llm_history_spin = QtWidgets.QSpinBox()
        self.llm_history_spin.setRange(0, 12)
        self.llm_history_spin.setValue(4)
        self.llm_history_spin.setStyleSheet("""
            QSpinBox {
                background: #1a1a1a;
                color: #f0f0f0;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 2px 6px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 14px;
                background: #222;
                border-left: 1px solid #333;
            }
            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 8px;
                height: 8px;
            }
            QSpinBox::up-arrow {
                image: url(data:image/x-xpm;base64,LyogWFBNICovCnN0YXRpYyBjaGFyICogdXBfeHBtW10gPSB7CiI4IDggMiAxIiwKIiAgYyBOb25lIiwKIi4gYyAjRkZGRkZGIiwKIiAgICAuICAgIiwKIiAgIC4uLiAgIiwKIiAgLi4uLi4gIiwKIiAuLi4uLi4uIiwKIiAgIC4uLiAgIiwKIiAgIC4uLiAgIiwKIiAgIC4uLiAgIiwKIiAgICAgICAgIn07Cg==);
            }
            QSpinBox::down-arrow {
                image: url(data:image/x-xpm;base64,LyogWFBNICovCnN0YXRpYyBjaGFyICogZG93bl94cG1bXSA9IHsKIjggOCAyIDEiLAoiICBjIE5vbmUiLAoiLiBjICNGRkZGRkYiLAoiICAgICAgICAiLAoiICAgLi4uICAiLAoiICAgLi4uICAiLAoiICAgLi4uICAiLAoiIC4uLi4uLi4iLAoiICAuLi4uLiAiLAoiICAgLi4uICAiLAoiICAgIC4gICAifTsK);
            }
        """)
        self.llm_history_spin.valueChanged.connect(self._on_llm_params_changed)
        toggles_row.addWidget(self.llm_history_spin)
        toggles_row.addStretch()

        layout.addLayout(toggles_row)
        self.llm_system_prompt = QtWidgets.QPlainTextEdit()
        self.llm_system_prompt.setPlaceholderText("System prompt for the model...")
        self.llm_system_prompt.setFixedHeight(90)
        self.llm_system_prompt.setStyleSheet("""
            QPlainTextEdit {
                background: #1a1a1a;
                color: #f0f0f0;
                border: 1px solid #2f2f2f;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.llm_system_prompt)
        self._on_llm_params_changed()
        
        self.llm_tab.setLayout(layout)
    
    def _reset_llm_view(self):
        """Reset LLM view back to prompt selection."""
        self.llm_content_stack.setCurrentWidget(self.llm_prompt_page)
        self.llm_prompt.clear()

    def _on_llm_params_changed(self):
        """Update LLM parameter labels as sliders move."""
        temp = self.llm_temp_slider.value() / 100.0
        tokens = self.llm_tokens_slider.value()
        ctx = self.llm_ctx_slider.value()
        threads = self.llm_threads_slider.value()
        batch = self.llm_batch_slider.value()
        self.llm_temp_label.setText(f"Temp: {temp:.2f}")
        self.llm_tokens_label.setText(f"Max tokens: {tokens}")
        self.llm_ctx_label.setText(f"Context: {ctx}")
        self.llm_threads_label.setText(f"Threads: {threads}")
        self.llm_batch_label.setText(f"Batch: {batch}")
        if hasattr(self, "llm_history_label"):
            self.llm_history_label.setText(f"History: {self.llm_history_spin.value()}")
        if hasattr(self, "llm_params_status"):
            self.llm_params_status.setText("")

    def _apply_llm_params(self) -> None:
        """Apply LLM params to the LLM node via ROS parameters."""
        if not hasattr(self.ros, "llm_param_client") or not self.ros.llm_param_client.service_is_ready():
            if hasattr(self, "llm_params_status"):
                self.llm_params_status.setText("LLM params not ready")
            return
        temp = float(self.llm_temp_slider.value()) / 100.0
        tokens = int(self.llm_tokens_slider.value())
        ctx = int(self.llm_ctx_slider.value())
        threads = int(self.llm_threads_slider.value())
        batch = int(self.llm_batch_slider.value())
        use_llm = bool(self.llm_use_llm_toggle.isChecked()) if hasattr(self, "llm_use_llm_toggle") else True
        use_stats = bool(self.llm_use_stats_toggle.isChecked()) if hasattr(self, "llm_use_stats_toggle") else True
        singlish = bool(self.llm_singlish_toggle.isChecked()) if hasattr(self, "llm_singlish_toggle") else False
        advice = bool(self.llm_advice_toggle.isChecked()) if hasattr(self, "llm_advice_toggle") else False
        memory = bool(self.llm_memory_toggle.isChecked()) if hasattr(self, "llm_memory_toggle") else False
        history_turns = int(self.llm_history_spin.value()) if hasattr(self, "llm_history_spin") else 4
        system_prompt = self.llm_system_prompt.toPlainText().strip() if hasattr(self, "llm_system_prompt") else ""
        ros_params = [
            Parameter("temperature", Parameter.Type.DOUBLE, temp),
            Parameter("max_tokens", Parameter.Type.INTEGER, tokens),
            Parameter("n_ctx", Parameter.Type.INTEGER, ctx),
            Parameter("n_threads", Parameter.Type.INTEGER, threads),
            Parameter("n_batch", Parameter.Type.INTEGER, batch),
            Parameter("use_llm_if_available", Parameter.Type.BOOL, use_llm),
            Parameter("use_stats_context", Parameter.Type.BOOL, use_stats),
            Parameter("singlish", Parameter.Type.BOOL, singlish),
            Parameter("advice", Parameter.Type.BOOL, advice),
            Parameter("memory", Parameter.Type.BOOL, memory),
            Parameter("history_turns", Parameter.Type.INTEGER, history_turns),
            Parameter("system_prompt", Parameter.Type.STRING, system_prompt),
        ]
        req = SetParameters.Request()
        req.parameters = [p.to_parameter_msg() for p in ros_params]
        self.ros.llm_param_client.call_async(req)
        if hasattr(self, "llm_params_status"):
            self.llm_params_status.setText("Applied")

    def _reset_llm_params(self) -> None:
        """Reset LLM params to defaults used by the system."""
        self.llm_temp_slider.setValue(70)
        self.llm_tokens_slider.setValue(32)
        self.llm_ctx_slider.setValue(512)
        self.llm_threads_slider.setValue(min(4, max(2, os.cpu_count() or 4)))
        self.llm_batch_slider.setValue(128)
        if hasattr(self, "llm_use_llm_toggle"):
            self.llm_use_llm_toggle.setChecked(True)
        if hasattr(self, "llm_use_stats_toggle"):
            self.llm_use_stats_toggle.setChecked(True)
        if hasattr(self, "llm_singlish_toggle"):
            self.llm_singlish_toggle.setChecked(False)
        if hasattr(self, "llm_advice_toggle"):
            self.llm_advice_toggle.setChecked(False)
        if hasattr(self, "llm_memory_toggle"):
            self.llm_memory_toggle.setChecked(False)
        if hasattr(self, "llm_history_spin"):
            self.llm_history_spin.setValue(4)
        if hasattr(self, "llm_system_prompt"):
            self.llm_system_prompt.setPlainText(
                "You are a helpful boxing coach. Give brief, actionable advice. One sentence only."
            )
        self._on_llm_params_changed()
        self._apply_llm_params()
    
    def _quick_llm_prompt(self, prompt_key: str):
        """Send a quick pre-defined prompt to LLM with random variation."""
        import random
        
        # If it's a key to our variations, pick a random prompt
        if hasattr(self, '_llm_prompt_variations') and prompt_key in self._llm_prompt_variations:
            prompt = random.choice(self._llm_prompt_variations[prompt_key])
        else:
            # Fallback - use the prompt directly
            prompt = prompt_key
        
        self.llm_prompt.setText(prompt)
        self._send_llm_prompt()
    
    def _quick_coach_action(self, mode: str):
        """Quick coach action from reaction drill page - shows result in coach bar."""
        import random
        
        # Simple prompts that DON'T reference user data
        prompts = {
            "tip": [
                "One boxing technique tip. Under 10 words.",
                "Quick defensive tip. Under 8 words.",
                "Footwork advice. Under 8 words.",
            ],
            "hype": [
                "Motivate me for training! One intense line.",
                "Fire me up! One powerful sentence.",
                "Champion energy! One line.",
            ], 
            "focus": [
                "Help me focus. One calming sentence.",
                "Mental reset cue. Under 10 words.",
                "Breathing tip for focus. Under 10 words.",
            ],
        }
        
        prompt_list = prompts.get(mode, prompts["tip"])
        prompt = random.choice(prompt_list)
        
        # Show loading state
        self.trash_label.setText("🤔 ...")
        
        # Send LLM request asynchronously
        def do_request():
            if not self.ros.llm_client.service_is_ready():
                return "Coach warming up..."
            req = GenerateLLM.Request()
            req.mode = "coach"
            req.prompt = prompt
            future = self.ros.llm_client.call_async(req)
            rclpy.spin_until_future_complete(self.ros, future, timeout_sec=8.0)
            if future.result() is not None:
                return future.result().response
            return "Try again!"
        
        # Run in thread to not block UI
        def run_and_update():
            response = do_request()
            # Update UI from main thread
            QtCore.QMetaObject.invokeMethod(
                self.trash_label, "setText",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, response)
            )
        
        threading.Thread(target=run_and_update, daemon=True).start()
    
    def _setup_shadow_tab(self) -> None:
        """Setup shadow sparring drill tab - with camera feed."""
        outer_layout = QtWidgets.QVBoxLayout(self.shadow_tab)
        outer_layout.setContentsMargins(8, 6, 8, 6)
        outer_layout.setSpacing(6)
        self._add_back_btn(outer_layout)
        
        # Content - horizontal layout
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(10)
        
        # === LEFT: Camera Feed ===
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(4)

        # Center camera vertically (match reaction layout)
        left_col.addStretch(1)
        
        video_frame = QtWidgets.QFrame()
        video_frame.setFixedSize(420, 340)
        video_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 2px solid #222;
                border-radius: 8px;
            }
        """)
        video_inner = QtWidgets.QVBoxLayout(video_frame)
        video_inner.setContentsMargins(4, 4, 4, 4)
        video_inner.setSpacing(4)
        
        video_header = QtWidgets.QHBoxLayout()
        self.shadow_video_status = QtWidgets.QLabel("📹 LIVE")
        self.shadow_video_status.setStyleSheet("font-size: 16px; font-weight: 700; color: #00cc00; padding: 4px;")
        video_header.addWidget(self.shadow_video_status)
        video_header.addStretch()
        video_inner.addLayout(video_header)
        
        self.shadow_preview = QtWidgets.QLabel()
        self.shadow_preview.setFixedSize(400, 300)
        self.shadow_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.shadow_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shadow_preview.setText("⏳ Connecting...")
        self.shadow_preview.setStyleSheet("""
            background: #000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #555;
            font-size: 13px;
        """)
        video_inner.addWidget(self.shadow_preview, stretch=1)
        
        left_col.addWidget(video_frame)
        left_col.addStretch(1)
        content.addLayout(left_col)
        
        # === RIGHT: Controls & Action Display ===
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(4)
        
        # Action prediction card - prominent display
        self.action_card = QtWidgets.QFrame()
        self.action_card.setMinimumHeight(76)
        self.action_card.setMaximumHeight(120)
        self.action_card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.action_card.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        ac_layout = QtWidgets.QVBoxLayout(self.action_card)
        ac_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ac_layout.setContentsMargins(8, 8, 8, 8)
        ac_layout.setSpacing(6)
        
        self.action_label = QtWidgets.QLabel("READY")
        self.action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_label.setMinimumHeight(55)
        self.action_label.setStyleSheet(
            "font-size: 34px; font-weight: 800; color: #ff8c00; "
            "background: transparent;"
        )
        ac_layout.addWidget(self.action_label)
        
        self.action_conf_label = QtWidgets.QLabel("Complete the selected combo")
        self.action_conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_conf_label.setStyleSheet("font-size: 13px; color: #ffa333; background: transparent;")
        ac_layout.addWidget(self.action_conf_label)
        
        right_col.addWidget(self.action_card)
        
        # Start/Stop row (aligned with Reaction drill layout)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.shadow_start_btn = QtWidgets.QPushButton("▶  START")
        self.shadow_start_btn.setMinimumHeight(48)
        self.shadow_start_btn.setMinimumWidth(120)
        self.shadow_start_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.shadow_start_btn.clicked.connect(self._start_shadow_drill)
        self.shadow_start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 18px;
                font-weight: 700;
                border-radius: 10px;
                padding: 14px 20px;
            }
            QPushButton:hover { background: #ffa333; }
            QPushButton:pressed { background: #cc7000; }
        """)
        self._shadow_start_style = self.shadow_start_btn.styleSheet()
        self.shadow_start_btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        
        self.shadow_stop_btn = QtWidgets.QPushButton("⬛  STOP")
        self.shadow_stop_btn.setMinimumHeight(48)
        self.shadow_stop_btn.setMinimumWidth(120)
        self.shadow_stop_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.shadow_stop_btn.clicked.connect(self._stop_shadow_drill)
        self.shadow_stop_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #888;
                font-size: 18px;
                font-weight: 700;
                border-radius: 10px;
                border: 1px solid #333;
                padding: 14px 20px;
            }
            QPushButton:hover { background: #333; color: #fff; }
            QPushButton:pressed { background: #222; }
        """)
        self.shadow_stop_btn.setEnabled(False)
        
        btn_row.addWidget(self.shadow_start_btn, stretch=1)
        btn_row.addWidget(self.shadow_stop_btn, stretch=1)
        right_col.addLayout(btn_row)
        
        # Combo selector (optional advanced)
        combo_frame = QtWidgets.QFrame()
        combo_frame.setStyleSheet("background: #1f1f1f; border-radius: 8px; border: 1px solid #2a2a2a;")
        combo_inner = QtWidgets.QHBoxLayout(combo_frame)
        combo_inner.setContentsMargins(12, 8, 12, 8)
        combo_inner.setSpacing(10)
        
        combo_label = QtWidgets.QLabel("Combo:")
        combo_label.setStyleSheet("font-weight: 700; font-size: 13px; color: #f5f5f5;")
        combo_inner.addWidget(combo_label)
        
        self.shadow_combo = QtWidgets.QComboBox()
        self.shadow_combo.setStyleSheet("""
            QComboBox {
                font-size: 13px;
                padding: 6px 10px;
                color: #f5f5f5;
                background: #1b1b1b;
                border: 1px solid #2f2f2f;
                border-radius: 6px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #1b1b1b;
                color: #f5f5f5;
                selection-background-color: #ff8c00;
                border: 1px solid #2f2f2f;
            }
        """)
        self._shadow_drill_defs = self._load_shadow_drill_definitions()
        self._shadow_drill_map = {drill["name"]: drill for drill in self._shadow_drill_defs}
        for drill in self._shadow_drill_defs:
            seq_label = " - ".join(step.replace("_", " ").upper() for step in drill["sequence"])
            label = f"{drill['name']} ({seq_label})" if seq_label else drill["name"]
            self.shadow_combo.addItem(label, drill["name"])
        if self.shadow_combo.count() == 0:
            self.shadow_combo.addItem("1-1-2 Combo (JAB - JAB - CROSS)", "1-1-2 Combo")
        self.shadow_combo.currentIndexChanged.connect(self._on_shadow_combo_changed)
        combo_inner.addWidget(self.shadow_combo, stretch=1)
        
        right_col.addWidget(combo_frame)
        
        # Progress stats grid (compact, no overlap)
        progress_frame = QtWidgets.QFrame()
        progress_frame.setMinimumHeight(130)
        progress_frame.setStyleSheet("background: #151515; border-radius: 8px; padding: 0px; border: none;")
        
        # Use VBox instead of Grid to prevent overlapping
        prog_layout = QtWidgets.QVBoxLayout(progress_frame)
        prog_layout.setContentsMargins(12, 12, 12, 12)
        prog_layout.setSpacing(8)
        
        # Row 1: Stats Row (Score counters + Time)
        stats_row = QtWidgets.QHBoxLayout()
        
        self.shadow_correct_label = QtWidgets.QLabel("Correct: 0")
        self.shadow_correct_label.setStyleSheet("font-size: 14px; color: #00ff00; font-weight: 600; background: transparent; border: none;")
        stats_row.addWidget(self.shadow_correct_label)
        
        stats_row.addStretch()
        
        self.shadow_progress_label = QtWidgets.QLabel("Step: 0/0")
        self.shadow_progress_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #ff8c00; background: transparent; border: none;")
        stats_row.addWidget(self.shadow_progress_label)
        
        stats_row.addStretch()
        
        self.shadow_wrong_label = QtWidgets.QLabel("Wrong: 0")
        self.shadow_wrong_label.setStyleSheet("font-size: 14px; color: #ff4757; font-weight: 600; background: transparent; border: none;")
        stats_row.addWidget(self.shadow_wrong_label)
        
        stats_row.addStretch()
        
        self.shadow_elapsed_label = QtWidgets.QLabel("Time: 0.0s")
        self.shadow_elapsed_label.setStyleSheet("font-size: 13px; color: #aaa; background: transparent; border: none;")
        stats_row.addWidget(self.shadow_elapsed_label)
        prog_layout.addLayout(stats_row)

        
        # Row 3: Expected Punch
        self.shadow_expected_label = QtWidgets.QLabel("Next: --")
        self.shadow_expected_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shadow_expected_label.setStyleSheet("font-size: 14px; color: #ccc; border-top: 1px solid #333; padding-top: 6px; margin-top: 4px; background: transparent;")
        prog_layout.addWidget(self.shadow_expected_label)
        
        right_col.addWidget(progress_frame)

        # Detected label in its own compact frame
        detected_frame = QtWidgets.QFrame()
        detected_frame.setStyleSheet("background: #151515; border-radius: 8px; border: none;")
        detected_layout = QtWidgets.QVBoxLayout(detected_frame)
        detected_layout.setContentsMargins(12, 8, 12, 8)
        
        self.shadow_detected_label = QtWidgets.QLabel("DETECTED: --")
        self.shadow_detected_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shadow_detected_label.setMinimumHeight(56)
        self.shadow_detected_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.shadow_detected_label.setStyleSheet("""
            QLabel {
                background: #222;
                color: #555;
                font-size: 24px;
                font-weight: 800;
                border-radius: 8px;
                padding: 6px 10px;
                border: none;
            }
        """)
        self._shadow_detected_style = self.shadow_detected_label.styleSheet()
        detected_layout.addWidget(self.shadow_detected_label)
        right_col.addWidget(detected_frame)
        
        # Punch progress and combo history in horizontal layout
        self.shadow_checkbox_container = QtWidgets.QFrame()
        self.shadow_checkbox_container.setStyleSheet("background: #151515; border-radius: 8px;")
        self.shadow_checkbox_container.setFixedHeight(100)
        
        # Main horizontal layout for punch boxes + history
        punch_row = QtWidgets.QHBoxLayout(self.shadow_checkbox_container)
        punch_row.setContentsMargins(16, 8, 16, 8)
        punch_row.setSpacing(16)
        
        # Left: Current combo punch boxes
        self.shadow_checkbox_progress = CheckboxProgressWidget(count=3, labels=["jab", "jab", "cross"])
        punch_row.addWidget(self.shadow_checkbox_progress, stretch=1)
        
        # Divider
        divider = QtWidgets.QFrame()
        divider.setFixedWidth(2)
        divider.setFixedHeight(60)
        divider.setStyleSheet("background: #333;")
        punch_row.addWidget(divider)
        
        # Right: Combo history (3 boxes)
        self.shadow_combo_history = ComboHistoryWidget(max_count=3)
        punch_row.addWidget(self.shadow_combo_history)
        
        right_col.addWidget(self.shadow_checkbox_container)
        
        # Combo result label (shows feedback after combo)
        self.shadow_combo_result = QtWidgets.QLabel("")
        self.shadow_combo_result.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.shadow_combo_result.setFixedHeight(24)
        self.shadow_combo_result.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #555;
                font-size: 12px;
                font-weight: 700;
            }
        """)
        right_col.addWidget(self.shadow_combo_result)
        
        content.addLayout(right_col, stretch=1)
        
        outer_layout.addLayout(content, stretch=1)
        
        # === BOTTOM: Coach Bar ===
        self.shadow_coach_bar = CoachBarWidget(self.ros)
        self.shadow_coach_bar.setMinimumHeight(70)
        self.shadow_coach_bar.setMaximumHeight(90)
        outer_layout.addWidget(self.shadow_coach_bar)

        # Initialize combo-dependent UI once labels exist
        self._on_shadow_combo_changed()

    def _load_shadow_drill_definitions(self) -> List[dict]:
        """Load shadow sparring drill definitions for combo selection."""
        try:
            from ament_index_python.packages import get_package_share_directory
            import yaml
            
            config_path = os.path.join(
                get_package_share_directory("boxbunny_drills"),
                "config",
                "drill_definitions.yaml",
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            drills = []
            for drill in config.get("shadow_sparring_drills", []):
                name = drill.get("name")
                sequence = drill.get("sequence") or []
                if name and isinstance(sequence, list):
                    drills.append({"name": name, "sequence": sequence})
            return drills
        except Exception:
            return [{"name": "1-1-2 Combo", "sequence": ["jab", "jab", "cross"]}]

    def _on_shadow_combo_changed(self) -> None:
        """Update labels when the combo selection changes."""
        drill_name = self.shadow_combo.currentData()
        drill = self._shadow_drill_map.get(drill_name)
        if not drill:
            return
        sequence = drill["sequence"]
        self._set_shadow_checkbox_count(max(1, len(sequence)), sequence)
        if sequence:
            self.shadow_expected_label.setText(f"Next: {sequence[0].replace('_', ' ').upper()}")
            self.action_conf_label.setText(f"Step 1/{len(sequence)}")
        else:
            self.shadow_expected_label.setText("Next: --")
            self.action_conf_label.setText("Step --")

    def _set_shadow_checkbox_count(self, count: int, labels: list = None) -> None:
        """Rebuild checkbox progress widget for the current combo length."""
        if not hasattr(self, "shadow_checkbox_progress"):
            return
        if self.shadow_checkbox_progress.count == count and labels is None:
            return
        
        # Get the container's layout (QHBoxLayout)
        layout = self.shadow_checkbox_container.layout()
        if layout is None:
            return
        
        # Find and remove old checkbox progress widget
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() == self.shadow_checkbox_progress:
                layout.takeAt(i)
                self.shadow_checkbox_progress.setParent(None)
                break
        
        # Create new checkbox progress with updated count/labels
        self.shadow_checkbox_progress = CheckboxProgressWidget(count=count, labels=labels or [])
        
        # Insert at position 0 (before divider and history)
        layout.insertWidget(0, self.shadow_checkbox_progress, stretch=1)

    
    def _setup_defence_tab(self) -> None:
        """Setup defence drill tab - with camera feed."""
        outer_layout = QtWidgets.QVBoxLayout(self.defence_tab)
        outer_layout.setContentsMargins(10, 6, 10, 6)
        outer_layout.setSpacing(6)
        self._add_back_btn(outer_layout)
        
        # Content - horizontal layout
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(10)
        
        # === LEFT: Camera Feed ===
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(4)
        
        # Add stretch at top to center camera
        left_col.addStretch(1)
        
        video_frame = QtWidgets.QFrame()
        video_frame.setFixedSize(420, 340)
        video_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        video_frame.setStyleSheet("""
            QFrame {
                background: #0a0a0a;
                border: 2px solid #222;
                border-radius: 8px;
            }
        """)
        video_inner = QtWidgets.QVBoxLayout(video_frame)
        video_inner.setContentsMargins(4, 4, 4, 4)
        video_inner.setSpacing(4)
        
        video_header = QtWidgets.QHBoxLayout()
        self.defence_video_status = QtWidgets.QLabel("📹 LIVE")
        self.defence_video_status.setStyleSheet("font-size: 16px; font-weight: 700; color: #00cc00; padding: 4px;")
        video_header.addWidget(self.defence_video_status)
        video_header.addStretch()
        video_inner.addLayout(video_header)
        
        self.defence_preview = QtWidgets.QLabel()
        self.defence_preview.setFixedSize(400, 300)
        self.defence_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.defence_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_preview.setText("⏳ Connecting...")
        self.defence_preview.setStyleSheet("""
            background: #000;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            color: #555;
            font-size: 13px;
        """)
        video_inner.addWidget(self.defence_preview, stretch=1)
        
        left_col.addWidget(video_frame)
        left_col.addStretch(1)
        content.addLayout(left_col)
        
        # === RIGHT: Controls & Block Indicator ===
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(8)
        
        # Add stretch at top to center content
        right_col.addStretch(1)
        
        # Block indicator - prominent display
        self.block_indicator = QtWidgets.QFrame()
        self.block_indicator.setMinimumHeight(96)
        self.block_indicator.setMaximumHeight(120)
        self.block_indicator.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.block_indicator.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        bi_layout = QtWidgets.QVBoxLayout(self.block_indicator)
        bi_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        bi_layout.setContentsMargins(8, 16, 8, 12)
        bi_layout.setSpacing(2)
        
        self.defence_action_label = QtWidgets.QLabel("READY")
        self.defence_action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_action_label.setStyleSheet("font-size: 34px; font-weight: 800; color: #ff8c00; background: transparent;")
        bi_layout.addWidget(self.defence_action_label)
        
        self.defence_sub_label = QtWidgets.QLabel("Block all incoming attacks")
        self.defence_sub_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.defence_sub_label.setStyleSheet("font-size: 13px; color: #ffa333; background: transparent;")
        bi_layout.addWidget(self.defence_sub_label)
        
        right_col.addWidget(self.block_indicator)
        
        # Big START button at top (Standardized)
        self.defence_start_btn = QtWidgets.QPushButton("▶  START DRILL")
        self.defence_start_btn.setMinimumHeight(54)
        self.defence_start_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.defence_start_btn.clicked.connect(self._start_defence_drill)
        self.defence_start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 18px;
                font-weight: 700;
                border-radius: 10px;
                padding: 14px 24px;
            }
            QPushButton:hover { background: #ffa333; }
            QPushButton:pressed { background: #cc7000; }
        """)
        self.defence_start_btn.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        right_col.addWidget(self.defence_start_btn)

        # Defence combo selector (motor pattern)
        combo_frame = QtWidgets.QFrame()
        combo_frame.setStyleSheet("background: #1f1f1f; border-radius: 8px; border: 1px solid #2a2a2a;")
        combo_inner = QtWidgets.QHBoxLayout(combo_frame)
        combo_inner.setContentsMargins(12, 8, 12, 8)
        combo_inner.setSpacing(10)

        combo_label = QtWidgets.QLabel("Combo:")
        combo_label.setStyleSheet("font-weight: 700; font-size: 13px; color: #f5f5f5;")
        combo_inner.addWidget(combo_label)

        self.defence_combo = QtWidgets.QComboBox()
        self.defence_combo.setStyleSheet("""
            QComboBox {
                font-size: 13px;
                padding: 6px 10px;
                color: #f5f5f5;
                background: #1b1b1b;
                border: 1px solid #2f2f2f;
                border-radius: 6px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #1b1b1b;
                color: #f5f5f5;
                selection-background-color: #ff8c00;
                border: 1px solid #2f2f2f;
            }
        """)
        self._defence_drill_defs = self._load_defence_drill_definitions()
        default_defence = {"name": "JAB-JAB-CROSS", "sequence": ["1", "1", "2"], "interval_s": 2.5}
        if not any(d.get("name") == default_defence["name"] for d in self._defence_drill_defs):
            self._defence_drill_defs.insert(0, default_defence)
        self._defence_drill_map = {drill["name"]: drill for drill in self._defence_drill_defs}
        for drill in self._defence_drill_defs:
            seq = drill["sequence"]
            seq_label = " - ".join(str(step) for step in seq)
            label = f"{drill['name']} ({seq_label})" if seq_label else drill["name"]
            self.defence_combo.addItem(label, drill["name"])
        if self.defence_combo.count() == 0:
            self.defence_combo.addItem("JAB-JAB-CROSS (1 - 1 - 2)", "JAB-JAB-CROSS")
            self._defence_drill_map["JAB-JAB-CROSS"] = default_defence
        self.defence_combo.currentIndexChanged.connect(self._on_defence_combo_changed)
        combo_inner.addWidget(self.defence_combo, stretch=1)

        right_col.addWidget(combo_frame)
        
        # Progress info
        progress_frame = QtWidgets.QFrame()
        progress_frame.setStyleSheet("background: #151515; border-radius: 8px; border: 1px solid #282828;")
        prog_layout = QtWidgets.QGridLayout(progress_frame)
        prog_layout.setSpacing(6)
        prog_layout.setContentsMargins(12, 10, 12, 10)
        
        self.defence_progress_label = QtWidgets.QLabel("Blocks: 0/3")
        self.defence_progress_label.setStyleSheet("font-size: 17px; font-weight: 700; color: #ff8c00;")
        self.defence_elapsed_label = QtWidgets.QLabel("Time: 0.0s")
        self.defence_elapsed_label.setStyleSheet("font-size: 16px; color: #888;")
        self.defence_status_label = QtWidgets.QLabel("Status: idle")
        self.defence_status_label.setStyleSheet("font-size: 16px; color: #666;")
        
        prog_layout.addWidget(self.defence_progress_label, 0, 0)
        prog_layout.addWidget(self.defence_elapsed_label, 0, 1)
        prog_layout.addWidget(self.defence_status_label, 1, 0, 1, 2)
        
        right_col.addWidget(progress_frame)
        
        # Checkbox progress indicator (3 blocks)
        self.defence_checkbox_container = QtWidgets.QFrame()
        self.defence_checkbox_container.setStyleSheet("background: transparent;")
        self.defence_checkbox_layout = QtWidgets.QVBoxLayout(self.defence_checkbox_container)
        self.defence_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.defence_checkbox_layout.setSpacing(0)
        self.defence_checkbox_progress = CheckboxProgressWidget(count=3)
        self.defence_checkbox_layout.addWidget(self.defence_checkbox_progress)
        right_col.addWidget(self.defence_checkbox_container)
        
        right_col.addStretch(1)
        content.addLayout(right_col, stretch=1)
        
        outer_layout.addLayout(content, stretch=1)
        
        # === BOTTOM: Coach Bar ===
        self.defence_coach_bar = CoachBarWidget(self.ros)
        outer_layout.addWidget(self.defence_coach_bar)
        
        # Initialize defence drill state
        self._defence_block_count = 0
        self._defence_total_blocks = 3
        self._defence_running = False
        self._defence_attack_interval_ms = 2500
        self._defence_waiting_for_robot = False
        self._defence_status_timer = None
        self._defence_last_status_stamp = None
        self._on_defence_combo_changed()

    def _load_defence_drill_definitions(self) -> List[dict]:
        """Load defence drill definitions for motor combo selection."""
        try:
            from ament_index_python.packages import get_package_share_directory
            import yaml

            config_path = os.path.join(
                get_package_share_directory("boxbunny_drills"),
                "config",
                "drill_definitions.yaml",
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            drills = []
            for drill in config.get("defence_drills", []):
                name = drill.get("name")
                positions = drill.get("positions") or []
                if name and isinstance(positions, list) and positions:
                    # For GUI combo selection, use the position list as the sequence.
                    drills.append({
                        "name": name,
                        "sequence": positions,
                        "interval_s": float(drill.get("attack_interval_s", 2.5)),
                    })
            return drills
        except Exception:
            return []

    def _set_defence_checkbox_count(self, count: int) -> None:
        """Rebuild checkbox progress widget for defence drill length."""
        if not hasattr(self, "defence_checkbox_progress"):
            return
        if self.defence_checkbox_progress.count == count:
            return
        if not hasattr(self, "defence_checkbox_layout"):
            return
        # Remove old widget
        while self.defence_checkbox_layout.count():
            item = self.defence_checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.defence_checkbox_progress = CheckboxProgressWidget(count=count)
        self.defence_checkbox_layout.addWidget(self.defence_checkbox_progress)

    def _on_defence_combo_changed(self) -> None:
        """Update defence sequence when combo selection changes."""
        if not hasattr(self, "defence_combo"):
            return
        drill_name = self.defence_combo.currentData()
        drill = self._defence_drill_map.get(drill_name)
        if not drill:
            return
        sequence = drill.get("sequence") or []
        self._defence_total_blocks = max(1, len(sequence))
        self._defence_attack_interval_ms = int(1000 * max(0.5, float(drill.get("interval_s", 2.5))))
        self._set_defence_checkbox_count(self._defence_total_blocks)
        self.defence_sub_label.setText(f"Selected: {drill_name}")

    def _create_hsv_group(self, title: str):
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                background: #141414;
                border: 1px solid #2a2a2a;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 6px 10px;
                color: #ff8c00;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel {
                color: #f0f0f0;
                font-size: 13px;
                font-weight: 600;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a2a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ff8c00;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #ffa333;
            }
        """)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
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
        req.num_trials = 3  # 3 attempts as requested
        self.ros.start_stop_client.call_async(req)

    def _stop_drill(self) -> None:
        if not self.ros.start_stop_client.service_is_ready():
            return
        req = StartStopDrill.Request()
        req.start = False
        req.num_trials = 0
        self.ros.start_stop_client.call_async(req)

    def _send_llm_prompt(self) -> None:
        if self._llm_request_inflight:
            return
        if not self.ros.llm_client.service_is_ready():
            self.llm_response.setPlainText("⚠️ Coach is not available right now. Please try again later.")
            self.llm_content_stack.setCurrentWidget(self.llm_response_page)
            return
        prompt = self.llm_prompt.text().strip()
        if not prompt:
            return
        self.llm_prompt.clear()
        self._set_llm_request_state(True)
        self._llm_prefix_text = f"You: {prompt}\n\nCoach: "
        self.llm_response.setPlainText(self._llm_prefix_text)
        self.llm_content_stack.setCurrentWidget(self.llm_response_page)
        
        req = GenerateLLM.Request()
        req.prompt = prompt
        req.mode = self.llm_mode.currentText()
        req.context = "gui"
        future = self.ros.llm_client.call_async(req)
        future.add_done_callback(self._on_llm_response)

    def _on_llm_response(self, future) -> None:
        try:
            response = future.result()
            text = response.response if response.response else "🤔 Coach didn't respond. Try again!"
        except Exception as exc:
            text = f"⚠️ Error: {exc}"
        # Start streaming animation on the UI thread
        QtCore.QMetaObject.invokeMethod(
            self, "_start_text_stream",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text)
        )
    
    @QtCore.Slot(str)
    def _start_text_stream(self, full_text: str):
        """Stream text word by word for a natural typing effect."""
        self._llm_stream_words = full_text.split()
        self._llm_stream_word_idx = 0
        self._llm_current_text = ""
        
        # Switch to response page immediately
        self.llm_content_stack.setCurrentWidget(self.llm_response_page)
        prefix = getattr(self, "_llm_prefix_text", "")
        self.llm_response.setPlainText(prefix)
        
        # Create timer if needed
        if not hasattr(self, '_word_stream_timer'):
            self._word_stream_timer = QtCore.QTimer(self)
            self._word_stream_timer.timeout.connect(self._stream_next_word)
        
        # Start timer - 60ms per word for readable speed
        self._word_stream_timer.start(60)
    
    def _stream_next_word(self):
        """Add next word to LLM response for typing effect."""
        if self._llm_stream_word_idx >= len(self._llm_stream_words):
            self._word_stream_timer.stop()
            self._set_llm_request_state(False)
            return
        
        # Add next word with space
        word = self._llm_stream_words[self._llm_stream_word_idx]
        if self._llm_current_text:
            self._llm_current_text += " " + word
        else:
            self._llm_current_text = word
        
        self._llm_stream_word_idx += 1
        prefix = getattr(self, "_llm_prefix_text", "")
        self.llm_response.setPlainText(prefix + self._llm_current_text)

    def _set_llm_request_state(self, inflight: bool) -> None:
        """Toggle input UI while a request is running."""
        self._llm_request_inflight = inflight
        if hasattr(self, "llm_send"):
            self.llm_send.setDisabled(inflight)
            self.llm_send.setText("SENDING..." if inflight else "SEND")
        if hasattr(self, "llm_prompt"):
            self.llm_prompt.setDisabled(inflight)

    def _update_ui(self) -> None:
        with self.ros.lock:
            state = self.ros.drill_state
            summary = self.ros.drill_summary
            trash = self.ros.trash_talk
            imu = self.ros.last_imu
            punch = self.ros.last_punch
            img = self.ros.last_image
            color_img = self.ros.last_color_image
            color_stamp = self.ros.last_color_image_stamp
            color_fallback_img = self.ros.last_color_image_fallback
            color_fallback_stamp = self.ros.last_color_image_fallback_stamp
            if color_fallback_img is not None:
                if color_stamp is None or (
                    color_fallback_stamp is not None and color_fallback_stamp > color_stamp
                ):
                    color_img = color_fallback_img
            pose_img = self.ros.last_pose_image  # Pose skeleton image
            debug_img = img
            # Prefer debug overlays for color tracking views
            shadow_display_img = debug_img if debug_img is not None else color_img
            countdown = self.ros.drill_countdown
            punch_counter = self.ros.punch_counter
            
        # Determine which image to show for reaction preview (pose skeleton preferred)
        reaction_display_img = (
            pose_img if pose_img is not None else (debug_img if debug_img is not None else color_img)
        )

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
        elif state == "early_penalty":
            # User punched too early!
            self.cue_panel.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff3333, stop:1 #cc0000);
                border-radius: 10px;
                border: 3px solid #ff6666;
            """)
            self.state_label.setText("⚠️ EARLY!")
            self.state_label.setStyleSheet("font-size: 36px; font-weight: 900; color: #ffffff; border: none; background: transparent;")
            self.countdown_label.setText("Wait for the cue!")
            self.countdown_label.setStyleSheet("font-size: 12px; color: #ffcccc; border: none; background: transparent;")
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
        elif state == "cue":
            self.countdown_label.setText("GO GO GO!")
        elif state == "early_penalty":
            self.countdown_label.setText("Wait for the cue!")
        elif state == "waiting":
            self.countdown_label.setText("Focus...")
        elif state == "idle":
            self.countdown_label.setText("Press START to begin")
        else:
            self.countdown_label.setText("")

        # Update attempt tracking
        last_rt = summary.get("last_reaction_time_s") if isinstance(summary, dict) else None
        mean_rt = summary.get("avg_time") if isinstance(summary, dict) else None
        best_rt = summary.get("best_time") if isinstance(summary, dict) else None
        reaction_times = summary.get("reaction_times", []) if isinstance(summary, dict) else []
        
        # Update individual attempt labels (3 attempts)
        if hasattr(self, 'attempt_labels'):
            for i, lbl in enumerate(self.attempt_labels):
                if i < len(reaction_times):
                    rt = reaction_times[i]
                    is_best = (best_rt is not None and abs(rt - best_rt) < 0.001)
                    if is_best:
                        lbl.setText(f"{rt:.3f}s")
                        lbl.setStyleSheet("font-size: 16px; color: #ff8c00; font-weight: 700;")
                    else:
                        lbl.setText(f"{rt:.3f}s")
                        lbl.setStyleSheet("font-size: 16px; color: #f0f0f0; font-weight: 700;")
                else:
                    lbl.setText("--")
                    lbl.setStyleSheet("font-size: 16px; color: #555; font-weight: 700;")
        
        if hasattr(self, 'best_attempt_label'):
            if best_rt is not None:
                self.best_attempt_label.setText(f"{best_rt:.3f}s")
            else:
                self.best_attempt_label.setText("--")
        
        self.last_reaction_label.setText(f"{last_rt:.3f}s" if last_rt is not None else "--")
        self.summary_label.setText(f"{mean_rt:.3f}s" if mean_rt is not None else "--")
        
        # Update session stats panel
        if hasattr(self, 'total_attempts_label'):
            self.total_attempts_label.setText(f"Attempts: {len(reaction_times)}")
        if hasattr(self, 'avg_reaction_label'):
            self.avg_reaction_label.setText(f"Avg: {mean_rt:.3f}s" if mean_rt is not None else "Avg: --")
        if hasattr(self, 'session_best_label'):
            self.session_best_label.setText(f"Best: {best_rt:.3f}s" if best_rt is not None else "Best: --")
        
        # Update trash talk (only if not controlled by local coach bar)
        if trash and not (hasattr(self, 'reaction_coach_bar')):
            self.trash_label.setText(trash)

        # IMU display
        if imu and self.ros.imu_input_enabled:
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

        # Update reaction preview with POSE image (skeleton), others with color tracking
        if reaction_display_img is not None and not self._replay_active:
            # First frame received - update status
            if not self._camera_received:
                self._camera_received = True
                self.video_status_label.setText("● LIVE")
                self.video_status_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #00ff00;")
            
            now = time.time()
            if now - self._last_reaction_frame_ts >= 1.0 / 15.0:
                self._last_reaction_frame_ts = now
                self._frame_buffer.append((now, reaction_display_img))
                qimg_pose = self._to_qimage(reaction_display_img)
                pix_pose = QtGui.QPixmap.fromImage(qimg_pose)
                self.reaction_preview.setPixmap(
                    pix_pose.scaled(
                        self.reaction_preview.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.FastTransformation,
                    )
                )
        
        # Shadow and defence use color tracking debug image when available
        if shadow_display_img is not None:
            qimg2 = self._to_qimage(shadow_display_img)
            pix2 = QtGui.QPixmap.fromImage(qimg2)
            
            # Update shadow and defence previews with color tracking
            if hasattr(self, 'shadow_preview'):
                self.shadow_preview.setPixmap(
                    pix2.scaled(
                        self.shadow_preview.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.FastTransformation,
                    )
                )
            if hasattr(self, 'defence_preview'):
                self.defence_preview.setPixmap(
                    pix2.scaled(
                        self.defence_preview.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.FastTransformation,
                    )
                )
            if hasattr(self, 'shadow_video_status'):
                self.shadow_video_status.setText("📹 LIVE")
                self.shadow_video_status.setStyleSheet("font-size: 16px; font-weight: 700; color: #00cc00; padding: 4px;")
            if hasattr(self, 'defence_video_status'):
                self.defence_video_status.setText("📹 LIVE")
                self.defence_video_status.setStyleSheet("font-size: 16px; font-weight: 700; color: #00cc00; padding: 4px;")
        
        # Check for connecting status only when neither image is available
        if reaction_display_img is None and shadow_display_img is None:
            if not self._camera_received:
                self.video_status_label.setText("● CONNECTING...")
                self.video_status_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #ffaa00;")

        if punch_counter != self._last_punch_counter:
            self._last_punch_counter = punch_counter
            self._capture_replay_clip()
        
        # Update new drill tabs
        self._update_shadow_ui()
        self._update_defence_ui()
        self._update_shadow_service_status()

    def _update_reaction_stats(self) -> None:
        with self.ros.lock:
            summary = self.ros.drill_summary
        if not summary or summary.get("drill_name") != "reaction_drill":
            return
        
        times = summary.get("reaction_times", [])
        summary_key = (
            summary.get("trial_index"),
            summary.get("last_reaction_time_s"),
            summary.get("best_time"),
            summary.get("avg_time"),
            len(times),
        )
        if summary_key == self._last_reaction_summary_key:
            return
        self._last_reaction_summary_key = summary_key

        last_rt = summary.get("last_reaction_time_s")
        mean_rt = summary.get("avg_time")
        best_rt = summary.get("best_time")

        # Update attempt labels
        if hasattr(self, 'attempt_labels'):
            for i, lbl in enumerate(self.attempt_labels):
                if i < len(times):
                    rt = times[i]
                    is_best = (best_rt is not None and abs(rt - best_rt) < 0.001)
                    if is_best:
                        lbl.setText(f"{rt:.3f}s")
                        lbl.setStyleSheet("font-size: 16px; color: #ff8c00; font-weight: 700;")
                    else:
                        lbl.setText(f"{rt:.3f}s")
                        lbl.setStyleSheet("font-size: 16px; color: #f0f0f0; font-weight: 700;")
                else:
                    lbl.setText("--")
                    lbl.setStyleSheet("font-size: 16px; color: #555; font-weight: 700;")

        if hasattr(self, 'best_attempt_label'):
            self.best_attempt_label.setText(f"{best_rt:.3f}s" if best_rt is not None else "--")
        self.last_reaction_label.setText(f"{last_rt:.3f}s" if last_rt is not None else "--")
        self.summary_label.setText(f"{mean_rt:.3f}s" if mean_rt is not None else "--")
        
        # Update session stats panel
        if hasattr(self, 'total_attempts_label'):
            self.total_attempts_label.setText(f"Attempts: {len(times)}")
        if hasattr(self, 'avg_reaction_label'):
            self.avg_reaction_label.setText(f"Avg: {mean_rt:.3f}s" if mean_rt is not None else "Avg: --")
        if hasattr(self, 'session_best_label'):
            self.session_best_label.setText(f"Best: {best_rt:.3f}s" if best_rt is not None else "Best: --")

        # Bind last clip to last reaction time
        if last_rt is not None and self._pending_replay_clip is not None:
            self._commit_reaction_clip(float(last_rt))
        if hasattr(self, "replay_btn"):
            self.replay_btn.setEnabled(self._best_reaction_clip is not None)

    def _capture_replay_clip(self) -> None:
        if not self._frame_buffer:
            return
        now = time.time()
        clip = [frame for ts, frame in self._frame_buffer if now - ts <= 1.2]
        self._replay_frames = clip
        self._replay_index = 0
        self._pending_replay_clip = clip

    def _commit_reaction_clip(self, reaction_time: float) -> None:
        if not self._pending_replay_clip:
            return
        clip = self._pending_replay_clip
        self._pending_replay_clip = None
        # Keep only the last 0.8s of the clip
        tail_len = max(1, int(len(clip) * 0.66))
        clip = clip[-tail_len:]
        self._reaction_clips.append({"time": reaction_time, "frames": clip})
        if self._best_reaction_clip is None or reaction_time <= self._best_reaction_clip["time"]:
            self._best_reaction_clip = {"time": reaction_time, "frames": clip}

    def _start_replay(self) -> None:
        if self._best_reaction_clip and self._best_reaction_clip["frames"]:
            self._replay_frames = self._best_reaction_clip["frames"]
            self._replay_index = 0
        if not self._replay_frames:
            return
        self._replay_active = True
        if hasattr(self, "replay_btn"):
            self.replay_btn.setEnabled(False)
        fps = 8
        interval_ms = int(1000 / fps)
        self.replay_timer.start(interval_ms)

    def _play_replay(self) -> None:
        if self._replay_index >= len(self._replay_frames):
            self.replay_timer.stop()
            self._replay_active = False
            if hasattr(self, "replay_btn"):
                self.replay_btn.setEnabled(True)
            return
        frame = self._replay_frames[self._replay_index]
        self._replay_index += 1
        qimg = self._to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.reaction_preview.setPixmap(
            pix.scaled(self.reaction_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        )

    def _to_qimage(self, img):
        # Resize to reduce GUI lag (max width 480)
        target_w = 480
        h, w, _ = img.shape
        if w > target_w:
            scale = target_w / w
            new_h = int(h * scale)
            img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w, _ = img.shape
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Keep reference to data to prevent garbage collection
        # But QImage from data copies it by default unless specified otherwise? 
        # Actually copying is safer here.
        qimg = QtGui.QImage(rgb.data, w, h, QtGui.QImage.Format.Format_RGB888)
        return qimg.copy() 
    
    def _start_shadow_drill(self) -> None:
        """Start shadow sparring drill with a short countdown."""
        if not self.ros.shadow_drill_client.service_is_ready():
            self.shadow_coach_bar.set_message("Shadow drill service not ready.")
            return
        self.shadow_countdown.set_status("Get ready…")
        self.shadow_countdown.start(3)
        self.stack.setCurrentWidget(self.shadow_countdown)

    def _begin_shadow_drill_service(self) -> None:
        """Call shadow sparring drill service after countdown."""
        if not self.ros.shadow_drill_client.service_is_ready():
            self.stack.setCurrentWidget(self.shadow_tab)
            self.shadow_coach_bar.set_message("Shadow drill service not ready.")
            return

        drill_name = self.shadow_combo.currentData() or self.shadow_combo.currentText()
        req = StartDrill.Request()
        req.drill_name = drill_name
        self.ros.shadow_drill_client.call_async(req)

        self._reset_shadow_ui()

        drill = self._shadow_drill_map.get(drill_name)
        if drill and drill["sequence"]:
            sequence = drill["sequence"]
            first_display = sequence[0].replace("_", " ").upper()
            self.shadow_expected_label.setText(f"Next: {first_display}")
            self.action_label.setText(first_display)
            self.action_conf_label.setText(f"Step 1/{len(sequence)}")

    def _reset_shadow_ui(self) -> None:
        """Reset shadow sparring UI to a clean state."""
        self._last_shadow_step = 0
        self._last_failures = 0
        self._last_iterations = 0
        self._shadow_tracking_step = 0
        self.shadow_checkbox_progress.reset()
        self.shadow_detected_label.setText("DETECTED: IDLE")
        if hasattr(self, "_shadow_detected_style"):
            self.shadow_detected_label.setStyleSheet(self._shadow_detected_style)
        self.shadow_progress_label.setText("Step: 0/0")
        self.shadow_elapsed_label.setText("Time: 0.0s")
        self.shadow_correct_label.setText("Correct: 0")
        self.shadow_wrong_label.setText("Wrong: 0")
        self.shadow_expected_label.setText("Next: --")
        self.action_label.setText("READY")
        self.action_label.setStyleSheet(
            "font-size: 34px; font-weight: 800; color: #ff8c00; "
            "background: transparent; padding: 4px 0px;"
        )
        self.action_conf_label.setText("Select combo • Press START")
        # Reset combo result and history
        self.shadow_combo_result.setText("")
        self.shadow_combo_result.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #555;
                font-size: 16px;
                font-weight: 700;
                border-radius: 8px;
                padding: 4px 10px;
            }
        """)
        if hasattr(self, 'shadow_combo_history'):
            self.shadow_combo_history.reset()

    def _stop_shadow_drill(self) -> None:
        """Stop shadow sparring drill via ROS service."""
        if not self.ros.shadow_stop_client.wait_for_service(timeout_sec=2.0):
            self.shadow_coach_bar.set_message("Stop service not ready.")
            return
        self.ros.shadow_stop_client.call_async(Trigger.Request())
        self.shadow_start_btn.setText("▶  START")
        if hasattr(self, "_shadow_start_style"):
            self.shadow_start_btn.setStyleSheet(self._shadow_start_style)
    
    def _start_defence_drill(self) -> None:
        """Start defence drill - block 3 incoming attacks."""
        # Initialize defence drill state
        self._defence_running = True
        self._defence_block_count = 0
        self._defence_start_time = time.time()
        self._defence_attack_index = 0
        self._defence_waiting_for_robot = False
        self._defence_last_status_stamp = None
        # Use selected combo sequence
        drill_name = None
        if hasattr(self, "defence_combo"):
            drill_name = self.defence_combo.currentData()
        drill = self._defence_drill_map.get(drill_name) if hasattr(self, "_defence_drill_map") else None
        sequence = (drill.get("sequence") if drill else None) or ["JAB", "JAB", "CROSS"]
        self._defence_attacks = list(sequence)
        self._defence_total_blocks = max(1, len(self._defence_attacks))
        self._defence_attack_interval_ms = int(1000 * max(0.5, float((drill or {}).get("interval_s", 2.5))))
        self._set_defence_checkbox_count(self._defence_total_blocks)
        
        # Reset UI
        self.defence_checkbox_progress.reset()
        self._show_defence_attack()
        self.defence_progress_label.setText(f"Blocks: 0/{self._defence_total_blocks}")
        self.defence_status_label.setText("Status: IN PROGRESS")
        self.defence_coach_bar.set_message("Here it comes! Keep your guard up! 🛡️")
        
        # Update button to show stop
        self.defence_start_btn.setText("⬛  STOP")
        self.defence_start_btn.setStyleSheet("""
            QPushButton {
                background: #333;
                color: #fff;
                font-size: 14px;
                font-weight: 700;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover { background: #444; }
        """)
        self.defence_start_btn.clicked.disconnect()
        self.defence_start_btn.clicked.connect(self._stop_defence_drill)

        # Start polling robot action status
        self._start_defence_status_timer()
    
    def _stop_defence_drill(self) -> None:
        """Stop defence drill."""
        self._defence_running = False
        self._defence_waiting_for_robot = False
        self._defence_attack_index = 0
        self._defence_block_count = 0
        self._defence_last_status_stamp = None
        self._stop_defence_status_timer()
        self._reset_defence_drill_ui()
        
        # Reset button
        self.defence_start_btn.setText("▶  START DEFENCE")
        self.defence_start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 14px;
                font-weight: 700;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        self.defence_start_btn.clicked.disconnect()
        self.defence_start_btn.clicked.connect(self._start_defence_drill)

    def _reset_defence_drill_ui(self) -> None:
        """Reset defence drill UI to ready state."""
        if self._defence_running:
            return
        self.defence_action_label.setText("READY")
        self.defence_action_label.setStyleSheet(
            "font-size: 34px; font-weight: 800; color: #ff8c00; background: transparent;"
        )
        drill_name = self.defence_combo.currentData() if hasattr(self, "defence_combo") else None
        if drill_name:
            self.defence_sub_label.setText(f"Selected: {drill_name}")
        else:
            self.defence_sub_label.setText("Block all incoming attacks")
        self.defence_status_label.setText("Status: idle")
        self.defence_progress_label.setText(f"Blocks: 0/{self._defence_total_blocks}")
        self.defence_elapsed_label.setText("Time: 0.0s")
        if hasattr(self, "defence_checkbox_progress"):
            self.defence_checkbox_progress.reset()
        self.block_indicator.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)

    def _start_defence_status_timer(self) -> None:
        """Poll robot action status while defence drill is active."""
        if self._defence_status_timer is None:
            self._defence_status_timer = QtCore.QTimer(self)
            self._defence_status_timer.timeout.connect(self._defence_check_robot_status)
        if self._defence_status_timer.isActive():
            self._defence_status_timer.stop()
        self._defence_status_timer.start(100)

    def _stop_defence_status_timer(self) -> None:
        """Stop polling robot action status."""
        if self._defence_status_timer is not None and self._defence_status_timer.isActive():
            self._defence_status_timer.stop()

    def _defence_check_robot_status(self) -> None:
        """Advance defence drill when robot reports action complete."""
        if not self._defence_running or not self._defence_waiting_for_robot:
            return
        with self.ros.lock:
            status = self.ros.robot_action_status
            status_stamp = self.ros.robot_action_status_stamp
        if status is None:
            return
        if status_stamp is None or status_stamp == self._defence_last_status_stamp:
            return
        self._defence_last_status_stamp = status_stamp
        if status != 0:
            return
        # status == 0: action complete
        self._defence_waiting_for_robot = False
        self._defence_attack_tick()
    
    def _show_defence_attack(self) -> None:
        """Show the current incoming attack prompt."""
        if not self._defence_running:
            return
        if self._defence_attack_index >= len(self._defence_attacks):
            return
        
        attack = self._defence_attacks[self._defence_attack_index]
        
        # Publish command to motors
        if hasattr(self.ros, 'motor_pub'):
            cmd_msg = String()
            cmd_msg.data = self._defence_attack_to_code(attack)
            self.ros.motor_pub.publish(cmd_msg)
        self._defence_waiting_for_robot = True
        with self.ros.lock:
            self._defence_last_status_stamp = self.ros.robot_action_status_stamp

        display = self._defence_display_for_attack(attack)
        self.defence_action_label.setText(f"🛡️ BLOCK {display}!")
        self.defence_action_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #ff4757; background: transparent;")
        self.defence_sub_label.setText(f"Attack {self._defence_attack_index + 1} of {self._defence_total_blocks}")
        
        # Flash the block indicator
        self.block_indicator.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 71, 87, 0.6), stop:1 rgba(200, 50, 60, 0.6));
                border: none;
                border-radius: 12px;
            }
        """)

    def _defence_display_for_attack(self, attack) -> str:
        """Map defence attack values to friendly labels."""
        labels = {
            1: "JAB",
            2: "CROSS",
            3: "LEFT HOOK",
            4: "RIGHT HOOK",
            5: "LEFT UPPERCUT",
            6: "RIGHT UPPERCUT",
        }
        try:
            val = int(attack)
        except Exception:
            val = None
        if val is not None:
            return labels.get(val, str(val))

        label = str(attack).strip().lower()
        text_labels = {
            "jab": "JAB",
            "cross": "CROSS",
            "left_hook": "LEFT HOOK",
            "right_hook": "RIGHT HOOK",
            "left_uppercut": "LEFT UPPERCUT",
            "right_uppercut": "RIGHT UPPERCUT",
        }
        return text_labels.get(label, str(attack).upper())

    def _defence_attack_to_code(self, attack) -> str:
        """Map defence attack labels to numeric codes for /robot/motor_command."""
        try:
            return str(int(attack))
        except Exception:
            pass

        if attack is None:
            return "0"

        label = str(attack).strip().lower()
        mapping = {
            "jab": "1",
            "cross": "2",
            "left_hook": "3",
            "right_hook": "4",
            "left_uppercut": "5",
            "right_uppercut": "6",
        }
        return mapping.get(label, "0")
    
    def _defence_attack_tick(self) -> None:
        """Handle completion of the current defence action."""
        if not self._defence_running:
            return
        
        # Robot reported completion; mark this step as blocked
        self.defence_checkbox_progress.tick(self._defence_attack_index)
        self._defence_block_count += 1
        self._defence_attack_index += 1
        
        self.defence_progress_label.setText(f"Blocks: {self._defence_block_count}/{self._defence_total_blocks}")
        
        # Show success briefly
        self.defence_action_label.setText("✓ BLOCKED!")
        self.defence_action_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #00ff00; background: transparent;")
        self.block_indicator.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 255, 0, 0.4), stop:1 rgba(0, 180, 0, 0.4));
                border: none;
                border-radius: 12px;
            }
        """)
        
        # Get coach tip
        tips = [
            "Great block! Keep those hands up!",
            "Nice defense! Stay focused!",
            "Perfect timing! You're doing great!",
        ]
        if self._defence_attack_index <= len(tips):
            self.defence_coach_bar.set_message(tips[self._defence_attack_index - 1])
        
        if self._defence_block_count >= self._defence_total_blocks:
            # Drill complete
            self._complete_defence_drill()
        else:
            # Show next attack after brief delay
            QtCore.QTimer.singleShot(self._defence_attack_interval_ms, self._show_defence_attack)
    
    def _complete_defence_drill(self) -> None:
        """Complete the defence drill."""
        self._defence_running = False
        self._defence_waiting_for_robot = False
        self._stop_defence_status_timer()
        elapsed = time.time() - self._defence_start_time
        
        self.defence_action_label.setText("COMPLETE! 🏆")
        self.defence_action_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #00ff00; background: transparent;")
        self.defence_sub_label.setText(f"Blocked all {self._defence_total_blocks} attacks!")
        self.defence_status_label.setText("Status: COMPLETE")
        self.defence_coach_bar.set_message(f"Excellent defense! All blocks in {elapsed:.1f}s! 💪")
        
        self.block_indicator.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0d0d0d);
                border: none;
                border-radius: 12px;
            }
        """)
        
        # Reset button
        self.defence_start_btn.setText("▶  START DEFENCE")
        self.defence_start_btn.setStyleSheet("""
            QPushButton {
                background: #ff8c00;
                color: #000000;
                font-size: 14px;
                font-weight: 700;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover { background: #ffa333; }
        """)
        self.defence_start_btn.clicked.disconnect()
        self.defence_start_btn.clicked.connect(self._start_defence_drill)

        # Return to READY after a brief moment
        QtCore.QTimer.singleShot(1500, self._reset_defence_drill_ui)
    
    def _toggle_imu_input(self, enabled: bool) -> None:
        """Toggle IMU input for menu selection."""
        if not self.ros.imu_input_client.service_is_ready():
            return
        req = SetBool.Request()
        req.data = enabled
        self.ros.imu_input_client.call_async(req)
    
    def _update_shadow_ui(self) -> None:
        """Update shadow sparring tab UI from ROS drill progress."""
        with self.ros.lock:
            progress = self.ros.drill_progress
            last_punch = self.ros.last_punch
            last_punch_stamp = self.ros.last_punch_stamp
            shadow_punch = self.ros.last_shadow_punch
            shadow_punch_stamp = self.ros.last_shadow_punch_stamp
            shadow_punch_counter = self.ros.shadow_punch_counter
            last_action = self.ros.last_action
        
        if progress is None or not progress.drill_name:
            # No drill active - show real-time glove detection preview
            # Use glove_detections for continuous state display (idle/jab/cross)
            self._update_shadow_preview_from_gloves()
            
            if hasattr(self, "shadow_punch_count_label"):
                self.shadow_punch_count_label.setText(f"Punches: {shadow_punch_counter}")
            
            # Use glove_detections for continuous DETECTED label
            self._update_shadow_detected_from_gloves()
            return

        current_step = progress.current_step
        total_steps = max(progress.total_steps, len(progress.expected_actions))
        expected = progress.expected_actions[current_step] if current_step < len(progress.expected_actions) else None
        status = progress.status

        self.shadow_progress_label.setText(f"Step: {current_step}/{total_steps}")
        self.shadow_elapsed_label.setText(f"Time: {progress.elapsed_time_s:.1f}s")
        
        # Update Score from summary
        iterations = 0
        failures = 0
        if isinstance(self.ros.drill_summary, dict):
             iterations = self.ros.drill_summary.get("iterations", 0)
             failures = self.ros.drill_summary.get("failures", 0)
        
        self.shadow_correct_label.setText(f"Correct: {iterations}")
        self.shadow_wrong_label.setText(f"Wrong: {failures}")

        # Visual feedback for wrong action (Check status OR failure count increase)
        
        # Track failures to trigger on increment
        if not hasattr(self, "_last_failures"):
            self._last_failures = 0
        if not hasattr(self, "_last_wrong_step"):
            self._last_wrong_step = -1
        if not hasattr(self, "_last_iterations"):
            self._last_iterations = 0
            
        new_failure = failures > self._last_failures
        new_success = iterations > self._last_iterations
        
        # Track which step was wrong (before reset)
        if new_failure:
            # The wrong step is tracked by comparing last known step before reset
            wrong_step = getattr(self, "_shadow_tracking_step", current_step)
            self._last_wrong_step = wrong_step
            self._last_failures = failures
            
            self._feedback_end_time = time.time() + 1.2  # Show for 1.2 seconds
            self.action_label.setText("WRONG! ❌")
            self.action_label.setStyleSheet(
                "font-size: 36px; font-weight: 800; color: #ff4757; "
                "background: transparent; padding: 4px 0px;"
            )
            # Mark the wrong step with X in checkbox
            try:
                self.shadow_checkbox_progress.set_wrong(wrong_step)
            except:
                pass
            
            # Show WRONG in combo result box
            self.shadow_combo_result.setText("✗ WRONG")
            self.shadow_combo_result.setStyleSheet("""
                QLabel {
                    background: #ff4757;
                    color: #fff;
                    font-size: 16px;
                    font-weight: 700;
                    border-radius: 8px;
                    padding: 4px 10px;
                    border: 2px solid #ff4757;
                }
            """)
            
            # Add to combo history
            if hasattr(self, 'shadow_combo_history'):
                self.shadow_combo_history.add_result('wrong')
        
        # Check for combo complete
        if new_success:
            self._last_iterations = iterations
            self._feedback_end_time = time.time() + 1.2
            
            # Show COMPLETE in combo result box
            self.shadow_combo_result.setText("✓ COMBO COMPLETE")
            self.shadow_combo_result.setStyleSheet("""
                QLabel {
                    background: #26d0ce;
                    color: #000;
                    font-size: 16px;
                    font-weight: 700;
                    border-radius: 8px;
                    padding: 4px 10px;
                    border: 2px solid #26d0ce;
                }
            """)
            
            # Add to combo history
            if hasattr(self, 'shadow_combo_history'):
                self.shadow_combo_history.add_result('success')
        
        # Clear combo result after feedback period
        if hasattr(self, "_feedback_end_time") and time.time() > self._feedback_end_time + 0.5:
            if self.shadow_combo_result.text():
                self.shadow_combo_result.setText("")
                self.shadow_combo_result.setStyleSheet("""
                    QLabel {
                        background: transparent;
                        color: #555;
                        font-size: 18px;
                        font-weight: 700;
                        border-radius: 8px;
                        padding: 6px 10px;
                    }
                """)
        
        # Track current step for next wrong detection
        self._shadow_tracking_step = current_step


        if expected:
            display_expected = expected.replace("_", " ").upper()
            
            # SHOW EXPECTED ACTION (unless overriding with feedback)
            if not (hasattr(self, "_feedback_end_time") and time.time() < self._feedback_end_time):
                self.shadow_expected_label.setText(f"Next: {display_expected}")
                color = "#26d0ce" if expected == "jab" else "#ff8c00" if expected == "cross" else "#ffa333"
                self.action_label.setText(display_expected)
                self.action_label.setStyleSheet(
                    f"font-size: 42px; font-weight: 800; color: {color}; "
                    "background: transparent; padding: 4px 0px;"
                )
            
            self.action_conf_label.setText(f"Step {current_step + 1}/{total_steps}")

        else:
            self.shadow_expected_label.setText("Throw: --")

        if status == "success":
            self.action_label.setText("COMPLETE! 🏆")
            self.action_label.setStyleSheet(
                "font-size: 36px; font-weight: 800; color: #00ff00; "
                "background: transparent; padding: 4px 0px;"
            )
            self.action_conf_label.setText(f"Time {progress.elapsed_time_s:.1f}s")
            self.shadow_detected_label.setText("DETECTED: --")
            if hasattr(self, "_shadow_detected_style"):
                self.shadow_detected_label.setStyleSheet(self._shadow_detected_style)
        elif status == "timeout":
            self.action_label.setText("TIME OUT")
            self.action_label.setStyleSheet(
                "font-size: 36px; font-weight: 800; color: #ff4757; "
                "background: transparent; padding: 4px 0px;"
            )
            self.shadow_detected_label.setText("DETECTED: --")
            if hasattr(self, "_shadow_detected_style"):
                self.shadow_detected_label.setStyleSheet(self._shadow_detected_style)
        elif status == "failed":
            self.action_label.setText("TRY AGAIN")
            self.action_label.setStyleSheet(
                "font-size: 36px; font-weight: 800; color: #ff4757; "
                "background: transparent; padding: 4px 0px;"
            )
            self.shadow_detected_label.setText("DETECTED: --")
            if hasattr(self, "_shadow_detected_style"):
                self.shadow_detected_label.setStyleSheet(self._shadow_detected_style)
            self.shadow_checkbox_progress.reset()
            self._last_shadow_step = 0
        
        if status == "in_progress":
            self.shadow_start_btn.setText("▶  START")
            self.shadow_stop_btn.setEnabled(True)
        else:
            self.shadow_start_btn.setText("▶  START")
            if hasattr(self, "_shadow_start_style"):
                self.shadow_start_btn.setStyleSheet(self._shadow_start_style)
            self.shadow_stop_btn.setEnabled(False)
            # Use real-time glove detection for preview when drill is not in_progress
            self._update_shadow_preview_from_gloves()

        # LIVE FEEDBACK (Always runs to prevent lag)
        # Use live glove state instead of action prediction
        match_expected = False
        if current_step < len(progress.expected_actions):
             exp_act = progress.expected_actions[current_step]
             # Check if our live glove state matches expected
             state, _, _ = self._get_realtime_glove_state()
             if state == exp_act:
                  match_expected = True

        # Use real-time glove detections for continuous DETECTED label during drill
        self._update_shadow_detected_from_gloves()
             
        # If we have a match, force green to show positive feedback
        if match_expected and self.shadow_detected_label.text() != "DETECTED: IDLE":
                  # Parse current text to preserve it
                  txt = self.shadow_detected_label.text()
                  self.shadow_detected_label.setStyleSheet("""
                    QLabel {
                        background: #222;
                        color: #00ff00;
                        font-size: 26px;
                        font-weight: 800;
                        border-radius: 8px;
                        padding: 6px 10px;
                        border: 2px solid #00ff00;
                    }
                """)

        # Update checkbox progress for completed steps
        completed = sum(progress.step_completed) if progress.step_completed else 0
        if not hasattr(self, "_last_shadow_step"):
            self._last_shadow_step = 0
        
        # Handle new iteration (success loop) - reset checkbox progress
        # Note: The combo result display is handled above in the new_success block
        if new_success:
            self.shadow_checkbox_progress.reset()
            self._last_shadow_step = 0
            
        # Handle reset/restart (completed dropped below last step)
        if completed < self._last_shadow_step:
             feedback_active = (hasattr(self, "_feedback_end_time") and time.time() < self._feedback_end_time)
             
             # If feedback active (wrong punch or combo complete), keep the X visible
             if feedback_active:
                 return
             
             # Otherwise (feedback expired and user ready to continue), perform reset
             self.shadow_checkbox_progress.reset()
             self._last_shadow_step = 0
             # Fall through to allow ticking up to current 'completed' if > 0
        
        if completed > self._last_shadow_step:
            for i in range(self._last_shadow_step, min(completed, total_steps)):
                self.shadow_checkbox_progress.tick(i)
        
        self._last_shadow_step = completed



    def _update_shadow_service_status(self) -> None:
        """Enable/disable shadow drill controls based on service availability."""
        if not hasattr(self, "shadow_start_btn"):
            return
        with self.ros.lock:
            progress = self.ros.drill_progress
        if progress and progress.status in {"in_progress", "success", "failed", "timeout"}:
            return
            
        # THROTTLE CHECK (Lag Fix) - Check every 2.0s
        now = time.time()
        if not hasattr(self, "_last_service_check"):
            self._last_service_check = 0
            self._cached_service_ready = False
            
        if now - self._last_service_check > 2.0:
             self._cached_service_ready = self.ros.shadow_drill_client.service_is_ready()
             self._last_service_check = now
             
        ready = self._cached_service_ready
        
        # Keep START enabled; enable/disable stop based on readiness
        self.shadow_start_btn.setEnabled(True)
        if not ready:
            self.shadow_stop_btn.setEnabled(False)
        else:
            self.shadow_stop_btn.setEnabled(True)

    def _update_shadow_preview_from_punch(self, punch, punch_stamp) -> bool:
        """Show last detected punch even when drill not running."""
        if punch is None or punch_stamp is None:
            return False
        if not punch.is_punch:
            return False
        
        # Only show punch if glove is close to camera (within 0.5m)
        distance = getattr(punch, "distance_m", 1.0)
        if distance > 0.5:
            return False
        
        # Check if punch is stale (older than 0.8s) - return False to allow action-based idle
        now = time.time()
        if isinstance(punch_stamp, tuple):
            ts = punch_stamp[0] + punch_stamp[1] * 1e-9
        else:
            ts = float(punch_stamp or 0)
        age = now - ts
        if age > 0.8:
            return False
        
        if self._shadow_last_preview_ts == punch_stamp:
            return False
        self._shadow_last_preview_ts = punch_stamp
        detected, display, color = self._shadow_label_for_punch(punch)
        if not detected:
            return False
        self.action_label.setText(display)
        self.action_conf_label.setText("Last punch")
        self.action_label.setStyleSheet(
            f"font-size: 34px; font-weight: 800; color: {color}; "
            "background: transparent; padding: 4px 0px;"
        )
        return True

    def _shadow_label_for_punch(self, punch):
        """Return (detected, display, color) for a punch event."""
        detected = ""
        if punch and punch.punch_type and punch.punch_type != "unknown":
            detected = punch.punch_type.lower()
        else:
            glove = (punch.glove or "").lower() if punch else ""
            if glove in ("left", "red"):
                detected = "jab"
            elif glove in ("right", "green"):
                detected = "cross"
        if not detected:
            return "", "", ""
        display = detected.replace("_", " ").upper()
        color = "#26d0ce" if detected == "jab" else "#ff8c00" if detected == "cross" else "#ffa333"
        return detected, display, color

    def _update_shadow_detected_from_punch(self, punch, punch_stamp=None) -> bool:
        """Update detected label from the latest punch event."""
        if punch is None or not getattr(punch, "is_punch", False):
            return False
        
        # Only show punch if glove is close to camera (within 0.5m)
        distance = getattr(punch, "distance_m", 1.0)
        if distance > 0.5:
            return False

        # Check for idle reset (0.8s timeout) - if punch is old, don't use it
        now = time.time()
        if punch_stamp:
            # Handle ROS2 Time tuple or float
            if isinstance(punch_stamp, tuple):
                 ts = punch_stamp[0] + punch_stamp[1] * 1e-9
            else:
                 ts = float(punch_stamp or 0)
            
            # If punch is older than 0.8s, return False to allow action-based detection
            age = now - ts
            if age > 0.8:
                return False
        else:
            # No timestamp means we can't validate freshness, default to idle
            return False

        detected, display, color = self._shadow_label_for_punch(punch)
        if not detected:
            return False
        self.shadow_detected_label.setText(f"DETECTED: {display}")
        self.shadow_detected_label.setStyleSheet(f"""
            QLabel {{
                background: #222;
                color: {color};
                font-size: 24px;
                font-weight: 800;
                border-radius: 8px;
                padding: 6px 10px;
                border: 2px solid {color};
            }}
        """)
        return True

    def _update_shadow_preview_from_action(self, action) -> bool:
        # If no action received or action is stale, show idle
        if action is None:
            self.action_label.setText("READY")
            self.action_conf_label.setText("Live")
            self.action_label.setStyleSheet(
                "font-size: 34px; font-weight: 800; color: #ff8c00; "
                "background: transparent;"
            )
            return True
        label = getattr(action, "action_label", None)
        
        # Handle Idle/Ready for Top Label
        if not label or label == "idle":
            self.action_label.setText("READY")
            self.action_conf_label.setText("Live")
            self.action_label.setStyleSheet(
                "font-size: 34px; font-weight: 800; color: #ff8c00; " # Orange default
                "background: transparent;"
            )
            return True
        
        display = label.replace("_", " ").upper()
        color = "#26d0ce" if label == "jab" else "#ff8c00" if label == "cross" else "#ffa333"
        self.action_label.setText(display)
        self.action_conf_label.setText("Live")
        self.action_label.setStyleSheet(
            f"font-size: 34px; font-weight: 800; color: {color}; "
            "background: transparent;"
        )
        return True


    def _update_shadow_detected_from_action(self, action) -> bool:
        # If no action received, show idle
        if action is None:
            self.shadow_detected_label.setText("DETECTED: IDLE")
            self.shadow_detected_label.setStyleSheet("""
                QLabel {
                    background: #222;
                    color: #888888;
                    font-size: 24px;
                    font-weight: 800;
                    border-radius: 8px;
                    padding: 6px 10px;
                    border: 2px solid #555555;
                }
            """)
            return True
        label = getattr(action, "action_label", None)
        
        # Handle Idle/Ready explicitly
        if not label or label == "idle":
            self.shadow_detected_label.setText("DETECTED: IDLE")
            self.shadow_detected_label.setStyleSheet("""
                QLabel {
                    background: #222;
                    color: #888888;
                    font-size: 24px;
                    font-weight: 800;
                    border-radius: 8px;
                    padding: 6px 10px;
                    border: 2px solid #555555;
                }
            """)
            return True
            
        display = label.replace("_", " ").upper()
        color = "#26d0ce" if label == "jab" else "#ff8c00" if label == "cross" else "#ffa333"
        self.shadow_detected_label.setText(f"DETECTED: {display}")
        self.shadow_detected_label.setStyleSheet(f"""
            QLabel {{
                background: #222;
                color: {color};
                font-size: 24px;
                font-weight: 800;
                border-radius: 8px;
                padding: 6px 10px;
                border: 2px solid {color};
            }}
        """)
        return True
    
    def _get_realtime_glove_state(self) -> tuple:
        """
        Get real-time punch state from glove_detections.
        Returns (state, display, color) where state is 'idle', 'jab', or 'cross'.
        
        A punch is only detected when:
        1. One glove is within the punch threshold (0.50m)
        2. That glove is significantly ahead of the other glove (by at least 0.15m)
        This prevents false positives when both gloves are slightly forward in neutral.
        """
        with self.ros.lock:
            detections = self.ros.last_detections
        
        if detections is None or not detections.detections:
            return ("idle", "IDLE", "#888888")
        
        # Distance threshold for considering a glove as "extended" (in punch position)
        PUNCH_DISTANCE_THRESHOLD = 0.50  # meters - glove must be this close
        # Minimum difference required between gloves to register a punch
        MIN_GLOVE_DIFFERENCE = 0.15  # meters - punching glove must be this much ahead
        
        left_dist = 999.0
        right_dist = 999.0
        
        for det in detections.detections:
            if det.glove == "left":
                left_dist = min(left_dist, det.distance_m)
            elif det.glove == "right":
                right_dist = min(right_dist, det.distance_m)
        
        # Check if left glove is punching (close AND significantly ahead of right)
        left_punching = (
            left_dist < PUNCH_DISTANCE_THRESHOLD and 
            (right_dist - left_dist) >= MIN_GLOVE_DIFFERENCE
        )
        
        # Check if right glove is punching (close AND significantly ahead of left)
        right_punching = (
            right_dist < PUNCH_DISTANCE_THRESHOLD and 
            (left_dist - right_dist) >= MIN_GLOVE_DIFFERENCE
        )
        
        if left_punching and right_punching:
            # Both meet criteria - pick the one that's more extended
            if left_dist < right_dist:
                return ("jab", "JAB", "#26d0ce")
            else:
                return ("cross", "CROSS", "#ff8c00")
        elif left_punching:
            return ("jab", "JAB", "#26d0ce")
        elif right_punching:
            return ("cross", "CROSS", "#ff8c00")
        else:
            return ("idle", "IDLE", "#888888")

    def _update_shadow_detected_from_gloves(self) -> bool:
        """
        Update the DETECTED label from real-time glove detections.
        This provides continuous feedback even when no punch event is triggered.
        """
        state, display, color = self._get_realtime_glove_state()
        
        self.shadow_detected_label.setText(f"DETECTED: {display}")
        if state == "idle":
            self.shadow_detected_label.setStyleSheet("""
                QLabel {
                    background: #222;
                    color: #888888;
                    font-size: 24px;
                    font-weight: 800;
                    border-radius: 8px;
                    padding: 6px 10px;
                    border: 2px solid #555555;
                }
            """)
        else:
            self.shadow_detected_label.setStyleSheet(f"""
                QLabel {{
                    background: #222;
                    color: {color};
                    font-size: 24px;
                    font-weight: 800;
                    border-radius: 8px;
                    padding: 6px 10px;
                    border: 2px solid {color};
                }}
            """)
        return True

    def _update_shadow_preview_from_gloves(self) -> bool:
        """
        Update the main action label from real-time glove detections.
        Shows JAB/CROSS/READY based on which glove is extended.
        """
        state, display, color = self._get_realtime_glove_state()
        
        if state == "idle":
            self.action_label.setText("READY")
            self.action_conf_label.setText("Live")
            self.action_label.setStyleSheet(
                "font-size: 34px; font-weight: 800; color: #ff8c00; background: transparent;"
            )
        else:
            self.action_label.setText(display)
            self.action_conf_label.setText("Live")
            self.action_label.setStyleSheet(
                f"font-size: 34px; font-weight: 800; color: {color}; background: transparent;"
            )
        return True

    def _update_defence_ui(self) -> None:
        """Update defence drill tab UI - only if NOT running local drill."""
        # Skip ROS updates if our local drill is running
        if hasattr(self, '_defence_running') and self._defence_running:
            return
        
        with self.ros.lock:
            action = self.ros.last_action
            progress = self.ros.drill_progress
        
        # Block detection (only if not in local drill mode)
        if action is not None and not (hasattr(self, '_defence_running') and self._defence_running):
            pass  # Let local drill control the display
        
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
        self._begin_shadow_drill_service()
    
    def _on_defence_countdown_done(self) -> None:
        """Handle defence countdown completion - start the actual drill."""
        self.stack.setCurrentWidget(self.defence_tab)
        self._start_defence_drill()
    
    def _on_blocking_zone_selected(self, zone: int) -> None:
        """Handle numpad button press for blocking zone selection."""
        # Update the action label to show selected zone
        zone_name = self._defence_display_for_attack(zone)
        
        # Flash the block indicator
        self.block_indicator.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(33, 150, 243, 0.8), stop:1 rgba(21, 101, 192, 0.8));
            border-radius: 16px;
            border: none;
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
        # Create a dedicated calibration countdown (don't reuse shadow_countdown)
        # Just show a simple dialog instead
        self.header.setText("📏 HEIGHT CALIBRATION")
        
        # Use a simple message box with countdown
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Height Calibration")
        msg.setText("Stand straight in front of the camera!\n\nCalibrating in 3 seconds...")
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        
        # Start calibration after brief delay
        def do_calibration():
            msg.close()
            self._trigger_height_calc()
        
        QtCore.QTimer.singleShot(3000, do_calibration)
        msg.exec()

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

    # Enable touchscreen support
    import os
    os.environ['QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS'] = ''
    os.environ['QT_QUICK_CONTROLS_STYLE'] = 'Material'
    
    app = QtWidgets.QApplication([])
    
    # Enable touch events and gestures for touchscreen
    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    
    # Set style hints for better touch support
    app.setStyleSheet(app.styleSheet() + """
        * {
            /* Ensure minimum touch target size */
        }
        QPushButton {
            min-height: 32px;
        }
    """)
    
    gui = BoxBunnyGui(ros_node)
    
    # Enable touch for the main window and all children
    gui.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
    
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
