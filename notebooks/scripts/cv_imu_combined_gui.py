#!/usr/bin/env python3
"""BoxBunny CV + IMU Fusion Debug GUI -- standalone Tkinter window showing
CV inference stats and Teensy simulator side by side.

Launch:  python3 notebooks/scripts/cv_imu_combined_gui.py
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import tkinter as tk
from collections import deque
from typing import Any, Optional

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    raise SystemExit(
        "rclpy not found. Source your ROS 2 workspace first:\n"
        "  source /opt/ros/humble/setup.bash"
    )

try:
    from boxbunny_msgs.msg import ConfirmedPunch, RobotCommand
except ImportError:
    raise SystemExit(
        "boxbunny_msgs not found. Build the workspace first:\n"
        "  cd boxing_robot_ws && colcon build --packages-select boxbunny_msgs"
    )

from std_msgs.msg import Float64MultiArray, String as StdString

# ---------------------------------------------------------------------------
# Theme (matches BoxBunny GUI palette)
# ---------------------------------------------------------------------------
BG = "#0B0F14"
SURFACE = "#131920"
SURFACE2 = "#1A2029"
SURFACE3 = "#222B37"
FG = "#E6EDF3"
FG_DIM = "#8B949E"
FG_MUTED = "#484F58"
PRIMARY = "#FF6B35"
BORDER = "#1C222A"
BORDER_LT = "#2A3340"
GREEN = "#56D364"
AMBER = "#FFAB40"
RED = "#FF5C5C"

PUNCH_JAB = "#58A6FF"
PUNCH_CROSS = "#FF5C5C"
PUNCH_HOOK = "#56D364"
PUNCH_UPPER = "#BC8CFF"
PUNCH_IDLE = "#484F58"

PAD_BG = "#1A1214"
PAD_FLASH = "#3D1A22"

FONT = "Helvetica"
FONT_M = "Monospace"

_PUNCH_COLORS: dict[str, str] = {
    "jab": PUNCH_JAB,
    "cross": PUNCH_CROSS,
    "l_hook": PUNCH_HOOK,
    "r_hook": PUNCH_HOOK,
    "l_upper": PUNCH_UPPER,
    "r_upper": PUNCH_UPPER,
    "hook": PUNCH_HOOK,
    "uppercut": PUNCH_UPPER,
    "block": AMBER,
    "idle": PUNCH_IDLE,
}

# IMU pad index -> pad name (user perspective, Teensy indices swapped)
_IMU_PAD_MAP: dict[int, str] = {0: "centre", 1: "right", 2: "left", 3: "head"}

# Fusion timing
_FUSION_WINDOW_SEC = 0.200  # 200ms to match IMU with CV

_FLASH_MS = 250

log = logging.getLogger("cv_imu_combined_gui")


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------
class CombinedDebugNode(Node):
    """Subscribes to CV debug, confirmed punches, strike detection, and
    motor feedback.  Publishes RobotCommand for the punch buttons."""

    def __init__(self) -> None:
        super().__init__("cv_imu_combined_gui")

        # Callbacks set by the GUI
        self.on_cv_debug: Optional[Any] = None
        self.on_confirmed_punch: Optional[Any] = None
        self.on_strike_detected: Optional[Any] = None
        self.on_motor_feedback: Optional[Any] = None

        # State
        self.teensy_connected: bool = False
        self.real_imu_accel: list[list[float]] = [
            [0.0, 0.0, 0.0] for _ in range(4)
        ]

        # Subscriptions
        self.create_subscription(
            StdString, "/boxbunny/cv/debug_info",
            self._on_cv_debug, 10,
        )
        self.create_subscription(
            ConfirmedPunch, "/boxbunny/punch/confirmed",
            self._on_confirmed_punch, 10,
        )
        self.create_subscription(
            StdString, "/robot/strike_detected",
            self._on_strike_detected, 10,
        )
        self.create_subscription(
            Float64MultiArray, "motor_feedback",
            self._on_motor_feedback, 10,
        )

        # Publisher for punch buttons
        self._pub_robot_cmd = self.create_publisher(
            RobotCommand, "/boxbunny/robot/command", 10,
        )

        self.get_logger().info("CV+IMU combined debug node started")

    # -- Subscription handlers -----------------------------------------------

    def _on_cv_debug(self, msg: StdString) -> None:
        try:
            data = json.loads(msg.data)
            if self.on_cv_debug:
                self.on_cv_debug(data)
        except Exception:
            pass

    def _on_confirmed_punch(self, msg: ConfirmedPunch) -> None:
        if self.on_confirmed_punch:
            self.on_confirmed_punch(msg)

    def _on_strike_detected(self, msg: StdString) -> None:
        try:
            data = json.loads(msg.data)
            self.teensy_connected = True
            if self.on_strike_detected:
                self.on_strike_detected(data)
        except Exception:
            pass

    def _on_motor_feedback(self, msg: Float64MultiArray) -> None:
        self.teensy_connected = True
        if len(msg.data) >= 21:
            for i in range(4):
                base = 9 + i * 3
                self.real_imu_accel[i] = [
                    msg.data[base], msg.data[base + 1], msg.data[base + 2],
                ]
        if self.on_motor_feedback:
            self.on_motor_feedback(msg.data)

    # -- Publishing ----------------------------------------------------------

    def publish_robot_command(self, punch_code: str) -> None:
        msg = RobotCommand()
        msg.command_type = "punch"
        msg.punch_code = punch_code
        msg.speed = "medium"
        self._pub_robot_cmd.publish(msg)
        self.get_logger().info(f"RobotCommand punch_code={punch_code}")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class CombinedGUI:
    """Three-panel Tkinter GUI: CV predictions, fusion log, IMU pads."""

    def __init__(self, node: CombinedDebugNode) -> None:
        self._node = node

        # CV state
        self._cv_action: str = "idle"
        self._cv_confidence: float = 0.0
        self._cv_consecutive: int = 0
        self._cv_fps: float = 0.0
        self._cv_raw: str = ""
        self._cv_probs: dict[str, float] = {}
        self._cv_trail: deque[str] = deque(maxlen=20)
        self._last_cv_time: float = 0.0
        self._cv_model_status: str = "Loading..."

        # Start local inference engine in background thread
        self._inference_running = False
        self._start_inference_thread()

        # IMU state
        self._last_imu_event: Optional[dict] = None
        self._last_imu_time: float = 0.0

        # Teensy state
        self._motor_positions: list[float] = [0.0] * 4
        self._motor_currents: list[float] = [0.0] * 4
        self._imu_magnitudes: list[float] = [0.0] * 4

        # Fusion log
        self._fusion_lines: deque[tuple[str, str]] = deque(maxlen=60)

    def _start_inference_thread(self) -> None:
        """Launch the CV inference engine in a background thread."""
        import sys, os
        ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ap_dir = os.path.join(ws, "action_prediction")
        if ap_dir not in sys.path:
            sys.path.insert(0, ap_dir)

        def _run():
            try:
                from lib.inference_runtime import InferenceEngine
                checkpoint = os.path.join(ap_dir, "model", "best_model.pth")
                yolo = os.path.join(ap_dir, "model", "yolo26n-pose.pt")
                self._cv_model_status = "Loading model..."
                engine = InferenceEngine(
                    checkpoint_path=checkpoint,
                    yolo_model_path=yolo,
                    device="cuda:0",
                )
                engine.initialize()
                self._cv_model_status = "Loading camera..."
                self._inference_running = True

                # Open camera
                import cv2, numpy as np
                try:
                    import pyrealsense2 as rs
                    pipe = rs.pipeline()
                    cfg = rs.config()
                    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
                    align = rs.align(rs.stream.color)
                    pipe.start(cfg)
                    self._cv_model_status = "Running"

                    while self._inference_running:
                        try:
                            frames = pipe.wait_for_frames(timeout_ms=1000)
                        except:
                            continue
                        aligned = align.process(frames)
                        color = aligned.get_color_frame()
                        depth = aligned.get_depth_frame()
                        if not color or not depth:
                            continue
                        rgb = np.asanyarray(color.get_data())
                        dep = np.asanyarray(depth.get_data())
                        result = engine.process_frame(rgb, dep)
                        if result:
                            self._cv_action = result.action
                            self._cv_confidence = result.confidence
                            self._cv_consecutive = result.consecutive_frames
                            self._cv_fps = result.fps
                            self._cv_raw = result.raw_action
                            self._cv_trail.append(result.action)
                            self._last_cv_time = time.time()
                            if result.smooth_probs is not None and hasattr(engine, 'labels'):
                                probs = {}
                                for idx, label in enumerate(engine.labels):
                                    if idx < len(result.smooth_probs):
                                        short = label.replace("left_", "l_").replace("right_", "r_")
                                        probs[short] = float(result.smooth_probs[idx])
                                self._cv_probs = probs
                    pipe.stop()
                except Exception as cam_err:
                    self._cv_model_status = f"Camera error: {cam_err}"
            except Exception as e:
                self._cv_model_status = f"Model error: {e}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Build window
        self._root = tk.Tk()
        self._root.title("BoxBunny CV + IMU Fusion Debug")
        self._root.configure(bg=BG)
        self._root.geometry("1100x700")
        self._root.resizable(True, True)

        self._pad_btns: dict[str, tk.Button] = {}
        self._build()

        # Wire callbacks
        self._node.on_cv_debug = self._handle_cv_debug
        self._node.on_confirmed_punch = self._handle_confirmed_punch
        self._node.on_strike_detected = self._handle_strike_detected
        self._node.on_motor_feedback = self._handle_motor_feedback

    # -----------------------------------------------------------------------
    # Build UI
    # -----------------------------------------------------------------------
    def _build(self) -> None:
        r = self._root

        # Title bar
        top = tk.Frame(r, bg=SURFACE, height=40)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="Box", font=(FONT, 14, "bold"),
                 bg=SURFACE, fg=FG).pack(side="left", padx=(12, 0))
        tk.Label(top, text="Bunny", font=(FONT, 14, "bold"),
                 bg=SURFACE, fg=PRIMARY).pack(side="left")
        tk.Label(top, text="CV + IMU Fusion Debug", font=(FONT, 12),
                 bg=SURFACE, fg=FG_DIM).pack(side="left", padx=8)

        # Three-column body
        body = tk.Frame(r, bg=BG)
        body.pack(fill="both", expand=True, padx=6, pady=6)
        body.columnconfigure(0, weight=4, minsize=320)
        body.columnconfigure(1, weight=2, minsize=200)
        body.columnconfigure(2, weight=4, minsize=320)
        body.rowconfigure(0, weight=1)

        self._build_cv_panel(body)
        self._build_fusion_panel(body)
        self._build_imu_panel(body)

        # Status bar
        self._build_status_bar(r)

    # -- Left panel: CV Predictions -----------------------------------------

    def _build_cv_panel(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        tk.Label(frame, text="CV PREDICTIONS", font=(FONT, 12, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10, pady=(8, 4))

        # Current prediction (big)
        self._lbl_action = tk.Label(
            frame, text="IDLE", font=(FONT, 28, "bold"),
            bg=SURFACE, fg=PUNCH_IDLE,
        )
        self._lbl_action.pack(anchor="w", padx=10)

        # Stats row
        stats = tk.Frame(frame, bg=SURFACE)
        stats.pack(fill="x", padx=10, pady=(2, 0))

        self._lbl_conf = tk.Label(stats, text="Conf: 0%",
                                  font=(FONT_M, 12), bg=SURFACE, fg=FG_DIM)
        self._lbl_conf.pack(side="left")
        self._lbl_consec = tk.Label(stats, text="Consec: 0",
                                    font=(FONT_M, 12), bg=SURFACE, fg=FG_DIM)
        self._lbl_consec.pack(side="left", padx=(12, 0))
        self._lbl_fps = tk.Label(stats, text="FPS: --",
                                 font=(FONT_M, 12), bg=SURFACE, fg=FG_DIM)
        self._lbl_fps.pack(side="left", padx=(12, 0))

        # Raw label
        self._lbl_raw = tk.Label(frame, text="Raw: --",
                                 font=(FONT_M, 11), bg=SURFACE, fg=FG_MUTED)
        self._lbl_raw.pack(anchor="w", padx=10, pady=(2, 0))

        # Separator
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)

        # Probability bars
        tk.Label(frame, text="Probabilities", font=(FONT, 11, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10)
        self._prob_frame = tk.Frame(frame, bg=SURFACE)
        self._prob_frame.pack(fill="x", padx=10, pady=(2, 0))
        self._prob_bars: dict[str, tuple[tk.Label, tk.Canvas]] = {}
        for cls in ["jab", "cross", "hook", "uppercut", "block", "idle"]:
            row = tk.Frame(self._prob_frame, bg=SURFACE)
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text=cls, font=(FONT_M, 11), bg=SURFACE,
                           fg=FG_DIM, width=8, anchor="w")
            lbl.pack(side="left")
            canvas = tk.Canvas(row, height=16, bg=SURFACE2,
                               highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True, padx=(4, 0))
            self._prob_bars[cls] = (lbl, canvas)

        # Separator
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)

        # Prediction trail
        tk.Label(frame, text="Trail (last 20)", font=(FONT, 11, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10)
        self._trail_canvas = tk.Canvas(frame, height=26, bg=SURFACE,
                                       highlightthickness=0)
        self._trail_canvas.pack(fill="x", padx=10, pady=(2, 6))

    # -- Middle panel: Fusion Log -------------------------------------------

    def _build_fusion_panel(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=1, sticky="nsew", padx=3)

        tk.Label(frame, text="FUSION LOG", font=(FONT, 12, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10, pady=(8, 4))

        self._fusion_text = tk.Text(
            frame, bg=SURFACE, fg=FG_DIM, font=(FONT_M, 11),
            wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, state="disabled",
            insertbackground=FG,
        )
        self._fusion_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Configure tags for color coding
        self._fusion_text.tag_configure("confirmed", foreground=GREEN)
        self._fusion_text.tag_configure("rejected", foreground=RED)
        self._fusion_text.tag_configure("imu_only", foreground=AMBER)
        self._fusion_text.tag_configure("cv_event", foreground=FG_DIM)
        self._fusion_text.tag_configure("imu_event", foreground=AMBER)
        self._fusion_text.tag_configure("idle", foreground=FG_MUTED)

    # -- Right panel: IMU Pads + Teensy -------------------------------------

    def _build_imu_panel(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=2, sticky="nsew", padx=(3, 0))

        tk.Label(frame, text="IMU PADS", font=(FONT, 12, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10, pady=(8, 4))

        # Pad grid
        pad_frame = tk.Frame(frame, bg=SURFACE)
        pad_frame.pack(padx=10, pady=(0, 4))

        self._pad_btns["head"] = self._make_pad_btn(pad_frame, "HEAD", 0, 1)
        self._pad_btns["left"] = self._make_pad_btn(pad_frame, "LEFT", 1, 0)
        self._pad_btns["centre"] = self._make_pad_btn(pad_frame, "CENTRE", 1, 1)
        self._pad_btns["right"] = self._make_pad_btn(pad_frame, "RIGHT", 1, 2)

        # Force display
        force_frame = tk.Frame(frame, bg=SURFACE)
        force_frame.pack(fill="x", padx=10, pady=(2, 0))
        tk.Label(force_frame, text="Force:", font=(FONT, 11),
                 bg=SURFACE, fg=FG_DIM).pack(side="left")
        self._force_canvas = tk.Canvas(force_frame, height=18, bg=SURFACE2,
                                       highlightthickness=0, width=100)
        self._force_canvas.pack(side="left", padx=(4, 0), fill="x", expand=True)
        self._lbl_force = tk.Label(force_frame, text="0 m/s\u00B2",
                                   font=(FONT_M, 11), bg=SURFACE, fg=FG_DIM)
        self._lbl_force.pack(side="left", padx=(4, 0))

        # Separator
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)

        # Teensy live data
        tk.Label(frame, text="TEENSY LIVE", font=(FONT, 11, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10)

        teensy_frame = tk.Frame(frame, bg=SURFACE)
        teensy_frame.pack(fill="x", padx=10, pady=(2, 0))

        self._lbl_teensy_pos = tk.Label(
            teensy_frame, text="Pos: -- -- -- --",
            font=(FONT_M, 11), bg=SURFACE, fg=FG_DIM, anchor="w",
        )
        self._lbl_teensy_pos.pack(fill="x")
        self._lbl_teensy_cur = tk.Label(
            teensy_frame, text="Cur: -- -- -- --",
            font=(FONT_M, 11), bg=SURFACE, fg=FG_DIM, anchor="w",
        )
        self._lbl_teensy_cur.pack(fill="x")

        # IMU magnitudes per pad
        self._lbl_teensy_imu = tk.Label(
            teensy_frame, text="IMU: -- -- -- --",
            font=(FONT_M, 11), bg=SURFACE, fg=FG_DIM, anchor="w",
        )
        self._lbl_teensy_imu.pack(fill="x")

        # Separator
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)

        # Punch buttons
        tk.Label(frame, text="SEND PUNCH", font=(FONT, 11, "bold"),
                 bg=SURFACE, fg=FG_MUTED).pack(anchor="w", padx=10)

        btn_frame = tk.Frame(frame, bg=SURFACE)
        btn_frame.pack(fill="x", padx=10, pady=(2, 6))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        punch_defs = [
            ("Jab", "1", PUNCH_JAB, 0, 0),
            ("Cross", "2", PUNCH_CROSS, 0, 1),
            ("Hook", "3", PUNCH_HOOK, 1, 0),
            ("Upper", "5", PUNCH_UPPER, 1, 1),
        ]
        for label, code, color, row, col in punch_defs:
            tk.Button(
                btn_frame, text=label, font=(FONT, 12, "bold"),
                bg=SURFACE2, fg=color,
                activebackground=color, activeforeground="#000",
                relief="flat", bd=0, pady=6,
                command=lambda c=code: self._node.publish_robot_command(c),
            ).grid(row=row, column=col, sticky="ew", padx=1, pady=1)

    # -- Status bar ---------------------------------------------------------

    def _build_status_bar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=SURFACE2, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._lbl_status_conn = tk.Label(
            bar, text="Connected", font=(FONT, 11),
            bg=SURFACE2, fg=GREEN,
        )
        self._lbl_status_conn.pack(side="left", padx=(12, 0))

        tk.Label(bar, text="|", font=(FONT, 11),
                 bg=SURFACE2, fg=FG_MUTED).pack(side="left", padx=6)

        self._lbl_status_fps = tk.Label(
            bar, text="Pred FPS: --", font=(FONT, 11),
            bg=SURFACE2, fg=FG_DIM,
        )
        self._lbl_status_fps.pack(side="left")

        tk.Label(bar, text="|", font=(FONT, 11),
                 bg=SURFACE2, fg=FG_MUTED).pack(side="left", padx=6)

        self._lbl_status_imu = tk.Label(
            bar, text="IMU: --", font=(FONT, 11),
            bg=SURFACE2, fg=FG_MUTED,
        )
        self._lbl_status_imu.pack(side="left")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_pad_btn(self, parent: tk.Frame, text: str,
                      row: int, col: int) -> tk.Button:
        btn = tk.Button(
            parent, text=text, font=(FONT, 11, "bold"),
            bg=PAD_BG, fg=FG_DIM,
            activebackground=PAD_FLASH, activeforeground=FG,
            relief="flat", bd=0, width=8, height=3,
        )
        btn.grid(row=row, column=col, padx=2, pady=2)
        return btn

    def _flash_pad(self, pad_name: str) -> None:
        """Flash a pad button briefly."""
        btn = self._pad_btns.get(pad_name)
        if not btn:
            return
        btn.configure(bg=PAD_FLASH, fg=FG)
        self._root.after(_FLASH_MS, lambda: btn.configure(bg=PAD_BG, fg=FG_DIM))

    def _add_fusion_line(self, text: str, tag: str) -> None:
        """Append a line to the fusion log."""
        self._fusion_lines.append((text, tag))
        self._fusion_text.configure(state="normal")
        self._fusion_text.insert("end", text + "\n", tag)
        self._fusion_text.see("end")
        self._fusion_text.configure(state="disabled")

    def _update_prob_bars(self) -> None:
        """Redraw probability bar canvases."""
        probs = dict(self._cv_probs)  # snapshot for thread safety
        for cls, (lbl, canvas) in self._prob_bars.items():
            prob = probs.get(cls, 0.0)
            canvas.delete("all")
            w = canvas.winfo_width()
            if w <= 1:
                w = 120
            bar_w = int(w * prob)
            color = _PUNCH_COLORS.get(cls, FG_DIM)
            if bar_w > 0:
                canvas.create_rectangle(0, 0, bar_w, 14, fill=color, outline="")
            # Percentage text
            canvas.create_text(
                w - 4, 6, text=f"{prob * 100:.0f}%",
                anchor="e", fill=FG_DIM, font=(FONT_M, 7),
            )

    def _update_trail(self) -> None:
        """Draw prediction trail as colored dots."""
        self._trail_canvas.delete("all")
        x = 4
        for action in list(self._cv_trail):
            color = _PUNCH_COLORS.get(action, FG_MUTED)
            self._trail_canvas.create_oval(x, 4, x + 12, 16, fill=color,
                                           outline="")
            # Label letter
            letter = action[0].upper() if action else "?"
            self._trail_canvas.create_text(
                x + 6, 10, text=letter, fill="#000",
                font=(FONT, 7, "bold"),
            )
            x += 16

    def _update_force_bar(self, accel: float) -> None:
        """Update the force bar display."""
        self._force_canvas.delete("all")
        w = self._force_canvas.winfo_width()
        if w <= 1:
            w = 100
        max_accel = 60.0
        bar_w = int(w * min(accel / max_accel, 1.0))
        if accel < 15:
            color = GREEN
        elif accel < 35:
            color = AMBER
        else:
            color = RED
        if bar_w > 0:
            self._force_canvas.create_rectangle(0, 0, bar_w, 14,
                                                fill=color, outline="")
        self._lbl_force.configure(text=f"{accel:.0f} m/s\u00B2")

    # -----------------------------------------------------------------------
    # ROS callback handlers (called from ROS spin thread)
    # -----------------------------------------------------------------------

    def _handle_cv_debug(self, data: dict) -> None:
        """Process CV debug JSON from /boxbunny/cv/debug_info."""
        self._cv_action = data.get("action", "idle")
        self._cv_confidence = float(data.get("confidence", 0.0))
        self._cv_consecutive = int(data.get("consecutive", 0))
        self._cv_fps = float(data.get("fps", 0.0))
        self._cv_raw = data.get("raw", "")
        self._cv_probs = data.get("probabilities", {})
        self._last_cv_time = time.time()

        # Add to trail (only non-idle with some confidence)
        if self._cv_action != "idle" and self._cv_confidence > 0.3:
            self._cv_trail.append(self._cv_action)

        # Schedule GUI update on main thread
        try:
            self._root.after_idle(self._refresh_cv_panel)
        except tk.TclError:
            pass

        # Fusion logic: log CV event
        try:
            self._root.after_idle(lambda: self._fusion_cv_event(data))
        except tk.TclError:
            pass

    def _handle_confirmed_punch(self, msg: ConfirmedPunch) -> None:
        """Process confirmed punch from fusion node."""
        tag = "confirmed" if msg.imu_confirmed else "cv_event"
        source = "CV+IMU" if msg.imu_confirmed else "CV-only"
        text = (
            f"\u2192 CONFIRMED {msg.punch_type} ({source}) "
            f"conf={msg.cv_confidence:.2f}"
        )
        try:
            self._root.after_idle(lambda: self._add_fusion_line(text, tag))
        except tk.TclError:
            pass

    def _handle_strike_detected(self, data: dict) -> None:
        """Process IMU strike from /robot/strike_detected."""
        pad_idx = int(data.get("pad_index", 0))
        pad_name = _IMU_PAD_MAP.get(pad_idx, data.get("pad_name", "unknown"))
        peak = float(data.get("peak_accel", 0.0))

        self._last_imu_event = data
        self._last_imu_time = time.time()

        try:
            self._root.after_idle(lambda: self._flash_pad(pad_name))
            self._root.after_idle(lambda: self._update_force_bar(peak))
            self._root.after_idle(
                lambda: self._add_fusion_line(
                    f"IMU: {pad_name} {peak:.1f} m/s\u00B2", "imu_event",
                )
            )
            self._root.after_idle(lambda: self._fusion_imu_event(
                pad_name, peak,
            ))
        except tk.TclError:
            pass

    def _handle_motor_feedback(self, data: list[float]) -> None:
        """Process motor_feedback Float64MultiArray data."""
        if len(data) < 21:
            return
        self._motor_positions = [data[i] for i in range(4)]
        self._motor_currents = [data[4 + i] for i in range(4)]
        # IMU magnitudes per pad
        for i in range(4):
            base = 9 + i * 3
            ax, ay, az = data[base], data[base + 1], data[base + 2]
            self._imu_magnitudes[i] = math.sqrt(ax * ax + ay * ay + az * az)

        try:
            self._root.after_idle(self._refresh_teensy_panel)
        except tk.TclError:
            pass

    # -----------------------------------------------------------------------
    # Fusion logic
    # -----------------------------------------------------------------------

    def _fusion_cv_event(self, data: dict) -> None:
        """Log a CV event and attempt fusion with recent IMU data."""
        action = data.get("action", "idle")
        conf = float(data.get("confidence", 0.0))
        consec = int(data.get("consecutive", 0))

        if action == "idle":
            return

        # Log the CV detection
        self._add_fusion_line(
            f"CV: {action} {conf:.2f} ({consec} frames)", "cv_event",
        )

        # Check for recent IMU match
        now = time.time()
        if (self._last_imu_event
                and (now - self._last_imu_time) < _FUSION_WINDOW_SEC):
            self._add_fusion_line(
                f"\u2192 CONFIRMED {action} (CV+IMU)", "confirmed",
            )
            self._last_imu_event = None
        elif consec >= 3 and conf >= 0.6:
            self._add_fusion_line(
                f"\u2192 CV-ONLY accepted ({consec}f, {conf:.0%})", "confirmed",
            )
        else:
            self._add_fusion_line(
                f"\u2192 REJECTED (CV only, {consec} frame)", "rejected",
            )

    def _fusion_imu_event(self, pad_name: str, peak: float) -> None:
        """Check if a recent CV prediction matches this IMU event."""
        now = time.time()
        if (now - self._last_cv_time) < _FUSION_WINDOW_SEC:
            # Already handled from CV side; skip double logging
            return
        # IMU-only event (no matching CV)
        self._add_fusion_line(
            f"\u2192 IMU-only strike on {pad_name}", "imu_only",
        )

    # -----------------------------------------------------------------------
    # GUI refresh (always called on main thread)
    # -----------------------------------------------------------------------

    def _refresh_cv_panel(self) -> None:
        """Update all CV panel widgets."""
        action = self._cv_action
        color = _PUNCH_COLORS.get(action, FG_DIM)

        self._lbl_action.configure(text=action.upper(), fg=color)
        self._lbl_conf.configure(
            text=f"Conf: {self._cv_confidence * 100:.0f}%",
        )
        self._lbl_consec.configure(text=f"Consec: {self._cv_consecutive}")
        self._lbl_fps.configure(text=f"FPS: {self._cv_fps:.1f}")
        self._lbl_raw.configure(text=f"Raw: {self._cv_raw}")

        self._update_prob_bars()
        self._update_trail()

        # Status bar
        self._lbl_status_fps.configure(
            text=f"Pred FPS: {self._cv_fps:.0f}",
        )

    def _refresh_teensy_panel(self) -> None:
        """Update Teensy live data labels."""
        pos_str = "  ".join(f"{p:+.1f}" for p in self._motor_positions)
        cur_str = "  ".join(f"{c:.2f}A" for c in self._motor_currents)
        imu_str = "  ".join(f"{m:.1f}" for m in self._imu_magnitudes)

        self._lbl_teensy_pos.configure(text=f"Pos: {pos_str}")
        self._lbl_teensy_cur.configure(text=f"Cur: {cur_str}")
        self._lbl_teensy_imu.configure(text=f"IMU: {imu_str}")

        # Update status bar IMU indicator
        if self._node.teensy_connected:
            self._lbl_status_imu.configure(text="IMU: LIVE", fg=GREEN)
        else:
            self._lbl_status_imu.configure(text="IMU: --", fg=FG_MUTED)

    # -----------------------------------------------------------------------
    # Periodic update for status bar
    # -----------------------------------------------------------------------

    def _periodic_update(self) -> None:
        """Called every 500ms — refresh CV panel + status bar."""
        now = time.time()

        # Refresh CV panel from local inference engine
        self._refresh_cv_panel()
        self._refresh_teensy_panel()

        # Connection status — show model status
        cv_recent = (now - self._last_cv_time) < 3.0 if self._last_cv_time else False
        if cv_recent:
            self._lbl_status_conn.configure(text=f"CV: {self._cv_model_status}", fg=GREEN)
        else:
            self._lbl_status_conn.configure(text=f"CV: {self._cv_model_status}", fg=AMBER if "Loading" in self._cv_model_status else FG_MUTED)

        # IMU status
        if self._node.teensy_connected:
            self._lbl_status_imu.configure(text="IMU: LIVE", fg=GREEN)
        else:
            imu_recent = (
                (now - self._last_imu_time) < 3.0 if self._last_imu_time else False
            )
            if imu_recent:
                self._lbl_status_imu.configure(text="IMU: active", fg=AMBER)
            else:
                self._lbl_status_imu.configure(
                    text="IMU: disconnected", fg=FG_MUTED,
                )

        try:
            self._root.after(500, self._periodic_update)
        except tk.TclError:
            pass

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Start the periodic update and enter Tk mainloop."""
        self._root.after(500, self._periodic_update)
        self._root.mainloop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    rclpy.init()
    node = CombinedDebugNode()

    # Spin ROS in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    gui = CombinedGUI(node)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
