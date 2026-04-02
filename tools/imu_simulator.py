#!/usr/bin/env python3
"""BoxBunny IMU Simulator -- standalone Tkinter GUI that publishes the same
ROS 2 messages as the real Teensy hardware so drills can be developed and
tested without physical sensors.

Launch:  ros2 run boxbunny_core imu_simulator   (or just: python3 tools/imu_simulator.py)
"""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from collections import deque

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    raise SystemExit(
        "rclpy not found. Source your ROS 2 workspace first:\n"
        "  source /opt/ros/humble/setup.bash"
    )

try:
    from boxbunny_msgs.msg import (
        ArmStrike, ConfirmedPunch, IMUStatus, PadImpact, RobotCommand,
    )
except ImportError:
    raise SystemExit(
        "boxbunny_msgs not found. Build the workspace first:\n"
        "  cd boxing_robot_ws && colcon build --packages-select boxbunny_msgs"
    )

# ── Constants ────────────────────────────────────────────────────────────
_TOPIC_PAD = "/boxbunny/imu/pad/impact"
_TOPIC_ARM = "/boxbunny/imu/arm/strike"
_TOPIC_STATUS = "/boxbunny/imu/status"
_TOPIC_PUNCH = "/boxbunny/punch/confirmed"
_TOPIC_ROBOT_CMD = "/boxbunny/robot/command"
_FLASH_MS = 250

# Punch code -> internal punch type name
_CODE_TO_PUNCH = {
    "1": "jab", "2": "cross", "3": "l_hook", "4": "r_hook",
    "5": "l_upper", "6": "r_upper",
}

# Punch type definitions: name -> (arm, pad)
_PUNCH_TYPES = {
    "jab":       ("left",  "centre"),
    "cross":     ("right", "centre"),
    "l_hook":    ("left",  "left"),
    "r_hook":    ("right", "right"),
    "l_upper":   ("left",  "centre"),
    "r_upper":   ("right", "centre"),
}

# Punch colours — match GUI theme
PUNCH_JAB    = "#58A6FF"
PUNCH_CROSS  = "#FF5C5C"
PUNCH_HOOK   = "#56D364"
PUNCH_UPPER  = "#BC8CFF"

# ── Theme — matches BoxBunny GUI palette ─────────────────────────────────
BG         = "#0B0F14"
SURFACE    = "#131920"
SURFACE2   = "#1A2029"
SURFACE3   = "#222B37"  # hover state
FG         = "#E6EDF3"
FG_DIM     = "#8B949E"
FG_MUTED   = "#484F58"
PRIMARY    = "#FF6B35"
PRIMARY_DK = "#E85E2C"
BORDER     = "#1C222A"
BORDER_LT  = "#2A3340"

GREEN   = "#56D364"
AMBER   = "#FFAB40"
RED     = "#FF5C5C"

PAD_BG     = "#1A1214"
PAD_FLASH  = "#3D1A22"

ARM_BG     = "#101820"
ARM_FLASH  = "#1A2E40"

SEQ_BG = "#0E1319"

FONT   = "Helvetica"
FONT_M = "Monospace"

log = logging.getLogger("imu_simulator")


class IMUSimulatorNode(Node):
    """Thin ROS 2 node with publishers and a robot-command subscriber.

    Also subscribes to real hardware topics (motor_feedback, strike_detected)
    so the simulator mirrors the physical robot state in dev mode.
    """

    def __init__(self) -> None:
        super().__init__("imu_simulator")
        self._pub_pad = self.create_publisher(PadImpact, _TOPIC_PAD, 10)
        self._pub_arm = self.create_publisher(ArmStrike, _TOPIC_ARM, 10)
        self._pub_status = self.create_publisher(IMUStatus, _TOPIC_STATUS, 10)
        self._pub_punch = self.create_publisher(ConfirmedPunch, _TOPIC_PUNCH, 10)
        self.create_timer(1.0, self._publish_status)

        # Publisher for robot punch commands (so simulator can command real arms)
        self._pub_robot_cmd = self.create_publisher(RobotCommand, _TOPIC_ROBOT_CMD, 10)

        # Subscribe to robot commands so the GUI drill cycling shows up
        self.create_subscription(
            RobotCommand, _TOPIC_ROBOT_CMD, self._on_robot_command, 10,
        )
        self._robot_cmd_callback = None  # set by GUI

        # ── Real hardware mirror subscriptions ───────────────────────────
        # Subscribe to Teensy strike detection so simulator flashes when
        # real pads are hit (dev mode runs both hardware and simulator)
        from std_msgs.msg import String as StdString, Float64MultiArray
        self.create_subscription(
            StdString, "/robot/strike_detected",
            self._on_real_strike, 10,
        )
        # Subscribe to motor_feedback for real-time IMU accel display
        self.create_subscription(
            Float64MultiArray, "motor_feedback",
            self._on_motor_feedback, 10,
        )
        self._real_strike_callback = None   # set by GUI
        self._motor_feedback_callback = None  # set by GUI
        self._real_imu_accel = [[0.0, 0.0, 0.0] for _ in range(4)]
        self._teensy_connected = False

        self.get_logger().info("IMU simulator node started")

    def _on_real_strike(self, msg) -> None:
        """Handle strike detection from real Teensy hardware."""
        import json as _json
        try:
            data = _json.loads(msg.data)
            self._teensy_connected = True
            if self._real_strike_callback:
                self._real_strike_callback(data)
        except Exception:
            pass

    def _on_motor_feedback(self, msg) -> None:
        """Handle motor feedback from Teensy for live position/current display.

        Strike detection is NOT done here — the V4 arm control GUI already
        does gravity calibration + strike detection and publishes the result
        on /robot/strike_detected.  We just display the raw values.
        """
        self._teensy_connected = True
        if len(msg.data) >= 21:
            for i in range(4):
                base = 9 + i * 3
                self._real_imu_accel[i] = [msg.data[base], msg.data[base+1], msg.data[base+2]]
        if self._motor_feedback_callback:
            self._motor_feedback_callback(msg.data)

    def _on_robot_command(self, msg: RobotCommand) -> None:
        """Forward incoming robot commands to the GUI for visual flash."""
        if msg.command_type == "punch" and self._robot_cmd_callback:
            punch_type = _CODE_TO_PUNCH.get(msg.punch_code, "")
            if punch_type:
                self._robot_cmd_callback(punch_type)

    def publish_pad(self, pad: str, level: str, accel: float = 0.0) -> None:
        msg = PadImpact()
        msg.timestamp = time.time()
        msg.pad = pad
        msg.level = level
        msg.accel_magnitude = accel
        self._pub_pad.publish(msg)
        self.get_logger().info(f"PadImpact pad={pad} level={level} accel={accel:.1f}")

    def publish_punch(self, punch_type: str, level: str,
                      force_normalized: float = 0.5,
                      pad_override: str = "") -> None:
        arm, pad = _PUNCH_TYPES.get(punch_type, ("left", "centre"))
        if pad_override:
            pad = pad_override
        msg = ConfirmedPunch()
        msg.timestamp = time.time()
        msg.punch_type = punch_type
        msg.pad = pad
        msg.level = level
        msg.force_normalized = force_normalized
        msg.cv_confidence = 0.95
        msg.imu_confirmed = True
        msg.cv_confirmed = True
        msg.accel_magnitude = force_normalized * 60.0  # approximate m/s²
        self._pub_punch.publish(msg)
        # Also publish the pad impact and arm strike
        self.publish_pad(pad, level, accel=force_normalized * 60.0)
        self.publish_arm(arm, True)
        self.get_logger().info(
            f"Punch {punch_type} level={level} force={force_normalized:.2f}"
        )

    def publish_arm(self, arm: str, contact: bool) -> None:
        msg = ArmStrike()
        msg.timestamp = time.time()
        msg.arm = arm
        msg.contact = contact
        self._pub_arm.publish(msg)
        self.get_logger().info(f"ArmStrike arm={arm} contact={contact}")

    def _publish_status(self) -> None:
        msg = IMUStatus()
        msg.left_pad_connected = True
        msg.centre_pad_connected = True
        msg.right_pad_connected = True
        msg.head_pad_connected = True
        msg.left_arm_connected = True
        msg.right_arm_connected = True
        # Mark as simulator, but indicate if real hardware is also connected
        msg.is_simulator = not self._teensy_connected
        self._pub_status.publish(msg)


class IMUSimulatorGUI:
    """Tkinter front-end that drives IMUSimulatorNode."""

    def __init__(self, node: IMUSimulatorNode) -> None:
        self._node = node
        self._force = "medium"
        self._log_lines: deque[str] = deque(maxlen=80)
        self._pad_btns: dict[str, tk.Button] = {}
        self._arm_btns: dict[str, tk.Button] = {}
        self._force_btns: dict[str, tk.Button] = {}
        self._punch_btns: dict[str, tk.Button] = {}
        self._sequence: list[dict] = []
        self._seq_playing = False

        self._root = tk.Tk()
        self._root.title("BoxBunny IMU Simulator")
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._build()

        # Wire up incoming robot commands to flash punch buttons
        node._robot_cmd_callback = self._on_incoming_punch
        # Wire up real hardware mirror callbacks
        node._real_strike_callback = self._on_real_strike_gui
        node._motor_feedback_callback = self._on_motor_feedback_gui

        # IMU pad index -> pad name (matches boxbunny.yaml imu_pad_map)
        self._imu_pad_map = {0: "centre", 1: "left", 2: "right", 3: "head"}

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self) -> None:
        r = self._root

        # ── Title bar ───────────────────────────────────────────────────
        top = tk.Frame(r, bg=SURFACE, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="Box", font=(FONT, 14, "bold"),
                 bg=SURFACE, fg=FG).pack(side="left", padx=(16, 0))
        tk.Label(top, text="Bunny", font=(FONT, 14, "bold"),
                 bg=SURFACE, fg=PRIMARY).pack(side="left")
        tk.Label(top, text="IMU Simulator", font=(FONT, 9),
                 bg=SURFACE, fg=FG_DIM).pack(side="left", padx=8)
        self._hw_indicator = tk.Label(
            top, text="HW: --", font=(FONT, 8, "bold"),
            bg=SURFACE, fg=FG_MUTED,
        )
        self._hw_indicator.pack(side="right", padx=12)

        # ── Force section ───────────────────────────────────────────────
        force_frame = tk.Frame(r, bg=BG)
        force_frame.pack(fill="x", padx=16, pady=(10, 0))

        tk.Label(force_frame, text="FORCE", font=(FONT, 8, "bold"),
                 bg=BG, fg=FG_MUTED).pack(side="left", padx=(0, 8))

        self._accel_var = tk.DoubleVar(value=30.0)
        for val, lbl, color in [
            (10, "Light", GREEN), (30, "Medium", AMBER), (50, "Hard", RED),
        ]:
            tk.Button(
                force_frame, text=lbl, font=(FONT, 9, "bold"),
                bg=SURFACE2, fg=color,
                activebackground=color, activeforeground="#000",
                relief="flat", bd=0, pady=6, padx=14,
                command=lambda v=val: self._accel_var.set(v),
            ).pack(side="left", padx=2)

        self._accel_lbl = tk.Label(
            force_frame, text="30 m/s\u00B2", font=(FONT, 10, "bold"),
            bg=BG, fg=AMBER, anchor="e",
        )
        self._accel_lbl.pack(side="right")
        self._accel_var.trace_add("write", self._update_accel_label)

        # Slider
        slider_frame = tk.Frame(r, bg=BG)
        slider_frame.pack(fill="x", padx=16, pady=(4, 0))
        self._accel_slider = tk.Scale(
            slider_frame, from_=0, to=60, orient="horizontal",
            variable=self._accel_var, resolution=0.5,
            bg=BG, fg=FG, troughcolor=SURFACE2,
            highlightthickness=0, font=(FONT, 8),
            showvalue=False,
        )
        self._accel_slider.pack(fill="x")

        # ── Pad & Arm section ───────────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(r, text="USER STRIKE DETECTION", font=(FONT, 8, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(8, 0))

        body = tk.Frame(r, bg=BG)
        body.pack(padx=16, pady=(6, 0))

        self._arm_btns["left"] = self._make_arm(body, "L", 0, 0)
        pf = tk.Frame(body, bg=BG)
        pf.grid(row=0, column=1, padx=6)
        self._pad_btns["head"] = self._make_pad(pf, "HEAD", 0, 1)
        self._pad_btns["left"] = self._make_pad(pf, "LEFT", 1, 0)
        self._pad_btns["centre"] = self._make_pad(pf, "CENTRE", 1, 1)
        self._pad_btns["right"] = self._make_pad(pf, "RIGHT", 1, 2)
        self._arm_btns["right"] = self._make_arm(body, "R", 0, 2)

        # ── Punch simulator ─────────────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(r, text="ROBOT ARM PUNCHES", font=(FONT, 8, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(8, 0))

        punch_row = tk.Frame(r, bg=BG)
        punch_row.pack(fill="x", padx=16, pady=(4, 0))

        punch_defs = [
            ("Jab",     "jab",     PUNCH_JAB),
            ("Cross",   "cross",   PUNCH_CROSS),
            ("L Hook",  "l_hook",  PUNCH_HOOK),
            ("R Hook",  "r_hook",  PUNCH_HOOK),
            ("L Upper", "l_upper", PUNCH_UPPER),
            ("R Upper", "r_upper", PUNCH_UPPER),
        ]
        self._punch_btns: dict[str, tk.Button] = {}
        for label, ptype, color in punch_defs:
            btn = tk.Button(
                punch_row, text=label, font=(FONT, 9, "bold"),
                bg=SURFACE2, fg=color,
                activebackground=color, activeforeground="#000",
                relief="flat", bd=0, pady=8,
                command=lambda pt=ptype, c=color: self._on_punch(pt, c),
            )
            btn.pack(side="left", expand=True, fill="x", padx=1)
            self._punch_btns[ptype] = btn

        # Combo presets
        combo_row = tk.Frame(r, bg=BG)
        combo_row.pack(fill="x", padx=16, pady=(4, 0))

        combos = [
            ("1-2",     ["jab", "cross"]),
            ("1-1-2",   ["jab", "jab", "cross"]),
            ("1-2-3",   ["jab", "cross", "l_hook"]),
            ("1-2-3-4", ["jab", "cross", "l_hook", "r_hook"]),
            ("1-2-5-6", ["jab", "cross", "l_upper", "r_upper"]),
        ]
        for label, seq in combos:
            tk.Button(
                combo_row, text=label, font=(FONT, 9, "bold"),
                bg=SURFACE, fg=PRIMARY,
                activebackground=PRIMARY, activeforeground="#000",
                relief="flat", bd=0, pady=6, padx=8,
                command=lambda s=seq: self._play_combo(s),
            ).pack(side="left", expand=True, fill="x", padx=1)

        # ── Sequence builder (compact) ──────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(r, text="CUSTOM SEQUENCE", font=(FONT, 8, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(8, 0))

        seq_ctrl = tk.Frame(r, bg=BG)
        seq_ctrl.pack(fill="x", padx=16, pady=(4, 0))

        for pad in ["HEAD", "LEFT", "CENTRE", "RIGHT"]:
            tk.Button(
                seq_ctrl, text=f"+{pad}", font=(FONT, 7),
                bg=SURFACE2, fg=FG_DIM,
                activebackground=SURFACE3, activeforeground=FG,
                relief="flat", bd=0, padx=4, pady=4,
                command=lambda p=pad.lower(): self._seq_add("pad", p),
            ).pack(side="left", padx=1, expand=True, fill="x")

        self._interval_var = tk.StringVar(value="500")
        self._play_btn = tk.Button(
            seq_ctrl, text="PLAY", font=(FONT, 8, "bold"),
            bg=GREEN, fg="#000", activebackground="#3FB950",
            relief="flat", bd=0, padx=10, pady=4,
            command=self._seq_play,
        )
        self._play_btn.pack(side="left", padx=(6, 1))

        tk.Button(
            seq_ctrl, text="CLR", font=(FONT, 8, "bold"),
            bg=SURFACE2, fg=FG_DIM, activebackground=BORDER_LT,
            relief="flat", bd=0, padx=8, pady=4,
            command=self._seq_clear,
        ).pack(side="left", padx=1)

        self._seq_frame = tk.Frame(r, bg=SEQ_BG, height=28)
        self._seq_frame.pack(fill="x", padx=16, pady=(3, 0))
        self._seq_label = tk.Label(
            self._seq_frame, text="(empty)",
            font=(FONT_M, 8), bg=SEQ_BG, fg=FG_MUTED, anchor="w",
        )
        self._seq_label.pack(fill="x", padx=8, pady=4)

        # ── Teensy live data ────────────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(8, 0))
        tk.Label(r, text="TEENSY LIVE DATA", font=(FONT, 8, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(6, 0))

        hw_frame = tk.Frame(r, bg=SURFACE, highlightthickness=0)
        hw_frame.pack(fill="x", padx=16, pady=(2, 0))

        # Motor positions row
        motor_row = tk.Frame(hw_frame, bg=SURFACE)
        motor_row.pack(fill="x", padx=8, pady=(6, 0))
        _motor_names = ["L1", "L2", "R1", "R2"]
        tk.Label(motor_row, text="Motors", font=(FONT, 9, "bold"),
                 bg=SURFACE, fg=FG_DIM).pack(side="left", padx=(0, 6))
        self._hw_pos_labels = []
        self._hw_cur_labels = []
        for i in range(4):
            cell = tk.Frame(motor_row, bg=SURFACE)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=_motor_names[i], font=(FONT, 8),
                     bg=SURFACE, fg=FG_MUTED).pack()
            plbl = tk.Label(cell, text="--", font=(FONT, 10),
                            bg=SURFACE, fg=FG)
            plbl.pack()
            clbl = tk.Label(cell, text="--", font=(FONT, 9),
                            bg=SURFACE, fg=FG_DIM)
            clbl.pack()
            self._hw_pos_labels.append(plbl)
            self._hw_cur_labels.append(clbl)

        # IMU accel magnitudes row (per pad) + threshold indicator
        imu_header = tk.Frame(hw_frame, bg=SURFACE)
        imu_header.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(imu_header, text="IMU", font=(FONT, 9, "bold"),
                 bg=SURFACE, fg=FG_DIM).pack(side="left")
        # Show the strike threshold (matches V4 IMU Diagnostics: 20 m/s²)
        self._strike_threshold = 20.0
        tk.Label(imu_header, text=f"strike threshold: {self._strike_threshold:.0f} m/s\u00B2",
                 font=(FONT, 8), bg=SURFACE, fg=FG_MUTED).pack(side="left", padx=(8, 0))
        tk.Label(imu_header, text="(gravity-subtracted)",
                 font=(FONT, 7), bg=SURFACE, fg=FG_MUTED).pack(side="left", padx=(4, 0))

        imu_row = tk.Frame(hw_frame, bg=SURFACE)
        imu_row.pack(fill="x", padx=8, pady=(2, 6))
        _imu_pad_names = ["Centre", "Left", "Right", "Head"]
        _imu_pad_colors = [PRIMARY, PUNCH_JAB, PUNCH_UPPER, AMBER]
        self._hw_imu_labels = []
        self._hw_imu_peak_labels = []
        for i in range(4):
            cell = tk.Frame(imu_row, bg=SURFACE)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=_imu_pad_names[i], font=(FONT, 8),
                     bg=SURFACE, fg=_imu_pad_colors[i]).pack()
            # Raw magnitude (from motor_feedback, includes gravity)
            lbl = tk.Label(cell, text="--", font=(FONT, 11, "bold"),
                           bg=SURFACE, fg=FG_DIM)
            lbl.pack()
            self._hw_imu_labels.append(lbl)
            # Last detected peak (from /robot/strike_detected, gravity-subtracted)
            peak_lbl = tk.Label(cell, text="", font=(FONT, 8),
                                bg=SURFACE, fg=FG_MUTED)
            peak_lbl.pack()
            self._hw_imu_peak_labels.append(peak_lbl)

        # ── Event log ───────────────────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(8, 0))
        tk.Label(r, text="EVENT LOG", font=(FONT, 8, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(6, 0))

        self._log_text = tk.Text(
            r, height=5, width=52, bg=SURFACE,
            fg=FG_DIM, font=(FONT_M, 8),
            state="disabled", wrap="word",
            borderwidth=0, highlightthickness=0,
        )
        self._log_text.pack(padx=16, pady=(2, 12), fill="both", expand=True)

    # ── Incoming robot commands (from GUI drill cycling) ────────────────
    def _on_incoming_punch(self, punch_type: str) -> None:
        """Called from the ROS thread — schedule Tkinter update."""
        self._root.after(0, lambda: self._flash_incoming(punch_type))

    def _flash_incoming(self, punch_type: str) -> None:
        """Flash the punch button and arm to show an incoming robot command."""
        color_map = {
            "jab": PUNCH_JAB, "cross": PUNCH_CROSS,
            "l_hook": PUNCH_HOOK, "r_hook": PUNCH_HOOK,
            "l_upper": PUNCH_UPPER, "r_upper": PUNCH_UPPER,
        }
        color = color_map.get(punch_type, PRIMARY)
        arm_side, _ = _PUNCH_TYPES.get(punch_type, ("left", "centre"))

        if punch_type in self._punch_btns:
            btn = self._punch_btns[punch_type]
            btn.configure(bg=color, fg="#000")
            btn.after(_FLASH_MS, lambda: btn.configure(bg=SURFACE2, fg=color))

        if arm_side in self._arm_btns:
            self._flash(self._arm_btns[arm_side], color, ARM_BG, PUNCH_JAB)

        _PUNCH_LABELS = {
            "jab": "Jab", "cross": "Cross", "l_hook": "L Hook",
            "r_hook": "R Hook", "l_upper": "L Upper", "r_upper": "R Upper",
        }
        self._log(f"CMD>  {_PUNCH_LABELS.get(punch_type, punch_type):<8s}  "
                  f"(from GUI drill)")

    # ── Real hardware mirror handlers ──────────────────────────────────
    def _on_real_strike_gui(self, data: dict) -> None:
        """Called from ROS thread when real Teensy detects a pad strike."""
        self._root.after(0, lambda: self._flash_real_strike(data))

    def _flash_real_strike(self, data: dict) -> None:
        """Flash the pad on the simulator matching a real hardware strike.

        The /robot/strike_detected message is published by the V4 arm control
        GUI after gravity calibration + peak scanning.  The peak_accel value
        is already gravity-subtracted, so the threshold comparison matches
        the V4 IMUDiagnosticsTab: check_val > _strike_cutoff (20 m/s²).
        """
        pad_index = data.get("pad_index", -1)
        peak_accel = data.get("peak_accel", 0.0)
        pad_name = self._imu_pad_map.get(pad_index)
        if pad_name is None:
            return

        # Force level classification (same thresholds as imu_node.py)
        if peak_accel >= 40:
            color = RED
            level = "hard"
        elif peak_accel >= 20:
            color = AMBER
            level = "medium"
        else:
            color = GREEN
            level = "light"

        # Flash the pad button
        if pad_name in self._pad_btns:
            self._flash(self._pad_btns[pad_name], color, PAD_BG, RED)

        # Update the peak label under the IMU value for this pad
        if 0 <= pad_index < len(self._hw_imu_peak_labels):
            self._hw_imu_peak_labels[pad_index].configure(
                text=f"pk: {peak_accel:.1f}", fg=color,
            )
            # Clear the peak label after 1.5s
            self._hw_imu_peak_labels[pad_index].after(
                1500,
                lambda idx=pad_index: self._hw_imu_peak_labels[idx].configure(
                    text="", fg=FG_MUTED,
                ),
            )

        self._log(
            f"HW>  {pad_name:<7s}  peak={peak_accel:.1f}m/s\u00B2  level={level}"
            f"  (threshold={self._strike_threshold:.0f})"
        )

    def _on_motor_feedback_gui(self, data) -> None:
        """Called from ROS thread with motor_feedback data."""
        # Schedule GUI update on main thread with a copy of the data
        self._root.after(0, lambda d=list(data): self._update_hw_display(d))

    def _update_hw_display(self, data: list) -> None:
        """Update the Teensy live data display (positions, currents, raw IMU).

        Strike detection is handled by _flash_real_strike() which is triggered
        by /robot/strike_detected — the V4 arm control GUI does the gravity
        calibration, peak scanning, and threshold check on its side, then
        publishes the final result.  We just display raw values here.
        """
        import math

        # Connection indicator
        if self._node._teensy_connected:
            self._hw_indicator.configure(text="HW: LIVE", fg=GREEN)
        else:
            self._hw_indicator.configure(text="HW: --", fg=FG_MUTED)
            return

        # Motor positions [0:4]
        if len(data) >= 4:
            for i in range(4):
                self._hw_pos_labels[i].configure(
                    text=f"{data[i]:+.2f}", fg=FG,
                )

        # Motor currents [4:8]
        if len(data) >= 8:
            for i in range(4):
                val = abs(data[4 + i])
                color = RED if val > 2.0 else (AMBER if val > 0.5 else FG_DIM)
                self._hw_cur_labels[i].configure(
                    text=f"{data[4+i]:+.2f}A", fg=color,
                )

        # IMU raw magnitudes (for reference — strike detection comes from
        # /robot/strike_detected which is published by the V4 arm control GUI
        # after gravity calibration + peak scanning + threshold check)
        if len(data) >= 21:
            for i in range(4):
                ax = data[9 + i * 3]
                ay = data[9 + i * 3 + 1]
                az = data[9 + i * 3 + 2]
                raw_mag = math.sqrt(ax * ax + ay * ay + az * az)
                # Just display; color is informational only
                if raw_mag > 20:
                    color = AMBER
                elif raw_mag > 12:
                    color = FG
                else:
                    color = FG_DIM
                self._hw_imu_labels[i].configure(text=f"{raw_mag:.1f}", fg=color)

    # ── Factories ───────────────────────────────────────────────────────
    def _make_pad(self, parent: tk.Frame, label: str,
                  row: int, col: int) -> tk.Button:
        btn = tk.Button(
            parent, text=label, width=8, height=3,
            font=(FONT, 10, "bold"),
            bg=PAD_BG, fg=RED,
            activebackground=PAD_FLASH, activeforeground="#FFF",
            relief="flat", bd=0,
            command=lambda: self._on_pad(label.lower()),
        )
        btn.grid(row=row, column=col, padx=2, pady=2)
        return btn

    def _make_arm(self, parent: tk.Frame, label: str,
                  row: int, col: int) -> tk.Button:
        btn = tk.Button(
            parent, text=label, width=4, height=10,
            font=(FONT, 14, "bold"),
            bg=ARM_BG, fg=PUNCH_JAB,
            activebackground=ARM_FLASH, activeforeground="#FFF",
            relief="flat", bd=0,
        )
        side = "left" if label == "L" else "right"
        btn.bind("<Button-1>", lambda e, s=side: self._on_arm(e, s))
        btn.grid(row=row, column=col, padx=3, pady=2, sticky="ns")
        return btn

    # ── Acceleration helpers ────────────────────────────────────────────
    def _update_accel_label(self, *_args) -> None:
        val = self._accel_var.get()
        if val < 20:
            color = GREEN
        elif val < 40:
            color = AMBER
        else:
            color = RED
        self._accel_lbl.configure(text=f"{val:.0f} m/s\u00B2", fg=color)

    def _get_force_level(self) -> str:
        val = self._accel_var.get()
        if val < 20:
            return "light"
        if val < 40:
            return "medium"
        return "hard"

    def _get_force_normalized(self) -> float:
        return min(1.0, self._accel_var.get() / 60.0)

    # ── Flash ───────────────────────────────────────────────────────────
    def _flash(self, btn: tk.Button, color: str, rest_bg: str,
               rest_fg: str) -> None:
        btn.configure(bg=color, fg="#000")
        btn.after(_FLASH_MS, lambda: btn.configure(bg=rest_bg, fg=rest_fg))

    # ── Pad / Arm handlers ──────────────────────────────────────────────
    def _on_pad(self, pad: str) -> None:
        level = self._get_force_level()
        force = self._get_force_normalized()
        accel = self._accel_var.get()
        self._node.publish_pad(pad, level, accel=accel)
        self._node.publish_punch("strike", level, force, pad_override=pad)
        colors = {"light": GREEN, "medium": AMBER, "hard": RED}
        if pad in self._pad_btns:
            self._flash(self._pad_btns[pad], colors[level],
                        PAD_BG, RED)
        self._log(f"PAD  {pad:<7s}  accel={self._accel_var.get():.1f}  level={level}")

    def _on_arm(self, event: tk.Event, side: str) -> None:
        contact = not bool(event.state & 0x0001)
        self._node.publish_arm(side, contact)
        color = PUNCH_JAB if contact else FG_MUTED
        if side in self._arm_btns:
            self._flash(self._arm_btns[side], color, ARM_BG, PUNCH_JAB)
        tag = "struck" if contact else "miss"
        self._log(f"ARM  {side:<7s}  {tag}")

    # ── Punch simulator ─────────────────────────────────────────────────
    def _on_punch(self, punch_type: str, color: str) -> None:
        """Simulate robot arm throwing a punch — publishes RobotCommand to
        make physical arms move, plus ConfirmedPunch for data tracking."""
        level = self._get_force_level()
        force = self._get_force_normalized()
        self._node.publish_punch(punch_type, level, force)
        arm_side, _ = _PUNCH_TYPES.get(punch_type, ("left", "centre"))

        # Also publish RobotCommand so physical robot arms execute the punch
        _PUNCH_TO_CODE = {
            "jab": "1", "cross": "2", "l_hook": "3",
            "r_hook": "4", "l_upper": "5", "r_upper": "6",
        }
        code = _PUNCH_TO_CODE.get(punch_type)
        if code:
            cmd = RobotCommand()
            cmd.command_type = "punch"
            cmd.punch_code = code
            cmd.speed = "medium"
            self._node._pub_robot_cmd.publish(cmd)

        # Flash the punch button
        if punch_type in self._punch_btns:
            btn = self._punch_btns[punch_type]
            btn.configure(bg=color, fg="#000")
            btn.after(_FLASH_MS, lambda: btn.configure(bg=SURFACE2, fg=color))

        # Flash the arm (robot arm moving)
        if arm_side in self._arm_btns:
            self._flash(self._arm_btns[arm_side], color, ARM_BG, PUNCH_JAB)

        _PUNCH_NAMES = {
            "jab": "Jab", "cross": "Cross", "l_hook": "L Hook",
            "r_hook": "R Hook", "l_upper": "L Upper", "r_upper": "R Upper",
        }
        self._log(f"ARM>  {_PUNCH_NAMES.get(punch_type, punch_type):<8s}  "
                  f"arm={arm_side}  accel={self._accel_var.get():.1f}")

    def _play_combo(self, sequence: list) -> None:
        """Play a combo sequence with animated delays."""
        interval = int(self._interval_var.get())

        def play_step(idx: int) -> None:
            if idx >= len(sequence):
                return
            ptype = sequence[idx]
            color_map = {
                "jab": PUNCH_JAB, "cross": PUNCH_CROSS,
                "l_hook": PUNCH_HOOK, "r_hook": PUNCH_HOOK,
                "l_upper": PUNCH_UPPER, "r_upper": PUNCH_UPPER,
            }
            self._on_punch(ptype, color_map.get(ptype, PRIMARY))
            self._root.after(interval, lambda: play_step(idx + 1))

        play_step(0)

    # ── Sequence builder ────────────────────────────────────────────────
    def _seq_add(self, kind: str, target: str) -> None:
        self._sequence.append({"kind": kind, "target": target})
        self._seq_update_label()

    def _seq_clear(self) -> None:
        self._sequence.clear()
        self._seq_update_label()

    def _seq_update_label(self) -> None:
        if not self._sequence:
            self._seq_label.configure(
                text="(empty — click +PAD buttons above to build)",
                fg=FG_MUTED,
            )
            return
        parts = []
        for item in self._sequence:
            if item["kind"] == "pad":
                parts.append(item["target"].upper())
            else:
                parts.append(f"{'L' if item['target'] == 'left' else 'R'}-ARM")
        self._seq_label.configure(
            text=" → ".join(parts),
            fg=FG,
        )

    def _seq_play(self) -> None:
        if not self._sequence or self._seq_playing:
            return
        self._seq_playing = True
        self._play_btn.configure(bg=FG_MUTED, text="...")
        interval = int(self._interval_var.get())

        def play_step(idx: int) -> None:
            if idx >= len(self._sequence):
                self._seq_playing = False
                self._play_btn.configure(bg=GREEN, text="PLAY")
                return
            item = self._sequence[idx]
            if item["kind"] == "pad":
                self._on_pad(item["target"])
            else:
                self._node.publish_arm(item["target"], True)
                side = item["target"]
                if side in self._arm_btns:
                    self._flash(self._arm_btns[side], "#3B82F6",
                                ARM_BG, "#6EA8DC")
                self._log(f"ARM  {side:<7s}  struck")
            self._root.after(interval, lambda: play_step(idx + 1))

        play_step(0)

    # ── Log ─────────────────────────────────────────────────────────────
    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self._log_lines.append(line)
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ── Run ─────────────────────────────────────────────────────────────
    def spin(self) -> None:
        t = threading.Thread(target=rclpy.spin, args=(self._node,),
                             daemon=True)
        t.start()
        try:
            self._root.mainloop()
        finally:
            self._node.destroy_node()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    rclpy.init()
    try:
        node = IMUSimulatorNode()
        gui = IMUSimulatorGUI(node)
        gui.spin()
    except KeyboardInterrupt:
        log.info("Shutting down IMU simulator")
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
