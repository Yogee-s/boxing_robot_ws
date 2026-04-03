#!/usr/bin/env python3
"""BoxBunny Teensy Simulator -- standalone Tkinter GUI that publishes the same
ROS 2 messages as the real Teensy hardware so drills can be developed and
tested without physical sensors.

Launch:  ros2 run boxbunny_core teensy_simulator   (or just: python3 tools/teensy_simulator.py)
"""

from __future__ import annotations

import json
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
        ArmStrike, ConfirmedPunch, HeightCommand, IMUStatus, PadImpact,
        RobotCommand,
    )
except ImportError:
    raise SystemExit(
        "boxbunny_msgs not found. Build the workspace first:\n"
        "  cd boxing_robot_ws && colcon build --packages-select boxbunny_msgs"
    )

from std_msgs.msg import String as StdString, Float64MultiArray

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

log = logging.getLogger("teensy_simulator")


class TeensySimulatorNode(Node):
    """ROS 2 node bridging the Teensy simulator to the V4 Arm Control GUI.

    Publishes:
      - PadImpact, ArmStrike, ConfirmedPunch (BoxBunny IMU topics)
      - /robot/strike_command   (tells V4 GUI to execute a strike)
      - /robot/punch_slots      (assigns punch types to V4 GUI slots)
      - /robot/system_enable    (arms/disarms V4 GUI motor control)

    Subscribes:
      - /boxbunny/robot/command (incoming drill commands from BoxBunny GUI)
      - /robot/strike_feedback  (V4 GUI reports strike completion)
      - /robot/strike_detected  (V4 GUI reports pad IMU strike)
      - motor_feedback          (Teensy raw positions/currents/IMU)
    """

    def __init__(self) -> None:
        super().__init__("teensy_simulator")

        # ── BoxBunny IMU topic publishers ────────────────────────────────
        self._pub_pad = self.create_publisher(PadImpact, _TOPIC_PAD, 10)
        self._pub_arm = self.create_publisher(ArmStrike, _TOPIC_ARM, 10)
        self._pub_status = self.create_publisher(IMUStatus, _TOPIC_STATUS, 10)
        self._pub_punch = self.create_publisher(ConfirmedPunch, _TOPIC_PUNCH, 10)
        self.create_timer(1.0, self._publish_status)

        # ── V4 GUI command publishers ────────────────────────────────────
        # These topics are what the V4 Arm Control GUI listens to
        self._pub_strike_cmd = self.create_publisher(
            StdString, "/robot/strike_command", 10)
        self._pub_punch_slots = self.create_publisher(
            StdString, "/robot/punch_slots", 10)
        self._pub_system_enable = self.create_publisher(
            StdString, "/robot/system_enable", 10)

        # ── Simulated strike feedback publishers ─────────────────────────
        self._pub_strike_feedback_sim = self.create_publisher(
            StdString, "/robot/strike_feedback", 10)
        self._pub_strike_complete = self.create_publisher(
            StdString, "/boxbunny/robot/strike_complete", 10)

        # ── Subscribe to BoxBunny robot commands (from drill cycling) ────
        self.create_subscription(
            RobotCommand, _TOPIC_ROBOT_CMD, self._on_robot_command, 10,
        )
        self._robot_cmd_callback = None  # set by GUI
        self._execute_cmd_callback = None  # set by GUI for auto/manual exec

        # ── Height command subscription ──────────────────────────────────
        self._height_action = "stop"
        self._height_callback = None  # set by GUI
        self.create_subscription(
            HeightCommand, "/boxbunny/robot/height",
            self._on_height_command, 10,
        )

        # ── Person tracking direction subscription ───────────────────────
        self._person_direction = "offline"
        self._last_direction_time = 0.0
        self._direction_callback = None  # set by GUI
        self.create_subscription(
            StdString, "/boxbunny/cv/person_direction",
            self._on_person_direction, 10,
        )

        # ── V4 GUI feedback subscriptions ────────────────────────────────
        # Strike completion from V4 GUI (after FSM execution)
        self.create_subscription(
            StdString, "/robot/strike_feedback",
            self._on_strike_feedback, 10,
        )
        self._strike_feedback_callback = None  # set by GUI

        # Pad IMU strike detection from V4 GUI (gravity-calibrated)
        self.create_subscription(
            StdString, "/robot/strike_detected",
            self._on_real_strike, 10,
        )
        self._real_strike_callback = None   # set by GUI

        # Motor feedback from Teensy (positions, currents, raw IMU)
        self.create_subscription(
            Float64MultiArray, "motor_feedback",
            self._on_motor_feedback, 10,
        )
        self._motor_feedback_callback = None  # set by GUI
        self._real_imu_accel = [[0.0, 0.0, 0.0] for _ in range(4)]
        self._teensy_connected = False

        # ── Punch code -> slot mapping ───────────────────────────────────
        # Maps BoxBunny punch codes ("1"-"6") to V4 GUI slot numbers (1-6)
        # The V4 GUI needs punch_slots assigned before strikes can execute
        self._code_to_slot = {
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        }

        self.get_logger().info("Teensy simulator node started")

    # ── V4 GUI arm control ───────────────────────────────────────────

    def send_punch_slots(self, slots: dict) -> None:
        """Assign punch types to V4 GUI slots.

        Args:
            slots: {1: {"arm": "left", "strike": "Jab"}, ...}
        """
        msg = StdString()
        msg.data = json.dumps({str(k): v for k, v in slots.items()})
        self._pub_punch_slots.publish(msg)
        self.get_logger().info(f"Punch slots assigned: {slots}")

    def send_system_enable(self, enable: bool) -> None:
        """Arm or disarm the V4 GUI motor control."""
        msg = StdString()
        msg.data = "enable" if enable else "disable"
        self._pub_system_enable.publish(msg)
        self.get_logger().info(f"System {'enabled' if enable else 'disabled'}")

    def send_strike_command(self, slot: int, duration: float = 5.0,
                            speed: float = None) -> None:
        """Tell the V4 GUI to execute a strike at the given slot.

        Speed is capped at 30 rad/s for gear safety.
        """
        from std_msgs.msg import String as StdString
        cmd = {"slot": slot, "duration": duration}
        if speed is not None:
            cmd["speed"] = min(float(speed), 30.0)
        msg = StdString()
        msg.data = json.dumps(cmd)
        self._pub_strike_cmd.publish(msg)
        spd_str = f", speed={cmd.get('speed', 'default')}" if speed is not None else ""
        self.get_logger().info(f"Strike command: slot={slot}, duration={duration}s{spd_str}")

    def _on_strike_feedback(self, msg) -> None:
        """Handle strike completion feedback from V4 GUI."""
        try:
            data = json.loads(msg.data)
            if self._strike_feedback_callback:
                self._strike_feedback_callback(data)
        except Exception:
            pass

    def _on_real_strike(self, msg) -> None:
        """Handle strike detection from real Teensy hardware."""
        try:
            data = json.loads(msg.data)
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

    def _on_height_command(self, msg: HeightCommand) -> None:
        """Handle height command from BoxBunny system."""
        self._height_action = msg.action
        if self._height_callback:
            self._height_callback(msg.action)

    def _on_person_direction(self, msg: StdString) -> None:
        """Handle person tracking direction updates."""
        self._person_direction = msg.data
        self._last_direction_time = time.time()
        if self._direction_callback:
            self._direction_callback(msg.data)

    def simulate_strike_complete(self, slot: int, strike_name: str,
                                 delay_s: float) -> None:
        """Publish simulated strike feedback as if V4 GUI completed it."""
        feedback = json.dumps({
            "slot": slot, "strike": strike_name, "status": "completed",
            "duration_allowed": 5.0, "duration_actual": round(delay_s, 2),
        })
        msg = StdString()
        msg.data = feedback
        self._pub_strike_feedback_sim.publish(msg)
        complete = json.dumps({
            "punch_code": str(slot), "status": "completed",
            "duration_ms": int(delay_s * 1000), "strike": strike_name,
        })
        msg2 = StdString()
        msg2.data = complete
        self._pub_strike_complete.publish(msg2)

    def _on_robot_command(self, msg: RobotCommand) -> None:
        """Translate BoxBunny RobotCommand → execute callback for GUI.

        When the BoxBunny GUI or sparring engine publishes a punch command,
        flash the punch button and pass the command info to the GUI, which
        decides whether to auto-execute, wait for manual execution, or
        forward to hardware.
        """
        if msg.command_type != "punch":
            return
        # Flash on simulator
        punch_type = _CODE_TO_PUNCH.get(msg.punch_code, "")
        arm_side = _PUNCH_TYPES.get(punch_type, ("left", "centre"))[0]
        if punch_type and self._robot_cmd_callback:
            self._robot_cmd_callback(punch_type)
        # Map speed string to rad/s
        _SPEED_MAP = {"slow": 8.0, "medium": 15.0, "fast": 25.0}
        speed = _SPEED_MAP.get(msg.speed)
        # Build command info for GUI execution handling
        slot = self._code_to_slot.get(msg.punch_code)
        self._pending_cmd = {
            "slot": slot, "punch_type": punch_type,
            "arm": arm_side, "speed": speed, "code": msg.punch_code,
        }
        if self._execute_cmd_callback:
            self._execute_cmd_callback(self._pending_cmd)

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


class TeensySimulatorGUI:
    """Tkinter front-end that drives TeensySimulatorNode."""

    def __init__(self, node: TeensySimulatorNode) -> None:
        self._node = node
        self._force = "medium"
        self._log_lines: deque[str] = deque(maxlen=80)
        self._pad_btns: dict[str, tk.Button] = {}
        self._arm_btns: dict[str, tk.Button] = {}
        self._force_btns: dict[str, tk.Button] = {}
        self._punch_btns: dict[str, tk.Button] = {}
        self._sequence: list[dict] = []
        self._seq_playing = False
        self._combo_queue: list[str] = []
        self._combo_playing = False
        self._combo_timeout_id = None
        self._pending_cmd = None  # dict or None
        self._executing = False

        self._root = tk.Tk()
        self._root.title("BoxBunny Teensy Simulator")
        self._root.configure(bg=BG)
        self._root.geometry("600x920")
        self._root.resizable(True, True)
        self._auto_execute = tk.BooleanVar(value=True)
        self._fwd_hardware = tk.BooleanVar(value=False)
        self._build()

        # Wire up incoming robot commands to flash punch buttons
        node._robot_cmd_callback = self._on_incoming_punch
        # Wire up real hardware mirror callbacks
        node._real_strike_callback = self._on_real_strike_gui
        node._motor_feedback_callback = self._on_motor_feedback_gui
        node._strike_feedback_callback = self._on_strike_feedback_gui
        # Wire up execute/height/direction callbacks
        node._execute_cmd_callback = self._on_execute_cmd
        node._height_callback = lambda action: self._root.after(
            0, lambda: self._update_height(action))
        node._direction_callback = lambda direction: self._root.after(
            0, lambda: self._update_direction(direction))
        # Periodic tracking offline check
        self._root.after(2000, self._check_tracking_offline)

        # IMU pad index -> pad name (matches boxbunny.yaml imu_pad_map)
        # Teensy IMU indices → user-perspective pad names
        # Physical wiring: Teensy index 1 = user's RIGHT, index 2 = user's LEFT
        self._imu_pad_map = {0: "centre", 1: "right", 2: "left", 3: "head"}

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self) -> None:
        r = self._root

        # ── Title bar ───────────────────────────────────────────────────
        top = tk.Frame(r, bg=SURFACE, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="Box", font=(FONT, 16, "bold"),
                 bg=SURFACE, fg=FG).pack(side="left", padx=(16, 0))
        tk.Label(top, text="Bunny", font=(FONT, 16, "bold"),
                 bg=SURFACE, fg=PRIMARY).pack(side="left")
        tk.Label(top, text="Teensy Simulator", font=(FONT, 11),
                 bg=SURFACE, fg=FG_DIM).pack(side="left", padx=8)
        self._hw_indicator = tk.Label(
            top, text="HW: --", font=(FONT, 10, "bold"),
            bg=SURFACE, fg=FG_MUTED,
        )
        self._hw_indicator.pack(side="right", padx=12)

        # ── Force section ───────────────────────────────────────────────
        force_frame = tk.Frame(r, bg=BG)
        force_frame.pack(fill="x", padx=16, pady=(10, 0))

        tk.Label(force_frame, text="FORCE", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(side="left", padx=(0, 8))

        self._accel_var = tk.DoubleVar(value=30.0)
        for val, lbl, color in [
            (10, "Light", GREEN), (30, "Medium", AMBER), (50, "Hard", RED),
        ]:
            tk.Button(
                force_frame, text=lbl, font=(FONT, 11, "bold"),
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

        # ── Strike speed section ────────────────────────────────────────
        spd_frame = tk.Frame(r, bg=BG)
        spd_frame.pack(fill="x", padx=16, pady=(10, 0))

        tk.Label(spd_frame, text="STRIKE SPEED", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(side="left", padx=(0, 8))

        self._speed_var = tk.DoubleVar(value=20.0)
        for val, lbl in [(5, "Slow"), (15, "Med"), (25, "Fast")]:
            tk.Button(
                spd_frame, text=lbl, font=(FONT, 11, "bold"),
                bg=SURFACE2, fg=AMBER,
                activebackground=AMBER, activeforeground="#000",
                relief="flat", bd=0, pady=6, padx=14,
                command=lambda v=val: self._speed_var.set(v),
            ).pack(side="left", padx=2)

        self._speed_lbl = tk.Label(
            spd_frame, text="20 rad/s", font=(FONT, 10, "bold"),
            bg=BG, fg=AMBER, anchor="e",
        )
        self._speed_lbl.pack(side="right")
        self._speed_var.trace_add("write", self._update_speed_label)

        spd_slider_frame = tk.Frame(r, bg=BG)
        spd_slider_frame.pack(fill="x", padx=16, pady=(4, 0))
        self._speed_slider = tk.Scale(
            spd_slider_frame, from_=1, to=30, orient="horizontal",
            variable=self._speed_var, resolution=1,
            bg=BG, fg=FG, troughcolor=SURFACE2,
            highlightthickness=0, font=(FONT, 8),
            showvalue=False,
        )
        self._speed_slider.pack(fill="x")

        # ── Pad & Arm section ───────────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(r, text="USER STRIKE DETECTION", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(8, 0))

        body = tk.Frame(r, bg=BG)
        body.pack(padx=16, pady=(6, 0))

        # Layout from USER's perspective: left pad on screen-left, right on screen-right
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
        tk.Label(r, text="ROBOT ARM PUNCHES", font=(FONT, 10, "bold"),
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
                punch_row, text=label, font=(FONT, 11, "bold"),
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
                combo_row, text=label, font=(FONT, 11, "bold"),
                bg=SURFACE, fg=PRIMARY,
                activebackground=PRIMARY, activeforeground="#000",
                relief="flat", bd=0, pady=6, padx=8,
                command=lambda s=seq: self._play_combo(s),
            ).pack(side="left", expand=True, fill="x", padx=1)

        # ── Robot arm execution section ─────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(r, text="ROBOT ARM EXECUTION", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(8, 0))

        exec_row1 = tk.Frame(r, bg=BG)
        exec_row1.pack(fill="x", padx=16, pady=(4, 0))

        tk.Checkbutton(
            exec_row1, text="Auto Execute", font=(FONT, 10),
            variable=self._auto_execute,
            bg=BG, fg=FG, selectcolor=SURFACE2,
            activebackground=BG, activeforeground=FG,
        ).pack(side="left", padx=(0, 12))

        tk.Checkbutton(
            exec_row1, text="Forward to HW", font=(FONT, 10),
            variable=self._fwd_hardware,
            bg=BG, fg=FG, selectcolor=SURFACE2,
            activebackground=BG, activeforeground=FG,
        ).pack(side="left")

        exec_row2 = tk.Frame(r, bg=BG)
        exec_row2.pack(fill="x", padx=16, pady=(4, 0))

        self._pending_lbl = tk.Label(
            exec_row2, text="Pending: --", font=(FONT_M, 10),
            bg=BG, fg=FG_MUTED, anchor="w",
        )
        self._pending_lbl.pack(side="left", expand=True, fill="x")

        exec_row3 = tk.Frame(r, bg=BG)
        exec_row3.pack(fill="x", padx=16, pady=(4, 0))

        tk.Button(
            exec_row3, text="EXECUTE", font=(FONT, 11, "bold"),
            bg=GREEN, fg="#000",
            activebackground="#3FB950", activeforeground="#000",
            relief="flat", bd=0, pady=6, padx=14,
            command=self._manual_execute,
        ).pack(side="left", padx=(0, 12))

        tk.Label(exec_row3, text=f"Delay: speed-based",
                 font=(FONT, 10), bg=BG, fg=FG_DIM).pack(side="left")

        arm_row = tk.Frame(r, bg=BG)
        arm_row.pack(fill="x", padx=16, pady=(4, 0))

        self._arm_status_lbls = {}
        self._arm_status_lbls["left"] = tk.Label(
            arm_row, text="L ARM: idle", font=(FONT_M, 10, "bold"),
            bg=BG, fg=FG_MUTED, anchor="w",
        )
        self._arm_status_lbls["left"].pack(side="left", expand=True, fill="x")

        self._arm_status_lbls["right"] = tk.Label(
            arm_row, text="R ARM: idle", font=(FONT_M, 10, "bold"),
            bg=BG, fg=FG_MUTED, anchor="w",
        )
        self._arm_status_lbls["right"].pack(side="left", expand=True, fill="x")

        # ── Height & Tracking status ────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(8, 0))
        tk.Label(r, text="ROBOT STATUS", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(6, 0))

        status_row = tk.Frame(r, bg=SURFACE, highlightthickness=0)
        status_row.pack(fill="x", padx=16, pady=(2, 0))

        height_cell = tk.Frame(status_row, bg=SURFACE)
        height_cell.pack(side="left", expand=True, fill="x", padx=4, pady=6)
        tk.Label(height_cell, text="Height", font=(FONT, 10),
                 bg=SURFACE, fg=FG_DIM).pack()
        self._height_lbl = tk.Label(
            height_cell, text="STOP", font=(FONT, 14, "bold"),
            bg=SURFACE, fg=FG_MUTED,
        )
        self._height_lbl.pack()

        tracking_cell = tk.Frame(status_row, bg=SURFACE)
        tracking_cell.pack(side="left", expand=True, fill="x", padx=4, pady=6)
        tk.Label(tracking_cell, text="Tracking", font=(FONT, 10),
                 bg=SURFACE, fg=FG_DIM).pack()
        self._tracking_lbl = tk.Label(
            tracking_cell, text="OFFLINE", font=(FONT, 14, "bold"),
            bg=SURFACE, fg=FG_MUTED,
        )
        self._tracking_lbl.pack()

        # ── Sequence builder (compact) ──────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(r, text="CUSTOM SEQUENCE", font=(FONT, 10, "bold"),
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
        tk.Label(r, text="TEENSY LIVE DATA", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(6, 0))

        hw_frame = tk.Frame(r, bg=SURFACE, highlightthickness=0)
        hw_frame.pack(fill="x", padx=16, pady=(2, 0))

        # Motor positions row
        motor_row = tk.Frame(hw_frame, bg=SURFACE)
        motor_row.pack(fill="x", padx=8, pady=(6, 0))
        _motor_names = ["L1", "L2", "R1", "R2"]
        tk.Label(motor_row, text="Motors", font=(FONT, 12, "bold"),
                 bg=SURFACE, fg=FG_DIM).pack(side="left", padx=(0, 6))
        self._hw_pos_labels = []
        self._hw_cur_labels = []
        for i in range(4):
            cell = tk.Frame(motor_row, bg=SURFACE)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=_motor_names[i], font=(FONT, 11),
                     bg=SURFACE, fg=FG_MUTED).pack()
            plbl = tk.Label(cell, text="--", font=(FONT, 13, "bold"),
                            bg=SURFACE, fg=FG)
            plbl.pack()
            clbl = tk.Label(cell, text="--", font=(FONT, 11),
                            bg=SURFACE, fg=FG_DIM)
            clbl.pack()
            self._hw_pos_labels.append(plbl)
            self._hw_cur_labels.append(clbl)

        # IMU accel magnitudes row (per pad) + threshold indicator
        imu_header = tk.Frame(hw_frame, bg=SURFACE)
        imu_header.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(imu_header, text="IMU", font=(FONT, 12, "bold"),
                 bg=SURFACE, fg=FG_DIM).pack(side="left")
        self._strike_threshold = 20.0
        tk.Label(imu_header, text=f"strike threshold: {self._strike_threshold:.0f} m/s\u00B2",
                 font=(FONT, 10), bg=SURFACE, fg=FG_MUTED).pack(side="left", padx=(8, 0))
        tk.Label(imu_header, text="(gravity-subtracted)",
                 font=(FONT, 9), bg=SURFACE, fg=FG_MUTED).pack(side="left", padx=(4, 0))

        imu_row = tk.Frame(hw_frame, bg=SURFACE)
        imu_row.pack(fill="x", padx=8, pady=(2, 6))
        _imu_pad_names = ["Centre", "Left", "Right", "Head"]
        _imu_pad_colors = [PRIMARY, PUNCH_JAB, PUNCH_UPPER, AMBER]
        self._hw_imu_labels = []
        self._hw_imu_peak_labels = []
        for i in range(4):
            cell = tk.Frame(imu_row, bg=SURFACE)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=_imu_pad_names[i], font=(FONT, 11),
                     bg=SURFACE, fg=_imu_pad_colors[i]).pack()
            # Raw magnitude (from motor_feedback, includes gravity)
            lbl = tk.Label(cell, text="--", font=(FONT, 14, "bold"),
                           bg=SURFACE, fg=FG_DIM)
            lbl.pack()
            self._hw_imu_labels.append(lbl)
            # Last detected peak (from /robot/strike_detected, gravity-subtracted)
            peak_lbl = tk.Label(cell, text="", font=(FONT, 10),
                                bg=SURFACE, fg=FG_MUTED)
            peak_lbl.pack()
            self._hw_imu_peak_labels.append(peak_lbl)

        # ── Event log ───────────────────────────────────────────────────
        tk.Frame(r, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(8, 0))
        tk.Label(r, text="EVENT LOG", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(6, 0))

        self._log_text = tk.Text(
            r, height=4, width=60, bg=SURFACE,
            fg=FG_DIM, font=(FONT_M, 10),
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

    # ── Execute / auto-execute handlers ──────────────────────────────────
    def _on_execute_cmd(self, cmd: dict) -> None:
        """Called from ROS thread when RobotCommand arrives."""
        self._root.after(0, lambda: self._handle_incoming_cmd(cmd))

    def _handle_incoming_cmd(self, cmd: dict) -> None:
        """Process an incoming robot command on the main thread.

        If hardware is connected OR Forward to HW is on, send to V4 GUI.
        Otherwise use simulated execution (auto or manual).
        Only one arm at a time — queue if already executing.
        """
        self._pending_cmd = cmd
        punch_name = _CODE_TO_PUNCH.get(cmd["code"], cmd["code"])
        arm = cmd.get("arm", "?")
        self._pending_lbl.configure(
            text=f"Pending: {punch_name} \u2192 {arm} arm", fg=AMBER)
        # Reset both arm labels, then highlight the active one
        for side, lbl in self._arm_status_lbls.items():
            prefix = "L" if side == "left" else "R"
            lbl.configure(text=f"{prefix} ARM: idle", fg=FG_MUTED)
        if arm in ("left", "right"):
            lbl = self._arm_status_lbls.get(arm)
            if lbl:
                prefix = "L" if arm == "left" else "R"
                lbl.configure(text=f"{prefix} ARM: pending", fg=AMBER)

        # Forward to V4 GUI hardware if connected or toggle is on
        use_hardware = self._fwd_hardware.get() or self._node._teensy_connected
        if use_hardware:
            slot = cmd.get("slot")
            speed = cmd.get("speed")
            if slot:
                self._node.send_strike_command(slot, duration=5.0, speed=speed)
            # V4 GUI handles execution — it will publish strike_feedback when done
            return

        # No hardware — use simulated execution
        if self._auto_execute.get() and not self._executing:
            self._start_simulated_execution(cmd)

    def _manual_execute(self) -> None:
        """Execute button pressed — run pending command if available."""
        if self._pending_cmd and not self._executing:
            self._start_simulated_execution(self._pending_cmd)

    def _start_simulated_execution(self, cmd: dict) -> None:
        """Begin simulated strike execution with speed-based delay."""
        self._executing = True
        arm = cmd.get("arm", "left")
        lbl = self._arm_status_lbls.get(arm)
        if lbl:
            prefix = "L" if arm == "left" else "R"
            lbl.configure(text=f"{prefix} ARM: EXECUTING", fg=RED)

        speed = self._speed_var.get()
        delay_s = 10.0 / max(speed, 1.0)
        delay_ms = int(delay_s * 1000)

        self._root.after(
            delay_ms, lambda: self._finish_execution(cmd, delay_s))

    def _finish_execution(self, cmd: dict, delay_s: float) -> None:
        """Complete simulated strike and publish feedback."""
        slot = cmd.get("slot", 0)
        punch_name = _CODE_TO_PUNCH.get(cmd.get("code", ""), "unknown")
        _STRIKE_NAMES = {
            "jab": "Jab", "cross": "Cross",
            "l_hook": "Left Hook", "r_hook": "Right Hook",
            "l_upper": "Left Uppercut", "r_upper": "Right Uppercut",
        }
        strike_name = _STRIKE_NAMES.get(punch_name, punch_name)

        self._node.simulate_strike_complete(slot, strike_name, delay_s)

        arm = cmd.get("arm", "left")
        lbl = self._arm_status_lbls.get(arm)
        if lbl:
            prefix = "L" if arm == "left" else "R"
            lbl.configure(text=f"{prefix} ARM: idle", fg=FG_MUTED)
        self._pending_lbl.configure(text="Pending: --", fg=FG_MUTED)
        self._executing = False
        self._log(f"SIM>  {strike_name:<12s}  completed  ({delay_s:.1f}s)")
        # If another command arrived during execution, process it now
        if self._pending_cmd and self._pending_cmd != cmd and self._auto_execute.get():
            self._start_simulated_execution(self._pending_cmd)
        else:
            self._pending_cmd = None
            self._pending_lbl.configure(text="Pending: --", fg=FG_MUTED)

    # ── Height & tracking status ──────────────────────────────────────
    def _update_height(self, action: str) -> None:
        """Update height status label."""
        colors = {
            "manual_up": GREEN, "manual_down": RED,
            "stop": FG_MUTED, "adjust": AMBER,
        }
        labels = {
            "manual_up": "UP", "manual_down": "DOWN",
            "stop": "STOP", "adjust": "AUTO",
        }
        self._height_lbl.configure(
            text=labels.get(action, action.upper()),
            fg=colors.get(action, FG_MUTED),
        )

    def _update_direction(self, direction: str) -> None:
        """Update person tracking direction label."""
        colors = {"left": PUNCH_JAB, "right": PUNCH_UPPER, "centre": GREEN}
        self._tracking_lbl.configure(
            text=direction.upper(),
            fg=colors.get(direction, FG_MUTED),
        )

    def _check_tracking_offline(self) -> None:
        """Periodically check if tracking direction has gone stale."""
        if time.time() - getattr(self._node, '_last_direction_time', 0.0) > 2.0:
            self._tracking_lbl.configure(text="OFFLINE", fg=FG_MUTED)
        self._root.after(2000, self._check_tracking_offline)

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
            parent, text=label, width=10, height=4,
            font=(FONT, 14, "bold"),
            bg=PAD_BG, fg=RED,
            activebackground=PAD_FLASH, activeforeground="#FFF",
            relief="flat", bd=0,
            command=lambda: self._on_pad(label.lower()),
        )
        btn.grid(row=row, column=col, padx=4, pady=4)
        return btn

    def _make_arm(self, parent: tk.Frame, label: str,
                  row: int, col: int) -> tk.Button:
        btn = tk.Button(
            parent, text=label, width=5, height=10,
            font=(FONT, 18, "bold"),
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

    def _update_speed_label(self, *_args) -> None:
        val = self._speed_var.get()
        if val <= 10:
            color = GREEN
        elif val <= 20:
            color = AMBER
        else:
            color = RED
        self._speed_lbl.configure(text=f"{val:.0f} rad/s", fg=color)

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
        """Execute a robot punch via the V4 Arm Control GUI.

        Sends /robot/strike_command to the V4 GUI which handles the full
        FSM execution (alignment, windup, apex, snap-back) and publishes
        motor_commands to the Teensy.
        """
        level = self._get_force_level()
        force = self._get_force_normalized()
        self._node.publish_punch(punch_type, level, force)
        arm_side, _ = _PUNCH_TYPES.get(punch_type, ("left", "centre"))

        # Send strike command to V4 GUI (slot-based)
        _PUNCH_TO_CODE = {
            "jab": "1", "cross": "2", "l_hook": "3",
            "r_hook": "4", "l_upper": "5", "r_upper": "6",
        }
        code = _PUNCH_TO_CODE.get(punch_type)
        if code:
            slot = self._node._code_to_slot.get(code)
            if slot:
                spd = self._speed_var.get()
                self._node.send_strike_command(slot, duration=5.0, speed=spd)

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
        """Play a combo sequence, waiting for each strike to complete."""
        self._combo_queue = list(sequence)
        self._combo_playing = True
        self._play_next_combo_step()

    def _play_next_combo_step(self) -> None:
        """Send the next punch in the combo queue."""
        if not self._combo_queue:
            self._combo_playing = False
            return
        ptype = self._combo_queue.pop(0)
        color_map = {
            "jab": PUNCH_JAB, "cross": PUNCH_CROSS,
            "l_hook": PUNCH_HOOK, "r_hook": PUNCH_HOOK,
            "l_upper": PUNCH_UPPER, "r_upper": PUNCH_UPPER,
        }
        self._on_punch(ptype, color_map.get(ptype, PRIMARY))
        # Timeout: if no feedback within 10s, move on anyway
        self._combo_timeout_id = self._root.after(
            10000, self._play_next_combo_step)

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

    # ── V4 GUI strike feedback ─────────────────────────────────────────
    def _on_strike_feedback_gui(self, data: dict) -> None:
        """Called from ROS thread when V4 GUI reports strike completion."""
        self._root.after(0, lambda: self._show_strike_feedback(data))

    def _show_strike_feedback(self, data: dict) -> None:
        strike = data.get("strike", "?")
        status = data.get("status", "?")
        dur = data.get("duration_actual", 0.0)
        color = GREEN if status == "completed" else (AMBER if status == "overtime" else RED)
        self._log(f"FB>  {strike:<12s}  {status}  ({dur:.1f}s)")
        # Advance combo queue on strike completion
        if getattr(self, '_combo_playing', False):
            # Cancel the timeout since we got real feedback
            if hasattr(self, '_combo_timeout_id'):
                self._root.after_cancel(self._combo_timeout_id)
            self._play_next_combo_step()

    # ── Auto-setup V4 GUI slots ────────────────────────────────────────
    def _setup_v4_slots(self) -> None:
        """Publish default punch slot assignments to the V4 GUI.

        Maps punch codes 1-6 to the standard strike names that must
        exist in the V4 GUI's strike library.
        """
        default_slots = {
            1: {"arm": "left",  "strike": "Jab"},
            2: {"arm": "right", "strike": "Cross"},
            3: {"arm": "left",  "strike": "Left Hook"},
            4: {"arm": "right", "strike": "Right Hook"},
            5: {"arm": "left",  "strike": "Left Uppercut"},
            6: {"arm": "right", "strike": "Right Uppercut"},
        }
        self._node.send_punch_slots(default_slots)
        self._node.send_system_enable(True)
        self._log("AUTO  Punch slots assigned to V4 GUI (1-6)")
        self._log("AUTO  System enabled — V4 GUI ROS Control must be ON")

    # ── Run ─────────────────────────────────────────────────────────────
    def spin(self) -> None:
        t = threading.Thread(target=rclpy.spin, args=(self._node,),
                             daemon=True)
        t.start()
        # Auto-setup V4 GUI slots after a short delay (let ROS topics connect)
        self._root.after(2000, self._setup_v4_slots)
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
        node = TeensySimulatorNode()
        gui = TeensySimulatorGUI(node)
        gui.spin()
    except KeyboardInterrupt:
        log.info("Shutting down Teensy simulator")
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
