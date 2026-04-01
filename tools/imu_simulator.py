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

# Punch colours for the simulator
PUNCH_JAB    = "#58A6FF"
PUNCH_CROSS  = "#FF5C5C"
PUNCH_HOOK   = "#56D364"
PUNCH_UPPER  = "#BC8CFF"

# ── Theme ────────────────────────────────────────────────────────────────
BG       = "#0D0D0D"
SURFACE  = "#161616"
SURFACE2 = "#1E1E1E"
FG       = "#E8E8E8"
FG_DIM   = "#888888"
FG_MUTED = "#444444"
PRIMARY  = "#FF6B35"
BORDER   = "#252525"

# Force level colours
GREEN   = "#2ECC71"
AMBER   = "#F39C12"
RED     = "#E74C3C"

# Pad resting colour (dark crimson)
PAD_BG     = "#2A1216"
PAD_BORDER = "#4A2030"

# Arm resting colour (dark navy)
ARM_BG     = "#121A2E"
ARM_BORDER = "#1E3050"

# Sequence panel
SEQ_BG = "#111111"

FONT     = "Helvetica"
FONT_M   = "Monospace"

log = logging.getLogger("imu_simulator")


class IMUSimulatorNode(Node):
    """Thin ROS 2 node with publishers and a robot-command subscriber."""

    def __init__(self) -> None:
        super().__init__("imu_simulator")
        self._pub_pad = self.create_publisher(PadImpact, _TOPIC_PAD, 10)
        self._pub_arm = self.create_publisher(ArmStrike, _TOPIC_ARM, 10)
        self._pub_status = self.create_publisher(IMUStatus, _TOPIC_STATUS, 10)
        self._pub_punch = self.create_publisher(ConfirmedPunch, _TOPIC_PUNCH, 10)
        self.create_timer(1.0, self._publish_status)

        # Subscribe to robot commands so the GUI drill cycling shows up
        self.create_subscription(
            RobotCommand, _TOPIC_ROBOT_CMD, self._on_robot_command, 10,
        )
        self._robot_cmd_callback = None  # set by GUI
        self.get_logger().info("IMU simulator node started")

    def _on_robot_command(self, msg: RobotCommand) -> None:
        """Forward incoming robot commands to the GUI for visual flash."""
        if msg.command_type == "punch" and self._robot_cmd_callback:
            punch_type = _CODE_TO_PUNCH.get(msg.punch_code, "")
            if punch_type:
                self._robot_cmd_callback(punch_type)

    def publish_pad(self, pad: str, level: str) -> None:
        msg = PadImpact()
        msg.timestamp = time.time()
        msg.pad = pad
        msg.level = level
        self._pub_pad.publish(msg)
        self.get_logger().info(f"PadImpact pad={pad} level={level}")

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
        self._pub_punch.publish(msg)
        # Also publish the pad impact and arm strike
        self.publish_pad(pad, level)
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
        msg.is_simulator = True
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

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self) -> None:
        r = self._root

        # Title
        top = tk.Frame(r, bg=SURFACE, height=48)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="Box", font=(FONT, 15, "bold"),
                 bg=SURFACE, fg=FG).pack(side="left", padx=(14, 0))
        tk.Label(top, text="Bunny", font=(FONT, 15, "bold"),
                 bg=SURFACE, fg=PRIMARY).pack(side="left")
        tk.Label(top, text="IMU Simulator", font=(FONT, 10),
                 bg=SURFACE, fg=FG_DIM).pack(side="left", padx=8)

        # ── Acceleration slider ─────────────────────────────────────────
        tk.Label(r, text="ACCELERATION (m/s\u00B2)", font=(FONT, 8),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=20, pady=(12, 0))

        accel_row = tk.Frame(r, bg=BG)
        accel_row.pack(fill="x", padx=20, pady=(4, 0))

        self._accel_var = tk.DoubleVar(value=30.0)
        self._accel_slider = tk.Scale(
            accel_row, from_=0, to=60, orient="horizontal",
            variable=self._accel_var, resolution=0.5,
            bg=BG, fg=FG, troughcolor=SURFACE2,
            highlightthickness=0, font=(FONT, 9),
            length=300,
        )
        self._accel_slider.pack(side="left", fill="x", expand=True)

        self._accel_lbl = tk.Label(
            accel_row, text="30.0", font=(FONT, 12, "bold"),
            bg=BG, fg=AMBER, width=6,
        )
        self._accel_lbl.pack(side="left", padx=(8, 0))
        self._accel_var.trace_add("write", self._update_accel_label)

        # Quick preset buttons
        preset_row = tk.Frame(r, bg=BG)
        preset_row.pack(fill="x", padx=20, pady=(2, 0))
        for val, lbl, color in [
            (10, "Light", GREEN), (30, "Medium", AMBER), (50, "Hard", RED),
        ]:
            tk.Button(
                preset_row, text=f"{lbl} ({val})", font=(FONT, 8),
                bg=SURFACE2, fg=color,
                activebackground=color, activeforeground="#000",
                relief="flat", bd=0, pady=3,
                command=lambda v=val: self._accel_var.set(v),
            ).pack(side="left", expand=True, fill="x", padx=1)

        # ── Robot body ──────────────────────────────────────────────────
        body = tk.Frame(r, bg=BG)
        body.pack(padx=16, pady=(14, 6))

        # Left arm
        self._arm_btns["left"] = self._make_arm(body, "L", 0, 0)

        # Pads
        pf = tk.Frame(body, bg=BG)
        pf.grid(row=0, column=1, padx=8)
        self._pad_btns["head"] = self._make_pad(pf, "HEAD", 0, 1)
        self._pad_btns["left"] = self._make_pad(pf, "LEFT", 1, 0)
        self._pad_btns["centre"] = self._make_pad(pf, "CENTRE", 1, 1)
        self._pad_btns["right"] = self._make_pad(pf, "RIGHT", 1, 2)

        # Right arm
        self._arm_btns["right"] = self._make_arm(body, "R", 0, 2)

        # Hint
        tk.Label(r, text="STRIKE DETECTION: click pad = user hit detected    click arm = user strike",
                 font=(FONT, 7), bg=BG, fg=FG_MUTED).pack(pady=(0, 6))

        # ── Sequence builder ────────────────────────────────────────────
        sep = tk.Frame(r, bg=BORDER, height=1)
        sep.pack(fill="x", padx=20, pady=(4, 0))

        tk.Label(r, text="SEQUENCE BUILDER", font=(FONT, 8),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=20, pady=(8, 0))

        # Add-to-sequence buttons row
        add_row = tk.Frame(r, bg=BG)
        add_row.pack(fill="x", padx=20, pady=(4, 0))

        for pad in ["HEAD", "LEFT", "CENTRE", "RIGHT"]:
            tk.Button(
                add_row, text=f"+{pad}", font=(FONT, 8),
                bg=PAD_BG, fg="#D88", activebackground=PAD_BORDER,
                relief="flat", bd=0, padx=6, pady=3,
                command=lambda p=pad.lower(): self._seq_add("pad", p),
            ).pack(side="left", padx=1, expand=True, fill="x")

        for arm in ["L ARM", "R ARM"]:
            side = "left" if "L" in arm else "right"
            tk.Button(
                add_row, text=f"+{arm}", font=(FONT, 8),
                bg=ARM_BG, fg="#88B8E8", activebackground=ARM_BORDER,
                relief="flat", bd=0, padx=6, pady=3,
                command=lambda s=side: self._seq_add("arm", s),
            ).pack(side="left", padx=1, expand=True, fill="x")

        # Sequence display
        self._seq_frame = tk.Frame(r, bg=SEQ_BG, height=36)
        self._seq_frame.pack(fill="x", padx=20, pady=(4, 0))
        self._seq_label = tk.Label(
            self._seq_frame, text="(empty — click +PAD buttons above to build)",
            font=(FONT_M, 9), bg=SEQ_BG, fg=FG_MUTED, anchor="w",
        )
        self._seq_label.pack(fill="x", padx=8, pady=6)

        # Interval + controls
        ctrl_row = tk.Frame(r, bg=BG)
        ctrl_row.pack(fill="x", padx=20, pady=(4, 0))

        tk.Label(ctrl_row, text="Interval:", font=(FONT, 9),
                 bg=BG, fg=FG_DIM).pack(side="left")

        self._interval_var = tk.StringVar(value="500")
        for ms_val, lbl in [("300", "Fast"), ("500", "Medium"), ("1000", "Slow")]:
            tk.Radiobutton(
                ctrl_row, text=lbl, variable=self._interval_var,
                value=ms_val, font=(FONT, 9),
                bg=BG, fg=FG_DIM, selectcolor=SURFACE2,
                activebackground=BG, activeforeground=FG,
                highlightthickness=0,
            ).pack(side="left", padx=4)

        tk.Frame(ctrl_row, bg=BG, width=20).pack(side="left", expand=True)

        self._play_btn = tk.Button(
            ctrl_row, text="PLAY", font=(FONT, 9, "bold"),
            bg=GREEN, fg="#000", activebackground="#27AE60",
            relief="flat", bd=0, padx=14, pady=3,
            command=self._seq_play,
        )
        self._play_btn.pack(side="left", padx=2)

        tk.Button(
            ctrl_row, text="CLEAR", font=(FONT, 9, "bold"),
            bg=SURFACE2, fg=FG_DIM, activebackground=BORDER,
            relief="flat", bd=0, padx=10, pady=3,
            command=self._seq_clear,
        ).pack(side="left", padx=2)

        # ── Punch simulator ─────────────────────────────────────────────
        sep_punch = tk.Frame(r, bg=BORDER, height=1)
        sep_punch.pack(fill="x", padx=20, pady=(8, 0))

        tk.Label(r, text="ROBOT ARM OUTPUT (simulates robot punching)", font=(FONT, 8),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=20, pady=(8, 0))

        punch_row = tk.Frame(r, bg=BG)
        punch_row.pack(fill="x", padx=20, pady=(4, 0))

        punch_defs = [
            ("Jab",      "jab",     PUNCH_JAB,   "L >"),
            ("Cross",    "cross",   PUNCH_CROSS,  "R >"),
            ("L Hook",   "l_hook",  PUNCH_HOOK,  "L ~"),
            ("R Hook",   "r_hook",  PUNCH_HOOK,  "R ~"),
            ("L Upper",  "l_upper", PUNCH_UPPER, "L ^"),
            ("R Upper",  "r_upper", PUNCH_UPPER, "R ^"),
        ]
        self._punch_btns: dict[str, tk.Button] = {}
        for label, ptype, color, arm_lbl in punch_defs:
            btn = tk.Button(
                punch_row, text=f"{arm_lbl}\n{label}", font=(FONT, 8, "bold"),
                bg=SURFACE2, fg=color,
                activebackground=color, activeforeground="#000",
                relief="flat", bd=0, pady=5,
                command=lambda pt=ptype, c=color: self._on_punch(pt, c),
            )
            btn.pack(side="left", expand=True, fill="x", padx=1)
            self._punch_btns[ptype] = btn

        # Combo preset row
        combo_row = tk.Frame(r, bg=BG)
        combo_row.pack(fill="x", padx=20, pady=(4, 0))

        combos = [
            ("1-2",       ["jab", "cross"]),
            ("1-1-2",     ["jab", "jab", "cross"]),
            ("1-2-3",     ["jab", "cross", "l_hook"]),
            ("1-2-3-4",   ["jab", "cross", "l_hook", "r_hook"]),
            ("1-2-5-6",   ["jab", "cross", "l_upper", "r_upper"]),
        ]
        for label, seq in combos:
            tk.Button(
                combo_row, text=label, font=(FONT, 8, "bold"),
                bg="#1A1510", fg=PRIMARY,
                activebackground=PRIMARY, activeforeground="#000",
                relief="flat", bd=0, pady=4, padx=8,
                command=lambda s=seq: self._play_combo(s),
            ).pack(side="left", expand=True, fill="x", padx=1)

        # ── Log area ────────────────────────────────────────────────────
        sep2 = tk.Frame(r, bg=BORDER, height=1)
        sep2.pack(fill="x", padx=20, pady=(8, 0))

        tk.Label(r, text="EVENT LOG", font=(FONT, 8),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=20, pady=(6, 0))

        self._log_text = tk.Text(
            r, height=6, width=55, bg=SEQ_BG,
            fg="#666", font=(FONT_M, 9),
            state="disabled", wrap="word",
            borderwidth=0, highlightthickness=0,
        )
        self._log_text.pack(padx=20, pady=(2, 14), fill="both", expand=True)

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
            self._flash(self._arm_btns[arm_side], color, ARM_BG, "#6EA8DC")

        _PUNCH_LABELS = {
            "jab": "Jab", "cross": "Cross", "l_hook": "L Hook",
            "r_hook": "R Hook", "l_upper": "L Upper", "r_upper": "R Upper",
        }
        self._log(f"CMD>  {_PUNCH_LABELS.get(punch_type, punch_type):<8s}  "
                  f"(from GUI drill)")

    # ── Factories ───────────────────────────────────────────────────────
    def _make_pad(self, parent: tk.Frame, label: str,
                  row: int, col: int) -> tk.Button:
        btn = tk.Button(
            parent, text=label, width=8, height=3,
            font=(FONT, 10, "bold"),
            bg=PAD_BG, fg="#E88",
            activebackground="#5A2030", activeforeground="#FFF",
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
            bg=ARM_BG, fg="#6EA8DC",
            activebackground="#2A5090", activeforeground="#FFF",
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
        self._accel_lbl.configure(text=f"{val:.1f}", fg=color)

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
        self._node.publish_pad(pad, level)
        # Also publish a ConfirmedPunch so the GUI detects it
        self._node.publish_punch("strike", level, force, pad_override=pad)
        colors = {"light": GREEN, "medium": AMBER, "hard": RED}
        if pad in self._pad_btns:
            self._flash(self._pad_btns[pad], colors[level],
                        PAD_BG, "#E88")
        self._log(f"PAD  {pad:<7s}  accel={self._accel_var.get():.1f}  level={level}")

    def _on_arm(self, event: tk.Event, side: str) -> None:
        contact = not bool(event.state & 0x0001)
        self._node.publish_arm(side, contact)
        color = "#3B82F6" if contact else "#666"
        if side in self._arm_btns:
            self._flash(self._arm_btns[side], color, ARM_BG, "#6EA8DC")
        tag = "struck" if contact else "miss"
        self._log(f"ARM  {side:<7s}  {tag}")

    # ── Punch simulator ─────────────────────────────────────────────────
    def _on_punch(self, punch_type: str, color: str) -> None:
        """Simulate robot arm throwing a punch — only flashes arm, not pads."""
        level = self._get_force_level()
        force = self._get_force_normalized()
        self._node.publish_punch(punch_type, level, force)
        arm_side, _ = _PUNCH_TYPES.get(punch_type, ("left", "centre"))

        # Flash the punch button
        if punch_type in self._punch_btns:
            btn = self._punch_btns[punch_type]
            btn.configure(bg=color, fg="#000")
            btn.after(_FLASH_MS, lambda: btn.configure(bg=SURFACE2, fg=color))

        # Flash the arm (robot arm moving)
        if arm_side in self._arm_btns:
            self._flash(self._arm_btns[arm_side], color, ARM_BG, "#6EA8DC")

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
