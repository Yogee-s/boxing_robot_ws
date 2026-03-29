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
from tkinter import ttk
from typing import Optional

# ── ROS 2 + boxbunny_msgs (graceful fallback) ──────────────────────────
try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    raise SystemExit(
        "rclpy not found. Source your ROS 2 workspace first:\n"
        "  source /opt/ros/humble/setup.bash"
    )

try:
    from boxbunny_msgs.msg import ArmStrike, IMUStatus, PadImpact
except ImportError:
    raise SystemExit(
        "boxbunny_msgs not found. Build the workspace first:\n"
        "  cd boxing_robot_ws && colcon build --packages-select boxbunny_msgs"
    )

# ── Constants (mirrors boxbunny_core.constants) ────────────────────────
_TOPIC_PAD_IMPACT = "/boxbunny/imu/pad/impact"
_TOPIC_ARM_STRIKE = "/boxbunny/imu/arm/strike"
_TOPIC_STATUS = "/boxbunny/imu/status"

_PAD_LOCATIONS = ("left", "centre", "right", "head")
_ARM_SIDES = ("left", "right")
_FORCE_LEVELS = ("light", "medium", "hard")

_STATUS_PERIOD_SEC = 1.0
_LOG_MAX_LINES = 80

# ── Colours ─────────────────────────────────────────────────────────────
_BG = "#1e1e2e"
_FG = "#cdd6f4"
_PAD_BG = "#2d5a3d"
_PAD_ACTIVE = "#40826d"
_ARM_BG = "#7a5c2e"
_ARM_ACTIVE = "#b8860b"
_LOG_BG = "#11111b"
_LOG_FG = "#a6adc8"
_ACCENT = "#89b4fa"

log = logging.getLogger("imu_simulator")


class IMUSimulatorNode(Node):
    """Thin ROS 2 node that owns the three publishers."""

    def __init__(self) -> None:
        super().__init__("imu_simulator")
        self._pub_pad = self.create_publisher(PadImpact, _TOPIC_PAD_IMPACT, 10)
        self._pub_arm = self.create_publisher(ArmStrike, _TOPIC_ARM_STRIKE, 10)
        self._pub_status = self.create_publisher(IMUStatus, _TOPIC_STATUS, 10)
        self._status_timer = self.create_timer(_STATUS_PERIOD_SEC, self._publish_status)
        self.get_logger().info("IMU simulator node started")

    # ── publishers ──────────────────────────────────────────────────────
    def publish_pad_impact(self, pad: str, level: str) -> None:
        msg = PadImpact()
        msg.timestamp = time.time()
        msg.pad = pad
        msg.level = level
        self._pub_pad.publish(msg)
        self.get_logger().info(f"PadImpact  pad={pad}  level={level}")

    def publish_arm_strike(self, arm: str, contact: bool) -> None:
        msg = ArmStrike()
        msg.timestamp = time.time()
        msg.arm = arm
        msg.contact = contact
        self._pub_arm.publish(msg)
        tag = "STRUCK" if contact else "MISS"
        self.get_logger().info(f"ArmStrike  arm={arm}  contact={tag}")

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
    """Tkinter front-end that drives *IMUSimulatorNode*."""

    def __init__(self, node: IMUSimulatorNode) -> None:
        self._node = node
        self._force_var: Optional[tk.StringVar] = None
        self._log_lines: deque[str] = deque(maxlen=_LOG_MAX_LINES)

        self._root = tk.Tk()
        self._root.title("BoxBunny IMU Simulator")
        self._root.configure(bg=_BG)
        self._root.resizable(False, False)
        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = self._root
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TRadiobutton", background=_BG, foreground=_FG,
                         font=("Monospace", 10))
        style.configure("TLabel", background=_BG, foreground=_FG,
                         font=("Monospace", 10))
        style.configure("TLabelframe", background=_BG, foreground=_ACCENT,
                         font=("Monospace", 10, "bold"))
        style.configure("TLabelframe.Label", background=_BG, foreground=_ACCENT)

        # ── Title ───────────────────────────────────────────────────────
        ttk.Label(root, text="BoxBunny IMU Simulator", font=("Monospace", 14, "bold"),
                  foreground=_ACCENT).pack(pady=(10, 4))

        # ── Force selector ──────────────────────────────────────────────
        force_frame = ttk.LabelFrame(root, text="Impact Force")
        force_frame.pack(padx=12, pady=(4, 6), fill="x")
        self._force_var = tk.StringVar(value="medium")
        for lvl in _FORCE_LEVELS:
            ttk.Radiobutton(force_frame, text=lvl.capitalize(),
                            variable=self._force_var, value=lvl).pack(
                                side="left", padx=14, pady=4)

        # ── Pad / arm button grid ───────────────────────────────────────
        grid_frame = tk.Frame(root, bg=_BG)
        grid_frame.pack(padx=12, pady=6)

        # Row 0: HEAD centred (spans columns 1-3)
        self._make_pad_btn(grid_frame, "head", row=0, col=1, colspan=3)

        # Row 1: L ARM | LEFT | CENTRE | RIGHT | R ARM
        self._make_arm_btn(grid_frame, "left", row=1, col=0)
        self._make_pad_btn(grid_frame, "left", row=1, col=1)
        self._make_pad_btn(grid_frame, "centre", row=1, col=2)
        self._make_pad_btn(grid_frame, "right", row=1, col=3)
        self._make_arm_btn(grid_frame, "right", row=1, col=4)

        # Hint label
        ttk.Label(root, text="Shift+click arm = miss",
                  foreground=_LOG_FG, font=("Monospace", 9)).pack(pady=(0, 4))

        # ── Log area ────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(root, text="Message Log")
        log_frame.pack(padx=12, pady=(2, 10), fill="both", expand=True)

        self._log_text = tk.Text(log_frame, height=10, width=56, bg=_LOG_BG,
                                 fg=_LOG_FG, font=("Monospace", 9),
                                 state="disabled", wrap="word",
                                 borderwidth=0, highlightthickness=0)
        self._log_text.pack(padx=4, pady=4, fill="both", expand=True)

    # ── Button factories ────────────────────────────────────────────────
    def _make_pad_btn(self, parent: tk.Frame, pad: str,
                      row: int, col: int, colspan: int = 1) -> None:
        label = "HEAD PAD" if pad == "head" else pad.upper()
        btn = tk.Button(parent, text=label, width=10, height=2,
                        bg=_PAD_BG, fg=_FG, activebackground=_PAD_ACTIVE,
                        font=("Monospace", 11, "bold"), relief="raised", bd=2,
                        command=lambda p=pad: self._on_pad_click(p))
        btn.grid(row=row, column=col, columnspan=colspan, padx=4, pady=4,
                 sticky="nsew")

    def _make_arm_btn(self, parent: tk.Frame, side: str,
                      row: int, col: int) -> None:
        label = "L ARM" if side == "left" else "R ARM"
        btn = tk.Button(parent, text=label, width=10, height=2,
                        bg=_ARM_BG, fg=_FG, activebackground=_ARM_ACTIVE,
                        font=("Monospace", 11, "bold"), relief="raised", bd=2)
        btn.bind("<Button-1>", lambda e, s=side: self._on_arm_click(e, s))
        btn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

    # ── Event handlers ──────────────────────────────────────────────────
    def _on_pad_click(self, pad: str) -> None:
        level = self._force_var.get() if self._force_var else "medium"
        self._node.publish_pad_impact(pad, level)
        self._append_log(f"PAD  {pad:<7s}  force={level}")

    def _on_arm_click(self, event: tk.Event, side: str) -> None:
        contact = not bool(event.state & 0x0001)  # Shift held = miss
        self._node.publish_arm_strike(side, contact)
        tag = "struck" if contact else "miss"
        self._append_log(f"ARM  {side:<7s}  {tag}")

    # ── Log helper ──────────────────────────────────────────────────────
    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self._log_lines.append(line)
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ── Main loop ───────────────────────────────────────────────────────
    def spin(self) -> None:
        """Run Tk mainloop while spinning the ROS node in a background thread."""
        spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,),
                                       daemon=True)
        spin_thread.start()
        try:
            self._root.mainloop()
        finally:
            self._node.destroy_node()


def main() -> None:
    """Entry point -- initialise rclpy, build the GUI, and run."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
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
