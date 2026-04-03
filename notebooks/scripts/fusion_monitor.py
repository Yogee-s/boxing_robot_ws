#!/usr/bin/env python3
"""Fusion Monitor — shows IMU strikes + CV+IMU confirmed results.

Only listens to:
  /robot/strike_detected     — from V4 GUI (already gravity-calibrated, debounced)
  /boxbunny/punch/confirmed  — final fused output from punch_processor

No duplicate sources. V4 GUI does the IMU filtering, punch_processor does CV+IMU fusion.
"""
import json
import threading
import time
import tkinter as tk

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String as StdString
except ImportError:
    raise SystemExit("rclpy not found. Source ROS 2 first.")

try:
    from boxbunny_msgs.msg import ConfirmedPunch
except ImportError:
    raise SystemExit("boxbunny_msgs not found. Build workspace first.")

BG = "#0B0F14"
SURFACE = "#131920"
SURFACE2 = "#1A2029"
FG = "#E6EDF3"
FG_DIM = "#8B949E"
FG_MUTED = "#484F58"
GREEN = "#56D364"
RED = "#FF5C5C"
AMBER = "#FFAB40"
BLUE = "#58A6FF"
PURPLE = "#BC8CFF"
PRIMARY = "#FF6B35"
FONT = "Helvetica"
FONT_M = "Monospace"

# Pad index → name (user perspective, matches imu_pad_map)
PAD_MAP = {0: "centre", 1: "right", 2: "left", 3: "head"}

PAD_EXPECTS = {
    "centre": "jab / cross",
    "left": "l_hook / l_upper",
    "right": "r_hook / r_upper",
    "head": "any",
}

PUNCH_COLORS = {
    "jab": BLUE, "cross": RED, "left_hook": GREEN, "right_hook": AMBER,
    "left_uppercut": PURPLE, "right_uppercut": "#F8E45C",
}


class FusionNode(Node):
    def __init__(self):
        super().__init__("fusion_monitor")
        self.cb_strike = None
        self.cb_confirmed = None
        self.cb_cv_state = None
        self.create_subscription(StdString, "/robot/strike_detected", self._on_strike, 10)
        self.create_subscription(ConfirmedPunch, "/boxbunny/punch/confirmed", self._on_confirmed, 10)
        # CV state for idle display (not logged, just updates the current label)
        try:
            from boxbunny_msgs.msg import PunchDetection
            self.create_subscription(PunchDetection, "/boxbunny/cv/detection", self._on_cv_state, 10)
        except Exception:
            pass
        self.get_logger().info("Fusion monitor started")

    def _on_strike(self, msg):
        try:
            d = json.loads(msg.data)
            if self.cb_strike:
                self.cb_strike(d)
        except Exception:
            pass

    def _on_cv_state(self, msg):
        """Update the current CV prediction display (no logging)."""
        if self.cb_cv_state:
            self.cb_cv_state(msg.punch_type, msg.confidence)

    def _on_confirmed(self, msg):
        # Only show IMU-triggered events (the filtered predictions)
        if self.cb_confirmed and msg.imu_confirmed:
            self.cb_confirmed({
                "type": msg.punch_type, "pad": msg.pad,
                "cv_conf": msg.cv_confidence, "imu": msg.imu_confirmed,
                "cv": msg.cv_confirmed, "accel": msg.accel_magnitude,
            })


class FusionMonitor:
    def __init__(self, node):
        self._node = node
        self._strikes = 0
        self._confirmed = 0
        self._last_strike_per_pad: dict = {}  # pad -> timestamp for debounce

        self._root = tk.Tk()
        self._root.title("Fusion Monitor")
        self._root.configure(bg=BG)
        self._root.geometry("460x480")
        self._root.resizable(True, True)
        self._build()

        node.cb_strike = lambda d: self._root.after(0, lambda: self._on_strike(d))
        node.cb_confirmed = lambda d: self._root.after(0, lambda: self._on_confirmed(d))
        node.cb_cv_state = lambda a, c: self._root.after(0, lambda: self._on_cv_state(a, c))

        # Periodic check to reset display to IDLE
        self._last_confirmed_t = 0.0
        self._idle_check()

    def _build(self):
        r = self._root

        top = tk.Frame(r, bg=SURFACE, height=40)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="Fusion Monitor", font=(FONT, 14, "bold"),
                 bg=SURFACE, fg=PRIMARY).pack(side="left", padx=12)
        self._lbl_last = tk.Label(top, text="", font=(FONT, 13, "bold"),
                                  bg=SURFACE, fg=FG_DIM)
        self._lbl_last.pack(side="right", padx=12)

        # Stats
        stats = tk.Frame(r, bg=SURFACE2)
        stats.pack(fill="x", padx=8, pady=(6, 0))
        for label, color, key in [("IMU Strikes", AMBER, "pad"), ("Confirmed", GREEN, "ok")]:
            cell = tk.Frame(stats, bg=SURFACE2)
            cell.pack(side="left", expand=True, fill="x", padx=4, pady=6)
            tk.Label(cell, text=label, font=(FONT, 10), bg=SURFACE2, fg=FG_MUTED).pack()
            lbl = tk.Label(cell, text="0", font=(FONT, 20, "bold"), bg=SURFACE2, fg=color)
            lbl.pack()
            setattr(self, f"_stat_{key}", lbl)

        # CV state (what the model sees right now — shows idle)
        cv_row = tk.Frame(r, bg=BG)
        cv_row.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(cv_row, text="CV:", font=(FONT, 12, "bold"), bg=BG, fg=FG_MUTED).pack(side="left")
        self._lbl_cv_state = tk.Label(cv_row, text="idle", font=(FONT, 14, "bold"), bg=BG, fg=FG_MUTED)
        self._lbl_cv_state.pack(side="left", padx=(8, 0))
        self._lbl_cv_conf = tk.Label(cv_row, text="", font=(FONT_M, 11), bg=BG, fg=FG_DIM)
        self._lbl_cv_conf.pack(side="left", padx=(8, 0))

        # Last confirmed (big)
        tk.Label(r, text="LAST CONFIRMED", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=12, pady=(8, 0))
        self._lbl_result = tk.Label(r, text="IDLE", font=(FONT, 32, "bold"), bg=BG, fg=FG_MUTED)
        self._lbl_result.pack(pady=(2, 0))
        self._lbl_detail = tk.Label(r, text="",
                                    font=(FONT_M, 12), bg=BG, fg=FG_DIM)
        self._lbl_detail.pack(pady=(2, 0))

        # Log
        tk.Label(r, text="EVENTS", font=(FONT, 10, "bold"),
                 bg=BG, fg=FG_MUTED).pack(anchor="w", padx=12, pady=(12, 2))

        self._log = tk.Text(r, bg=SURFACE, fg=FG_DIM, font=(FONT_M, 11),
                            wrap="none", relief="flat", borderwidth=0,
                            highlightthickness=0, state="disabled")
        self._log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._log.tag_configure("strike", foreground=AMBER)
        self._log.tag_configure("confirmed", foreground=GREEN)

    def _add(self, text, tag):
        self._log.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}] {text}\n", tag)
        lines = int(self._log.index("end-1c").split(".")[0])
        if lines > 100:
            self._log.delete("1.0", f"{lines - 80}.0")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _on_cv_state(self, action, conf):
        """Update the live CV prediction display + detect sustained blocks."""
        color = PUNCH_COLORS.get(action, FG_MUTED)
        self._lbl_cv_state.configure(text=action.replace("_", " "), fg=color)
        self._lbl_cv_conf.configure(text=f"{conf:.0%}")

        # Track block frames
        if action == "block":
            self._block_frames = getattr(self, '_block_frames', 0) + 1
            # Show BLOCK in result if sustained for 10+ frames (~0.3s) and no recent strike
            last_strike_age = time.time() - getattr(self, '_last_strike_t', 0.0)
            if self._block_frames >= 10 and last_strike_age > 1.0:
                if not getattr(self, '_block_shown', False):
                    self._lbl_result.configure(text="BLOCK", fg=PUNCH_COLORS.get("block", FG_DIM))
                    self._lbl_detail.configure(text=f"CV-only  conf={conf:.0%}  ({self._block_frames} frames)")
                    self._block_shown = True
        else:
            if getattr(self, '_block_shown', False):
                # Block ended — go back to idle
                self._lbl_result.configure(text="IDLE", fg=FG_MUTED)
                self._lbl_detail.configure(text="")
                self._block_shown = False
            self._block_frames = 0

    def _idle_check(self):
        """Reset display to IDLE if no activity for 2 seconds."""
        now = time.time()
        if (now - self._last_confirmed_t) > 2.0 and not getattr(self, '_block_shown', False):
            current_text = self._lbl_result.cget("text")
            if current_text not in ("--", "IDLE", "BLOCK"):
                self._lbl_result.configure(text="IDLE", fg=FG_MUTED)
                self._lbl_detail.configure(text="")
                self._lbl_last.configure(text="", fg=FG_DIM)
        try:
            self._root.after(500, self._idle_check)
        except Exception:
            pass

    def _on_strike(self, data):
        pad_idx = data.get("pad_index", -1)
        pad = PAD_MAP.get(pad_idx, "?")
        accel = data.get("peak_accel", 0.0)

        # Debounce: skip if same pad struck within 300ms
        now = time.time()
        last = self._last_strike_per_pad.get(pad, 0.0)
        if now - last < 0.5:
            return
        self._last_strike_per_pad[pad] = now

        self._strikes += 1
        self._last_strike_t = now
        self._stat_pad.configure(text=str(self._strikes))
        expects = PAD_EXPECTS.get(pad, "?")
        self._add(f"STRIKE  {pad:<7s} {accel:.1f} m/s\u00B2  expects: {expects}", "strike")

    def _on_confirmed(self, d):
        ptype = d["type"]
        if ptype in ("idle", "unclassified"):
            return

        self._confirmed += 1
        self._last_confirmed_t = time.time()
        self._stat_ok.configure(text=str(self._confirmed))

        color = PUNCH_COLORS.get(ptype, FG)
        pad = d["pad"] or "--"
        accel = d["accel"]
        cv_conf = d["cv_conf"]
        imu = d["imu"]
        cv = d["cv"]

        if imu and cv:
            source = "CV+IMU"
        elif cv and not imu:
            source = "CV-only"
        elif imu and not cv:
            source = "IMU-only"
        else:
            source = "?"

        self._lbl_result.configure(
            text=ptype.replace("_", " ").upper(), fg=color,
        )
        parts = [source]
        if pad != "--": parts.append(f"pad={pad}")
        if accel > 0: parts.append(f"accel={accel:.0f}")
        parts.append(f"conf={cv_conf:.0%}")
        self._lbl_detail.configure(text="  ".join(parts))
        self._lbl_last.configure(text=ptype.replace("_", " "), fg=color)
        self._add(
            f"  \u2192  {ptype:<16s} {source:<8s} pad={pad:<7s} conf={cv_conf:.0%}",
            "confirmed",
        )

    def run(self):
        self._root.mainloop()


def main():
    rclpy.init()
    node = FusionNode()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    gui = FusionMonitor(node)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
