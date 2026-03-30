"""CV + IMU Fusion Test -- action prediction with IMU impact filtering.

Runs the action prediction CV model on RealSense camera frames while
receiving IMU impact data via UDP from ``imu_udp_bridge.py``.  Fuses both
signals within a configurable time window to separate true punches
(confirmed by both CV and IMU) from false positives (CV-only).

The display shows:
- Camera feed with the CV prediction and confidence
- IMU acceleration bars per sensor with impact indicators
- Running fusion statistics (confirmed, CV-only, IMU-only)
- Recent event log colour-coded by fusion status

Run in the conda/torch terminal::

    conda activate boxing_ai
    python3 notebooks/scripts/cv_imu_fusion_test.py [--show-video]
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WS_ROOT = Path(__file__).resolve().parents[2]
_AP_ROOT = _WS_ROOT / "action_prediction"
sys.path.insert(0, str(_WS_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("cv_imu_test")

# ---------------------------------------------------------------------------
# Config from boxbunny.yaml
# ---------------------------------------------------------------------------
_CFG: dict = {}
try:
    import yaml
    _cfg_path = _WS_ROOT / "config" / "boxbunny.yaml"
    if _cfg_path.exists():
        with open(_cfg_path) as f:
            _CFG = yaml.safe_load(f) or {}
except ImportError:
    pass

FUSION_WINDOW_S: float = float(
    _CFG.get("fusion", {}).get("fusion_window_ms", 200)
) / 1000.0
CV_PENALTY: float = float(
    _CFG.get("fusion", {}).get("cv_unconfirmed_confidence_penalty", 0.3)
)
MIN_CV_CONF: float = float(
    _CFG.get("cv", {}).get("min_confidence", 0.4)
)
IMPACT_THRESHOLD: float = float(
    _CFG.get("fusion", {}).get("imu_impact_threshold", 15.0)
)
IMU_UDP_PORT = 9876
IDLE_ACTIONS = {"idle", "block"}

# IMU index -> pad name mapping (depends on physical wiring, adjust as needed)
# Override in config/boxbunny.yaml under fusion.imu_pad_map
_DEFAULT_PAD_MAP = {0: "left", 1: "centre", 2: "right", 3: "head"}
IMU_PAD_MAP: Dict[int, str] = {
    int(k): v
    for k, v in _CFG.get("fusion", {}).get("imu_pad_map", _DEFAULT_PAD_MAP).items()
}
PAD_NAMES = ["head", "left", "centre", "right"]

# Pad model colours (BGR)
_PAD_REST = (18, 22, 42)       # dark crimson
_PAD_BORDER = (48, 32, 74)     # muted border
_ARM_REST = (46, 26, 18)       # dark navy
_ARM_BORDER = (80, 48, 30)
_HIT_LIGHT = (113, 204, 46)    # green
_HIT_MEDIUM = (18, 156, 243)   # amber
_HIT_HARD = (60, 76, 231)      # red
_FLASH_DURATION = 0.30         # seconds


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IMUEvent:
    timestamp: float
    imu_index: int
    magnitude: float
    accel: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FusionStats:
    cv_total: int = 0
    imu_total: int = 0
    confirmed: int = 0
    cv_only: int = 0
    imu_only: int = 0
    events: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IMU receiver thread
# ---------------------------------------------------------------------------

class IMUReceiver(threading.Thread):
    """Receives IMU data via UDP from imu_udp_bridge.py."""

    def __init__(self, port: int = IMU_UDP_PORT) -> None:
        super().__init__(daemon=True)
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(0.1)
        self._running = True
        self._lock = threading.Lock()
        self._impacts: deque = deque(maxlen=200)
        self._latest_dynamic: Dict[int, float] = {}
        self._calibrated = False
        self._connected = False
        self._last_recv = 0.0

    def run(self) -> None:
        logger.info("IMU receiver listening on UDP :%d", self._port)
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                msg = json.loads(data.decode())
                self._last_recv = time.time()
                self._connected = True

                with self._lock:
                    self._calibrated = msg.get("calibrated", False)
                    for imu in msg.get("imus", []):
                        idx = imu["index"]
                        self._latest_dynamic[idx] = imu.get("dynamic", 0.0)

                    for impact in msg.get("impacts", []):
                        self._impacts.append(IMUEvent(
                            timestamp=impact["timestamp"],
                            imu_index=impact["imu_index"],
                            magnitude=impact["magnitude"],
                            accel=tuple(impact.get("accel", (0, 0, 0))),
                        ))

            except socket.timeout:
                if self._connected and time.time() - self._last_recv > 3.0:
                    self._connected = False
            except Exception as exc:
                logger.debug("IMU recv error: %s", exc)

    def pop_recent_impacts(self, since: float) -> List[IMUEvent]:
        """Return and remove impacts with timestamp >= *since*."""
        with self._lock:
            result = [e for e in self._impacts if e.timestamp >= since]
            # Remove consumed
            self._impacts = deque(
                (e for e in self._impacts if e.timestamp < since),
                maxlen=200,
            )
            return result

    def get_latest_dynamic(self) -> Dict[int, float]:
        """Return dynamic acceleration dict keyed by IMU index."""
        with self._lock:
            return dict(self._latest_dynamic)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._calibrated

    def stop(self) -> None:
        self._running = False
        self._sock.close()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
IMU_LABELS = ["IMU-0", "IMU-1", "IMU-2", "IMU-3"]


def _rounded_rect(
    img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
    color: Tuple[int, int, int], thickness: int = -1, radius: int = 8,
) -> None:
    """Draw a rounded rectangle (filled if thickness == -1)."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _hit_color(dynamic: float, threshold: float) -> Tuple[int, int, int]:
    """Return flash colour based on impact intensity."""
    if dynamic < threshold:
        return _PAD_REST
    ratio = min(1.0, dynamic / (threshold * 3))
    if ratio < 0.4:
        return _HIT_LIGHT
    if ratio < 0.7:
        return _HIT_MEDIUM
    return _HIT_HARD


def draw_pad_model(
    frame: np.ndarray,
    x: int,
    y: int,
    dynamics: Dict[int, float],
    pad_hit_times: Dict[str, float],
    connected: bool,
    threshold: float,
    now: float,
    calibrated: bool = False,
) -> None:
    """Draw the robot pad model -- HEAD on top, LEFT/CENTRE/RIGHT below, arms on sides.

    Pads flash with impact colour when hit and fade back to rest colour.
    """
    # Layout constants
    pad_w, pad_h = 56, 44
    gap = 4
    arm_w, arm_h = 28, pad_h * 2 + gap + pad_h + gap
    total_w = arm_w + gap + pad_w * 3 + gap * 2 + gap + arm_w
    total_h = pad_h + gap + pad_h + 26  # +26 for title + status
    panel_w = total_w + 20
    panel_h = total_h + 20

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (60, 60, 60), 1)

    # Title + connection/calibration status
    cv2.putText(frame, "IMU PADS", (x + 8, y + 16), _FONT, 0.42, (220, 220, 220), 1)
    if not connected:
        status_str, status_col = "WAITING...", (0, 100, 200)
    elif not calibrated:
        status_str, status_col = "CALIBRATING", (0, 200, 255)
    else:
        status_str, status_col = "READY", (0, 200, 0)
    cv2.putText(frame, status_str, (x + 90, y + 16), _FONT, 0.35, status_col, 1)

    # Origin for the pad grid (after title)
    ox = x + 10 + arm_w + gap
    oy = y + 24

    def _pad_colour(pad_name: str) -> Tuple[int, int, int]:
        """Compute colour: flash if recently hit, else resting."""
        hit_t = pad_hit_times.get(pad_name, 0.0)
        age = now - hit_t
        if age < _FLASH_DURATION:
            # Find the IMU index for this pad to get dynamic value
            for imu_idx, pname in IMU_PAD_MAP.items():
                if pname == pad_name:
                    dyn = dynamics.get(imu_idx, 0.0)
                    flash = _hit_color(dyn, threshold)
                    # Fade: interpolate back to rest
                    t = age / _FLASH_DURATION
                    return tuple(
                        int(flash[c] * (1 - t) + _PAD_REST[c] * t) for c in range(3)
                    )
            # If not mapped, use generic red flash
            t = age / _FLASH_DURATION
            return tuple(
                int(_HIT_HARD[c] * (1 - t) + _PAD_REST[c] * t) for c in range(3)
            )
        return _PAD_REST

    # HEAD pad (centred above the 3-pad row)
    head_x = ox + pad_w + gap // 2 - pad_w // 2 + gap // 2
    head_col = _pad_colour("head")
    _rounded_rect(frame, (head_x, oy), (head_x + pad_w, oy + pad_h), head_col, -1, 6)
    _rounded_rect(frame, (head_x, oy), (head_x + pad_w, oy + pad_h), _PAD_BORDER, 1, 6)
    cv2.putText(frame, "HEAD", (head_x + 8, oy + pad_h // 2 + 5),
                _FONT, 0.38, (230, 130, 130), 1)

    # LEFT / CENTRE / RIGHT pads
    row_y = oy + pad_h + gap
    for i, pname in enumerate(["left", "centre", "right"]):
        px = ox + i * (pad_w + gap)
        col = _pad_colour(pname)
        _rounded_rect(frame, (px, row_y), (px + pad_w, row_y + pad_h), col, -1, 6)
        _rounded_rect(frame, (px, row_y), (px + pad_w, row_y + pad_h), _PAD_BORDER, 1, 6)
        label = pname.upper()
        lw = cv2.getTextSize(label, _FONT, 0.33, 1)[0][0]
        cv2.putText(frame, label, (px + (pad_w - lw) // 2, row_y + pad_h // 2 + 5),
                    _FONT, 0.33, (230, 130, 130), 1)

    # Left arm
    la_x = x + 10
    la_col = _pad_colour("left_arm") if "left_arm" in pad_hit_times else _ARM_REST
    arm_top = oy
    arm_bot = row_y + pad_h
    _rounded_rect(frame, (la_x, arm_top), (la_x + arm_w, arm_bot), la_col, -1, 6)
    _rounded_rect(frame, (la_x, arm_top), (la_x + arm_w, arm_bot), _ARM_BORDER, 1, 6)
    cv2.putText(frame, "L", (la_x + 9, (arm_top + arm_bot) // 2 + 5),
                _FONT, 0.45, (110, 184, 220), 1)

    # Right arm
    ra_x = ox + 3 * (pad_w + gap)
    _rounded_rect(frame, (ra_x, arm_top), (ra_x + arm_w, arm_bot), _ARM_REST, -1, 6)
    _rounded_rect(frame, (ra_x, arm_top), (ra_x + arm_w, arm_bot), _ARM_BORDER, 1, 6)
    cv2.putText(frame, "R", (ra_x + 9, (arm_top + arm_bot) // 2 + 5),
                _FONT, 0.45, (110, 184, 220), 1)

    # Per-pad dynamic readout under the pads
    bar_y = row_y + pad_h + 4
    for i, pname in enumerate(["left", "centre", "right"]):
        px = ox + i * (pad_w + gap)
        dyn = 0.0
        for imu_idx, mapped in IMU_PAD_MAP.items():
            if mapped == pname:
                dyn = dynamics.get(imu_idx, 0.0)
                break
        col = (0, 0, 255) if dyn > threshold else (100, 100, 100)
        cv2.putText(frame, f"{dyn:.0f}", (px + 14, bar_y + 12), _FONT, 0.3, col, 1)

    # Head readout
    dyn_head = 0.0
    for imu_idx, mapped in IMU_PAD_MAP.items():
        if mapped == "head":
            dyn_head = dynamics.get(imu_idx, 0.0)
            break
    col = (0, 0, 255) if dyn_head > threshold else (100, 100, 100)
    cv2.putText(frame, f"{dyn_head:.0f}", (head_x + 18, oy - 4), _FONT, 0.3, col, 1)


def draw_imu_bars(
    frame: np.ndarray,
    x: int,
    y: int,
    dynamics: Dict[int, float],
    threshold: float,
) -> None:
    """Draw compact IMU acceleration bars (below the pad model)."""
    for i in range(4):
        by = y + i * 20
        dyn = dynamics.get(i, 0.0)
        pad_name = IMU_PAD_MAP.get(i, f"imu{i}")

        cv2.putText(frame, pad_name[:5].upper(), (x, by + 12),
                    _FONT, 0.3, (150, 150, 150), 1)

        bx, bw, bh = x + 48, 80, 12
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        fill = min(bw, int(dyn / 50.0 * bw))
        is_hit = dyn > threshold
        fill_col = (0, 0, 255) if is_hit else (0, 110, 0)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), fill_col, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (55, 55, 55), 1)
        cv2.putText(frame, f"{dyn:.0f}", (bx + bw + 3, by + 10),
                    _FONT, 0.27, (130, 130, 130), 1)


def draw_stats_panel(
    frame: np.ndarray, x: int, y: int, stats: FusionStats,
) -> None:
    """Draw fusion statistics panel."""
    pw, ph = 260, 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), (80, 80, 80), 1)

    cv2.putText(frame, "FUSION STATS", (x + 8, y + 20),
                _FONT, 0.5, (255, 255, 255), 1)

    rows = [
        (f"CV Detections:  {stats.cv_total}", (200, 200, 200)),
        (f"IMU Impacts:    {stats.imu_total}", (200, 200, 200)),
        (f"Confirmed:      {stats.confirmed}", (0, 255, 100)),
        (f"CV-Only (FP?):  {stats.cv_only}", (0, 100, 255)),
        (f"IMU-Only:       {stats.imu_only}", (255, 200, 0)),
    ]
    for i, (text, col) in enumerate(rows):
        cv2.putText(frame, text, (x + 12, y + 44 + i * 22),
                    _FONT, 0.4, col, 1)

    total_fused = stats.confirmed + stats.cv_only
    if total_fused > 0:
        prec = stats.confirmed / total_fused * 100
        cv2.putText(frame, f"CV Precision: {prec:.0f}%", (x + 12, y + 158),
                    _FONT, 0.45, (0, 255, 200), 1)


def draw_event_log(
    frame: np.ndarray, x: int, y: int, events: List[dict],
) -> None:
    """Draw the last N fusion events."""
    pw, ph = 320, 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), (80, 80, 80), 1)

    cv2.putText(frame, "RECENT EVENTS", (x + 8, y + 20),
                _FONT, 0.5, (255, 255, 255), 1)

    recent = events[-8:]
    for i, evt in enumerate(reversed(recent)):
        ey = y + 40 + i * 19
        status = evt["status"]
        action = evt.get("action", "?").upper()
        conf = evt.get("confidence", 0)

        if status == "CONFIRMED":
            col = (0, 255, 100)
            mag = evt.get("imu_mag", 0)
            text = f"[OK]  {action} ({conf:.0%}) imu={mag:.1f}"
        elif status == "CV_ONLY":
            col = (0, 100, 255)
            text = f"[FP?] {action} ({conf:.0%}) no imu"
        else:
            col = (255, 200, 0)
            text = f"[IMU] impact imu={evt.get('imu_mag', 0):.1f}"

        cv2.putText(frame, text, (x + 10, ey), _FONT, 0.33, col, 1)


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class CVIMUFusionTest:
    """Runs CV inference with IMU filtering and renders a combined display."""

    def __init__(
        self,
        checkpoint_path: str,
        pose_weights: str,
        device: str = "cuda:0",
        impact_threshold: float = IMPACT_THRESHOLD,
        imu_port: int = IMU_UDP_PORT,
        show_video: bool = True,
    ) -> None:
        self._checkpoint = checkpoint_path
        self._pose_weights = pose_weights
        self._device = device
        self._impact_threshold = impact_threshold
        self._show_video = show_video
        self._stats = FusionStats()
        self._imu = IMUReceiver(port=imu_port)
        self._last_offensive_time = 0.0
        self._pad_hit_times: Dict[str, float] = {}  # pad_name -> last hit timestamp

    def _init_engine(self) -> object:
        from action_prediction.lib.inference_runtime import InferenceEngine
        logger.info("Loading CV model: %s", self._checkpoint)
        engine = InferenceEngine(
            checkpoint_path=self._checkpoint,
            device=self._device,
        )
        engine.initialize()
        logger.info("CV model loaded")
        return engine

    def _try_fuse(self, action: str, confidence: float, now: float) -> str:
        """Attempt to fuse a CV detection with an IMU impact."""
        self._stats.cv_total += 1
        impacts = self._imu.pop_recent_impacts(now - FUSION_WINDOW_S)

        if impacts:
            best = max(impacts, key=lambda e: e.magnitude)
            self._stats.confirmed += 1
            self._stats.imu_total += len(impacts)
            self._stats.events.append({
                "time": now, "action": action, "confidence": confidence,
                "imu_mag": best.magnitude, "status": "CONFIRMED",
            })
            # Record hit for pad model flash
            pad_name = IMU_PAD_MAP.get(best.imu_index)
            if pad_name:
                self._pad_hit_times[pad_name] = now
            return "CONFIRMED"

        self._stats.cv_only += 1
        self._stats.events.append({
            "time": now, "action": action,
            "confidence": max(0.0, confidence - CV_PENALTY),
            "status": "CV_ONLY",
        })
        return "CV_ONLY"

    def run(self) -> None:
        """Main loop: capture -> infer -> fuse -> display."""
        import pyrealsense2 as rs

        self._imu.start()
        engine = self._init_engine()

        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        rs_cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        align = rs.align(rs.stream.color)

        logger.info("Starting RealSense pipeline...")
        pipeline.start(rs_cfg)
        logger.info("Camera started. Press 'q' to quit.")

        last_action, last_conf = "idle", 0.0
        last_status = ""
        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                now = time.time()

                # CV inference
                result = engine.process_frame(rgb, depth)
                if result is not None:
                    last_action = result.action
                    last_conf = result.confidence

                    if (last_action not in IDLE_ACTIONS
                            and last_conf >= MIN_CV_CONF):
                        # Debounce: only fuse if >200ms since last offensive
                        if now - self._last_offensive_time > FUSION_WINDOW_S:
                            last_status = self._try_fuse(
                                last_action, last_conf, now,
                            )
                            self._last_offensive_time = now

                # Check for IMU-only impacts (no CV match)
                stale = self._imu.pop_recent_impacts(0.0)
                if stale:
                    self._stats.imu_total += len(stale)
                    for imp in stale:
                        # Record hit for pad model flash regardless
                        pad_name = IMU_PAD_MAP.get(imp.imu_index)
                        if pad_name:
                            self._pad_hit_times[pad_name] = imp.timestamp
                        age = now - imp.timestamp
                        if age > FUSION_WINDOW_S * 2:
                            self._stats.imu_only += 1
                            self._stats.events.append({
                                "time": imp.timestamp,
                                "imu_mag": imp.magnitude,
                                "status": "IMU_ONLY",
                            })

                # FPS
                frame_count += 1
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_timer = now

                # Display
                if self._show_video:
                    display = self._draw(
                        rgb, last_action, last_conf, last_status, fps, now,
                    )
                    cv2.imshow("CV + IMU Fusion Test", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                else:
                    if frame_count == 0:
                        self._log_stats()

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            pipeline.stop()
            self._imu.stop()
            cv2.destroyAllWindows()
            self._print_summary()

    def _draw(
        self,
        rgb: np.ndarray,
        action: str,
        confidence: float,
        status: str,
        fps: float,
        now: float,
    ) -> np.ndarray:
        display = rgb.copy()
        h, w = display.shape[:2]

        # Prediction box (top-left)
        if action in IDLE_ACTIONS:
            pred_col = (200, 200, 200)
        elif status == "CONFIRMED":
            pred_col = (0, 255, 100)
        else:
            pred_col = (0, 165, 255)

        cv2.rectangle(display, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (250, 80), (100, 100, 100), 1)
        cv2.putText(display, action.upper(), (20, 42),
                    _FONT, 0.8, pred_col, 2)
        cv2.putText(display, f"{confidence * 100:.0f}%", (20, 68),
                    _FONT, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"{fps:.0f} FPS", (170, 68),
                    _FONT, 0.4, (150, 150, 150), 1)
        if status and action not in IDLE_ACTIONS:
            scol = (0, 255, 100) if status == "CONFIRMED" else (0, 100, 255)
            cv2.putText(display, status, (170, 42),
                        _FONT, 0.4, scol, 1)

        # IMU pad model (top-right)
        dyns = self._imu.get_latest_dynamic()
        # Update pad hit times from current dynamics (real-time flash)
        for imu_idx, dyn in dyns.items():
            if dyn > self._impact_threshold:
                pname = IMU_PAD_MAP.get(imu_idx)
                if pname:
                    self._pad_hit_times[pname] = now
        draw_pad_model(
            display, w - 250, 10, dyns, self._pad_hit_times,
            self._imu.connected, self._impact_threshold, now,
            self._imu.calibrated,
        )
        # IMU acceleration bars (below pad model)
        draw_imu_bars(display, w - 245, 155, dyns, self._impact_threshold)

        # Stats panel (bottom-left)
        draw_stats_panel(display, 10, h - 180, self._stats)

        # Event log (bottom-right)
        draw_event_log(display, w - 330, h - 200, self._stats.events)

        return display

    def _log_stats(self) -> None:
        s = self._stats
        logger.info(
            "CV=%d  IMU=%d  confirmed=%d  cv_only=%d  imu_only=%d",
            s.cv_total, s.imu_total, s.confirmed, s.cv_only, s.imu_only,
        )

    def _print_summary(self) -> None:
        s = self._stats
        logger.info("=" * 50)
        logger.info("CV + IMU FUSION TEST RESULTS")
        logger.info("=" * 50)
        logger.info("CV Detections:     %d", s.cv_total)
        logger.info("IMU Impacts:       %d", s.imu_total)
        logger.info("Confirmed (both):  %d", s.confirmed)
        logger.info("CV-Only (FP?):     %d", s.cv_only)
        logger.info("IMU-Only:          %d", s.imu_only)
        if s.cv_total > 0:
            precision = s.confirmed / s.cv_total * 100
            logger.info("CV Precision:      %.1f%%", precision)
        logger.info("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CV + IMU Fusion Test")
    parser.add_argument(
        "--checkpoint",
        default=str(_AP_ROOT / "model" / "best_model.pth"),
    )
    parser.add_argument(
        "--pose-weights",
        default=str(_AP_ROOT / "model" / "yolo26n-pose.pt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--impact-threshold", type=float, default=IMPACT_THRESHOLD,
        help="Dynamic accel threshold (m/s^2) for impact. Default from config.",
    )
    parser.add_argument("--imu-port", type=int, default=IMU_UDP_PORT)
    parser.add_argument(
        "--show-video", action="store_true", default=True,
        help="Show OpenCV display window (default: on).",
    )
    parser.add_argument(
        "--no-video", action="store_false", dest="show_video",
        help="Disable display for headless testing.",
    )
    args = parser.parse_args()

    test = CVIMUFusionTest(
        checkpoint_path=args.checkpoint,
        pose_weights=args.pose_weights,
        device=args.device,
        impact_threshold=args.impact_threshold,
        imu_port=args.imu_port,
        show_video=args.show_video,
    )
    test.run()


if __name__ == "__main__":
    main()
