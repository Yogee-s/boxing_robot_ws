"""IMU UDP Bridge -- reads motor_feedback from micro-ROS Teensy and forwards via UDP.

Subscribes to the ``motor_feedback`` Float64MultiArray topic published by the
Teensy firmware (V4) through the micro-ROS agent.  Extracts raw IMU
accelerometer data from indices [9..20] (4 IMUs x 3 axes, m/s²).

Uses the same calibration approach as ``Boxing_Arm_Control/unified_GUI_V3.py``:
first 500 samples (~2.5s at 200Hz) are averaged per-IMU to compute a gravity
offset vector, then ``clean = raw - offset`` gives the dynamic component whose
magnitude is compared against the strike threshold.

Detected impacts are sent as JSON over UDP for ``cv_imu_fusion_test.py``.

Run in a terminal WITHOUT conda, with ROS 2 sourced::

    source /opt/ros/humble/setup.bash
    python3 notebooks/scripts/imu_udp_bridge.py
"""
from __future__ import annotations

import json
import logging
import math
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_WS_ROOT = Path(__file__).resolve().parents[2]
_CFG: dict = {}
try:
    import yaml
    _cfg_path = _WS_ROOT / "config" / "boxbunny.yaml"
    if _cfg_path.exists():
        with open(_cfg_path) as _f:
            _CFG = yaml.safe_load(_f) or {}
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("imu_bridge")

UDP_HOST = "127.0.0.1"
UDP_PORT = 9876
N_IMUS = 4
CALIB_SAMPLES = 500  # ~2.5s at 200Hz, matches unified_GUI_V3

# Strike threshold after calibration (m/s²).  V3 default = 5.0
IMPACT_THRESHOLD = float(
    _CFG.get("fusion", {}).get("imu_impact_threshold", 5.0)
)
DEBOUNCE_S = float(
    _CFG.get("fusion", {}).get("imu_debounce_ms", 150)
) / 1000.0

# Pad mapping from config (matches unified_GUI_V3: 0=Centre, 1=Left, 2=Right)
_DEFAULT_PAD_MAP = {0: "centre", 1: "left", 2: "right", 3: "head"}
IMU_PAD_MAP: Dict[int, str] = {
    int(k): v
    for k, v in _CFG.get("fusion", {}).get("imu_pad_map", _DEFAULT_PAD_MAP).items()
}


class IMUBridgeNode(Node):
    """Bridges Teensy IMU data to UDP with auto-calibration."""

    def __init__(
        self,
        udp_host: str = UDP_HOST,
        udp_port: int = UDP_PORT,
        threshold: float = IMPACT_THRESHOLD,
    ) -> None:
        super().__init__("imu_udp_bridge")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (udp_host, udp_port)
        self._threshold = threshold
        self._last_impact: Dict[int, float] = {}
        self._msg_count = 0

        # Calibration state (mirrors unified_GUI_V3 approach)
        self._calibrated = False
        self._calib_samples = 0
        self._calib_sum: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(N_IMUS)]
        self._offset: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(N_IMUS)]

        self.create_subscription(
            Float64MultiArray, "motor_feedback", self._on_feedback, 10,
        )
        self.create_timer(5.0, self._heartbeat)

        logger.info(
            "IMU bridge started -> UDP %s:%d  (threshold=%.1f m/s², calibrating %d samples...)",
            udp_host, udp_port, threshold, CALIB_SAMPLES,
        )

    def _on_feedback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 21:
            return

        self._msg_count += 1
        now = time.time()
        imus = []
        impacts = []

        for i in range(N_IMUS):
            base = 9 + i * 3
            ax = msg.data[base]
            ay = msg.data[base + 1]
            az = msg.data[base + 2]

            if not self._calibrated:
                # Accumulate for gravity offset
                self._calib_sum[i][0] += ax
                self._calib_sum[i][1] += ay
                self._calib_sum[i][2] += az
            else:
                # Subtract per-IMU gravity offset vector (same as V3)
                cx = ax - self._offset[i][0]
                cy = ay - self._offset[i][1]
                cz = az - self._offset[i][2]
                dynamic = math.sqrt(cx * cx + cy * cy + cz * cz)

                pad_name = IMU_PAD_MAP.get(i, f"imu{i}")
                imus.append({
                    "index": i,
                    "pad": pad_name,
                    "accel": [round(ax, 2), round(ay, 2), round(az, 2)],
                    "dynamic": round(dynamic, 2),
                })

                if dynamic > self._threshold:
                    last = self._last_impact.get(i, 0.0)
                    if now - last > DEBOUNCE_S:
                        self._last_impact[i] = now
                        impacts.append({
                            "timestamp": now,
                            "imu_index": i,
                            "pad": pad_name,
                            "magnitude": round(dynamic, 2),
                        })
                        logger.info(
                            "IMPACT [%s] (IMU %d): %.1f m/s²",
                            pad_name.upper(), i, dynamic,
                        )

        # Finish calibration after enough samples
        if not self._calibrated:
            self._calib_samples += 1
            if self._calib_samples >= CALIB_SAMPLES:
                for i in range(N_IMUS):
                    self._offset[i] = [
                        self._calib_sum[i][j] / self._calib_samples
                        for j in range(3)
                    ]
                self._calibrated = True
                logger.info(
                    "Calibration complete (%d samples). Offsets:",
                    self._calib_samples,
                )
                for i in range(N_IMUS):
                    pad = IMU_PAD_MAP.get(i, f"imu{i}")
                    logger.info(
                        "  IMU %d [%s]: offset=[%.2f, %.2f, %.2f]",
                        i, pad, *self._offset[i],
                    )
            return  # Don't send data during calibration

        packet = json.dumps({
            "timestamp": now,
            "calibrated": self._calibrated,
            "imus": imus,
            "impacts": impacts,
        }).encode()

        try:
            self._sock.sendto(packet, self._target)
        except OSError as exc:
            logger.debug("UDP send error: %s", exc)

    def _heartbeat(self) -> None:
        status = "calibrated" if self._calibrated else f"calibrating ({self._calib_samples}/{CALIB_SAMPLES})"
        logger.info(
            "Heartbeat: %d msgs, %s, threshold=%.1f m/s²",
            self._msg_count, status, self._threshold,
        )


def main() -> None:
    """Entry point."""
    rclpy.init()
    node = IMUBridgeNode()

    def _shutdown(signum: int, frame: object) -> None:
        logger.info("Shutting down IMU bridge")
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
