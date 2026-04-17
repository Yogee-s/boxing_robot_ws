"""Unified BLE bridge between the Jetson ROS graph and the Arduino base.

Maintains a single BLE connection to the "BoxBunny Base" Arduino (which runs
`ble_control.ino`) and multiplexes two command streams onto it:

* Height — subscribes to ``/boxbunny/robot/height`` (HeightCommand) and
  translates to ``HUP:<pwm>`` / ``HDOWN:<pwm>`` / ``HSTOP``. A 500 ms Jetson-
  side deadman re-issues ``HSTOP`` if the publisher goes silent mid-motion;
  the Arduino has its own 250 ms refresh-window watchdog as a second layer.

* Base tracking — when tracking is enabled via the ``tracking_start`` service,
  subscribes to ``/boxbunny/cv/person_direction`` + ``/boxbunny/cv/user_tracking``
  and emits ``L:<rpm>`` / ``R:<rpm>`` / ``S``. Near-range gating: the motor
  only rotates while the user is within the ``near_range_m`` depth threshold
  (default 0.8 m) with ``hysteresis_m`` (default 0.15 m) before disengaging.

BLE is a single-owner link, so this node owns it exclusively. It auto-
connects on startup and transparently reconnects if the link drops.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String as StdString
from std_srvs.srv import Trigger

from boxbunny_msgs.msg import HeightCommand, UserTracking

from boxbunny_core.constants import Topics


logger = logging.getLogger("boxbunny.ble_bridge")


# ── BLE constants (must match ble_control.ino) ───────────────────────────
DEVICE_NAME = "BoxBunny Base"
SERVICE_UUID = "00001820-0000-1000-8000-00805f9b34fb"
CMD_UUID = "00002a68-0000-1000-8000-00805f9b34fb"
FEEDBACK_UUID = "00002a69-0000-1000-8000-00805f9b34fb"

# ── Tunable defaults (overridden by config/boxbunny.yaml: ble_tracking) ──
DEFAULT_NEAR_RANGE_M = 0.8
DEFAULT_HYSTERESIS_M = 0.15
DEFAULT_TRACK_RPM = 500

# ── Height command PWM (matches robot_node's existing translation) ───────
HEIGHT_MANUAL_PWM = 200

# ── Jetson deadman (secondary to Arduino's 250 ms watchdog) ──────────────
HEIGHT_DEADMAN_S = 0.5


def _load_tracking_config() -> dict:
    """Load ble_tracking section from config/boxbunny.yaml."""
    ws_root = Path(__file__).resolve().parents[3]
    cfg_path = ws_root / "config" / "boxbunny.yaml"
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("ble_tracking", {}) or {}
    except FileNotFoundError:
        logger.warning("boxbunny.yaml not found at %s — using defaults", cfg_path)
        return {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse boxbunny.yaml: %s", exc)
        return {}


class BleBridgeNode(Node):
    """ROS 2 ↔ BLE bridge. Owns the single central connection to the Arduino."""

    def __init__(self) -> None:
        super().__init__("ble_bridge_node")

        cfg = _load_tracking_config()
        self._near_range_m: float = float(cfg.get("near_range_m", DEFAULT_NEAR_RANGE_M))
        self._hysteresis_m: float = float(cfg.get("hysteresis_m", DEFAULT_HYSTERESIS_M))
        self._track_rpm: int = int(cfg.get("rpm", DEFAULT_TRACK_RPM))

        # ── Shared state ────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._ble_connected: bool = False
        self._tracking_enabled: bool = False
        self._track_engaged: bool = False  # True once user is inside near range
        self._last_track_cmd: str = ""
        self._last_height_cmd_ts: float = 0.0
        self._height_active: bool = False
        self._last_cmd_sent: str = ""

        # Thread-safe command queue drained by the asyncio BLE thread.
        self._cmd_queue: "queue.Queue[str]" = queue.Queue()

        # Cache of the latest tracking state so the direction callback can
        # decide whether to emit a rotation command.
        self._user_detected: bool = False
        self._user_depth: float = 0.0
        self._last_direction: str = "centre"

        # ── ROS interfaces ──────────────────────────────────────────────
        self.create_subscription(
            HeightCommand, Topics.ROBOT_HEIGHT,
            self._on_height, 10,
        )
        self.create_subscription(
            StdString, Topics.CV_PERSON_DIRECTION,
            self._on_person_direction, 10,
        )
        self.create_subscription(
            UserTracking, Topics.CV_USER_TRACKING,
            self._on_user_tracking, 10,
        )

        self._status_pub = self.create_publisher(
            StdString, "/boxbunny/ble/status", 10,
        )

        self.create_service(
            Trigger, "/boxbunny/ble/tracking_start", self._srv_tracking_start,
        )
        self.create_service(
            Trigger, "/boxbunny/ble/tracking_stop", self._srv_tracking_stop,
        )

        # ── Timers ──────────────────────────────────────────────────────
        self.create_timer(0.1, self._deadman_tick)
        self.create_timer(1.0, self._publish_status)

        # ── BLE worker thread ───────────────────────────────────────────
        self._shutdown_event = threading.Event()
        self._ble_thread = threading.Thread(
            target=self._ble_thread_main, name="ble-bridge", daemon=True,
        )
        self._ble_thread.start()

        self.get_logger().info(
            f"BLE bridge started — near_range={self._near_range_m:.2f}m "
            f"hysteresis={self._hysteresis_m:.2f}m rpm={self._track_rpm}"
        )

    # ── Subscriber callbacks ────────────────────────────────────────────

    def _on_height(self, msg: HeightCommand) -> None:
        """Translate HeightCommand → BLE HUP/HDOWN/HSTOP."""
        action = msg.action
        cmd: Optional[str] = None
        if action == "stop":
            cmd = "HSTOP"
            with self._lock:
                self._height_active = False
        elif action == "manual_up":
            cmd = f"HUP:{HEIGHT_MANUAL_PWM}"
        elif action == "manual_down":
            cmd = f"HDOWN:{HEIGHT_MANUAL_PWM}"
        elif action == "adjust":
            error = msg.current_height_px - msg.target_height_px
            pwm = min(255, int(abs(error) * 2))
            direction = "HUP" if error > 0 else "HDOWN"
            cmd = f"{direction}:{pwm}"
        elif action == "calibrate":
            self.get_logger().debug("Height calibrate requested (no motion)")
            return
        else:
            return

        if cmd is None:
            return

        if cmd.startswith(("HUP", "HDOWN")):
            with self._lock:
                self._height_active = True
                self._last_height_cmd_ts = self.get_clock().now().nanoseconds / 1e9

        self._enqueue(cmd)

    def _on_person_direction(self, msg: StdString) -> None:
        """Update cached direction and re-evaluate tracking output."""
        self._last_direction = msg.data.strip().lower()
        self._reevaluate_tracking()

    def _on_user_tracking(self, msg: UserTracking) -> None:
        """Update cached depth/detection and re-evaluate tracking output."""
        self._user_detected = bool(msg.user_detected)
        if msg.user_detected and msg.depth > 0:
            self._user_depth = float(msg.depth)
        else:
            self._user_depth = 0.0
        self._reevaluate_tracking()

    # ── Service handlers ────────────────────────────────────────────────

    def _srv_tracking_start(
        self, _request: Trigger.Request, response: Trigger.Response,
    ) -> Trigger.Response:
        with self._lock:
            self._tracking_enabled = True
            self._track_engaged = False
            self._last_track_cmd = ""
        self.get_logger().info("Tracking ENABLED")
        response.success = True
        response.message = "Tracking started"
        return response

    def _srv_tracking_stop(
        self, _request: Trigger.Request, response: Trigger.Response,
    ) -> Trigger.Response:
        with self._lock:
            self._tracking_enabled = False
            self._track_engaged = False
            self._last_track_cmd = ""
        self._enqueue("S")
        self.get_logger().info("Tracking DISABLED — sent S")
        response.success = True
        response.message = "Tracking stopped"
        return response

    # ── Decision logic ──────────────────────────────────────────────────

    def _reevaluate_tracking(self) -> None:
        """Decide which tracking command to emit based on cached state."""
        if not self._tracking_enabled:
            return

        cmd = self._compute_tracking_cmd()
        if cmd != self._last_track_cmd:
            self._last_track_cmd = cmd
            self._enqueue(cmd)

    def _compute_tracking_cmd(self) -> str:
        """Return the tracking command for the current cached state."""
        if not self._user_detected or self._user_depth <= 0.0:
            self._track_engaged = False
            return "S"

        # Near-range gating with hysteresis.
        engage_limit = self._near_range_m
        disengage_limit = self._near_range_m + self._hysteresis_m

        if self._track_engaged:
            if self._user_depth > disengage_limit:
                self._track_engaged = False
                return "S"
        else:
            if self._user_depth > engage_limit:
                return "S"
            self._track_engaged = True

        direction = self._last_direction
        if direction == "left":
            return f"L:{self._track_rpm}"
        if direction == "right":
            return f"R:{self._track_rpm}"
        return "S"  # centre or unknown — hold still

    # ── Deadman + status ────────────────────────────────────────────────

    def _deadman_tick(self) -> None:
        """Zero the height motor if commands stop arriving."""
        with self._lock:
            if not self._height_active:
                return
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_height_cmd_ts <= HEIGHT_DEADMAN_S:
                return
            self._height_active = False
        self.get_logger().warning(
            f"Height deadman fired after {HEIGHT_DEADMAN_S:.2f}s — sending HSTOP"
        )
        self._enqueue("HSTOP")

    def _publish_status(self) -> None:
        status = {
            "connected": self._ble_connected,
            "tracking": self._tracking_enabled,
            "last_cmd": self._last_cmd_sent,
            "near_range_m": self._near_range_m,
        }
        msg = StdString()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

    def _enqueue(self, cmd: str) -> None:
        """Enqueue a BLE command for the async worker thread."""
        self._cmd_queue.put(cmd)

    # ── BLE worker thread ───────────────────────────────────────────────

    def _ble_thread_main(self) -> None:
        """Run the asyncio BLE loop until shutdown."""
        try:
            asyncio.run(self._ble_async_main())
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"BLE thread crashed: {exc}")

    async def _ble_async_main(self) -> None:
        """Outer reconnect loop: scan → connect → run → on disconnect repeat."""
        from bleak import BleakClient, BleakScanner
        from bleak.exc import BleakError

        backoff_s = 2.0
        while not self._shutdown_event.is_set():
            device = None
            try:
                self.get_logger().info(f"Scanning for '{DEVICE_NAME}'...")
                devices = await BleakScanner.discover(timeout=5.0)
                for d in devices:
                    if d.name and DEVICE_NAME in d.name:
                        device = d
                        break
            except BleakError as exc:
                self.get_logger().warning(f"BLE scan error: {exc}")

            if device is None:
                if self._shutdown_event.is_set():
                    break
                await asyncio.sleep(backoff_s)
                continue

            try:
                async with BleakClient(device.address, timeout=10.0) as client:
                    await self._run_connected(client)
            except BleakError as exc:
                self.get_logger().warning(f"BLE connection error: {exc}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Unexpected BLE error: {exc}")

            self._ble_connected = False
            # Force tracking off on disconnect — matches the GUI behaviour so
            # that after a reconnect the bridge and the UI agree on whether
            # tracking is active. User must press Start Tracking again.
            with self._lock:
                if self._tracking_enabled:
                    self.get_logger().info(
                        "BLE disconnected — forcing tracking off"
                    )
                self._tracking_enabled = False
                self._track_engaged = False
                self._last_track_cmd = ""
                self._height_active = False
            if not self._shutdown_event.is_set():
                await asyncio.sleep(backoff_s)

    async def _run_connected(self, client) -> None:
        """Drain the command queue while the BLE client is connected."""
        # Drop anything accumulated while we were disconnected — otherwise
        # a backlog of stale HUP/HDOWN commands would flood the motor on
        # reconnect even if the button was released minutes ago.
        self._drain_queue()
        self._ble_connected = True
        self.get_logger().info(f"BLE connected to {DEVICE_NAME}")

        try:
            await client.start_notify(FEEDBACK_UUID, self._on_feedback)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(f"Feedback notify not available: {exc}")

        while not self._shutdown_event.is_set() and client.is_connected:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue

            try:
                await client.write_gatt_char(CMD_UUID, cmd.encode("utf-8"))
                self._last_cmd_sent = cmd
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"BLE write failed ({cmd}): {exc}")
                break

            await asyncio.sleep(0.05)

        # Best-effort safe-stop before releasing the link.
        try:
            await client.write_gatt_char(CMD_UUID, b"HSTOP")
            await client.write_gatt_char(CMD_UUID, b"S")
        except Exception:  # noqa: BLE001
            pass

    def _drain_queue(self) -> None:
        """Empty the command queue. Called on reconnect to drop stale cmds."""
        dropped = 0
        while True:
            try:
                self._cmd_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            self.get_logger().info(
                f"Dropped {dropped} stale commands on reconnect"
            )

    def _on_feedback(self, _sender, data: bytearray) -> None:
        """Arduino → Jetson telemetry (currently logged-only)."""
        try:
            text = data.decode("utf-8", errors="ignore").strip()
            if text:
                self.get_logger().debug(f"BLE fb: {text}")
        except (UnicodeDecodeError, ValueError):
            pass

    # ── Shutdown ────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Signal the BLE thread to exit and queue a final stop."""
        with self._lock:
            self._tracking_enabled = False
            self._height_active = False
        self._enqueue("HSTOP")
        self._enqueue("S")
        self._shutdown_event.set()
        self._ble_thread.join(timeout=3.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = BleBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
