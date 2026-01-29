import json
import os
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu
from boxbunny_msgs.msg import ImuPunch
from boxbunny_msgs.srv import CalibrateImuPunch


PUNCH_TYPES = ("straight", "hook", "uppercut")
PUNCH_TYPE_ALIASES = {"jab_or_cross": "straight"}


class ImuPunchClassifier(Node):
    def __init__(self) -> None:
        super().__init__("imu_punch_classifier")

        self.declare_parameter("enable_punch_classification", True)
        self.declare_parameter("window_size", 10)
        self.declare_parameter("cooldown_s", 0.5)
        self.declare_parameter("gyro_threshold", 1.5)
        self.declare_parameter("accel_threshold", 3.0)
        self.declare_parameter("accel_peak_ratio", 1.05)
        self.declare_parameter("gyro_peak_ratio", 1.05)
        self.declare_parameter("axis_dominance_ratio", 1.0)
        self.declare_parameter("imu_hand", "right")
        self.declare_parameter("calibration_path", os.path.expanduser("~/.boxbunny/imu_calibration.json"))
        self.declare_parameter("use_calibration", True)

        self.sub = self.create_subscription(Imu, "imu/data", self._on_imu, 10)
        self.pub = self.create_publisher(ImuPunch, "imu/punch", 10)
        self.calib_srv = self.create_service(CalibrateImuPunch, "calibrate_imu_punch", self._on_calibrate)

        self._history: Deque[Imu] = deque(maxlen=int(self.get_parameter("window_size").value))
        self._last_time = 0.0
        self._calibrating: Optional[str] = None
        self._calibration_end = 0.0
        self._calibration_peaks: Dict[str, Tuple[float, float]] = {}
        
        self._first_strike_accel = None # Baseline from first strike of a sequence
        self._last_calib_type = None
        self._prev_accel = 9.8  # For jerk detection (rate of change)

        self._waiting_for_quiet = False
        self._waiting_for_trigger = False
        self._templates = self._load_templates()
        self._calib_timer = self.create_timer(0.05, self._calibration_tick)
        self._loading_defaults = False
        
        # Gravity Calibration
        self._gravity_samples = []
        self._gravity_vector = np.array([0.0, 9.8, 0.0]) # Default
        self._calibrated_gravity = False
        
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.get_logger().info("IMU punch classifier initialized")
        
    def _on_params_changed(self, params):
        changed = False
        for param in params:
            if param.name == "calibration_path":
                self.get_logger().info(f"Reloading calibration from: {param.value}")
                try:
                    self._templates = self._load_templates(path_override=str(param.value))
                except Exception as e:
                    self.get_logger().error(f"Failed to reload templates: {e}")
            elif param.name in ["accel_threshold", "gyro_threshold"]:
                changed = True
                
        # If thresholds changed and we aren't currently loading them from file, save new defaults
        if changed and not self._loading_defaults:
            if "settings" not in self._templates:
                self._templates["settings"] = {}
                
            # Note: param values here are the NEW values
            for param in params:
                if param.name in ["accel_threshold", "gyro_threshold"]:
                    self._templates["settings"][param.name] = float(param.value)
            
            self._save_templates()
            
        return SetParametersResult(successful=True)

    def _load_templates(self, path_override: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        if path_override:
            path = path_override
        else:
            path = self.get_parameter("calibration_path").value
            
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            
            templates = {}
            settings = raw.get("settings", {})
            
            # Apply settings if present
            if settings:
                self._loading_defaults = True
                updates = []
                if "accel_threshold" in settings:
                    updates.append(rclpy.parameter.Parameter("accel_threshold", rclpy.Parameter.Type.DOUBLE, settings["accel_threshold"]))
                if "gyro_threshold" in settings:
                    updates.append(rclpy.parameter.Parameter("gyro_threshold", rclpy.Parameter.Type.DOUBLE, settings["gyro_threshold"]))
                    
                if updates:
                    self.set_parameters(updates)
                    self.get_logger().info(f"Loaded settings from file: {settings}")
                self._loading_defaults = False

            for key, value in raw.items():
                if key == "settings":
                    templates["settings"] = value
                    continue
                canonical = PUNCH_TYPE_ALIASES.get(key, key)
                templates[canonical] = value
            return templates
        except Exception as e:
            self.get_logger().error(f"Error loading templates: {e}")
            return {}

    def _save_templates(self) -> None:
        path = self.get_parameter("calibration_path").value
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._templates, f, indent=2)

    def _on_calibrate(self, request, response):
        punch_type = PUNCH_TYPE_ALIASES.get(request.punch_type, request.punch_type)
        duration_s = float(request.duration_s)
        
        if punch_type not in PUNCH_TYPES:
            response.accepted = False
            response.message = f"Unknown punch_type: {punch_type}"
            return response
            
        # Reset baseline if strictly new type
        if punch_type != self._last_calib_type:
            self._first_strike_accel = None
            self._last_calib_type = punch_type
            self.get_logger().info(f"New punch type {punch_type}: Resetting baseline.")
            
        self._calibrating = punch_type
        self._calibration_peaks[punch_type] = (0.0, 0.0)
        
        if duration_s < 0:
            self._waiting_for_quiet = True
            self._waiting_for_trigger = False
            self._calibration_end = 0.0 # Will be set on trigger
            response.message = f"Stabilizing... then {punch_type}..."
        else:
            self._waiting_for_trigger = False
            self._calibration_end = time.time() + max(0.5, duration_s)
            response.message = f"Calibrating {punch_type} for {duration_s:.1f}s"
            
        response.accepted = True
        self.get_logger().info(response.message)
        return response

    def _calibration_tick(self) -> None:
        if self._calibrating is None or self._waiting_for_trigger or self._waiting_for_quiet:
            return
            
        if time.time() < self._calibration_end:
            return
            
        punch_type = self._calibrating
        peak_accel, peak_gyro = self._calibration_peaks.get(punch_type, (0.0, 0.0))
        
        # Capture baseline if this is the first strike of the sequence
        if self._first_strike_accel is None:
             self._first_strike_accel = max(15.0, peak_accel)
             self.get_logger().info(f"Baseline set to {self._first_strike_accel:.1f}")
        
        self._templates[punch_type] = {"peak_accel": peak_accel, "peak_gyro": peak_gyro}
        self._save_templates()
        
        # Publish completion message for GUI
        out = ImuPunch()
        out.stamp = self.get_clock().now().to_msg()
        out.glove = str(self.get_parameter("imu_hand").value)
        out.punch_type = punch_type
        out.peak_accel = peak_accel
        out.peak_gyro = peak_gyro
        out.confidence = 1.0
        out.method = "calibration_complete"
        self.pub.publish(out)
        
        self.get_logger().info(f"Saved calibration for {punch_type}: accel={peak_accel:.2f} gyro={peak_gyro:.2f}")
        self._calibrating = None

    def _on_imu(self, msg: Imu) -> None:
        # Gravity Calibration
        ax_raw = msg.linear_acceleration.x
        ay_raw = msg.linear_acceleration.y
        az_raw = msg.linear_acceleration.z
        gx_raw = msg.angular_velocity.x
        gy_raw = msg.angular_velocity.y
        gz_raw = msg.angular_velocity.z

        if not self._calibrated_gravity:
            if len(self._gravity_samples) < 50:
                self._gravity_samples.append([ax_raw, ay_raw, az_raw])
                if len(self._gravity_samples) % 10 == 0:
                    self.get_logger().info(f"Calibrating gravity... {len(self._gravity_samples)}/50")
                return
            else:
                self._gravity_vector = np.mean(self._gravity_samples, axis=0)
                self._calibrated_gravity = True
                norm = np.linalg.norm(self._gravity_vector)
                self.get_logger().info(f"Gravity calibrated: {self._gravity_vector} (norm={norm:.2f})")

        if self._calibrating:
            # Motion vector (Acceleration - Gravity)
            acc_vec = np.array([ax_raw, ay_raw, az_raw])
            motion_vec = acc_vec - self._gravity_vector
            
            # Components relative to gravity (parallel/perpendicular?)
            # Ideally just magnitude of motion for "Motion Thresh"
            
            # Use raw magnitudes for "peak" tracking still? 
            # Existing code tracked absolute axis values.
            # To preserve existing logic structure but fix gravity:
            
            # ay_motion "was" the gravity axis. Now we calculate motion magnitude generally.
            # We can project motion onto the gravity vector to simulate "vertical" vs "horizontal"?
            # For simplicity, let's treat "ay" (in code logic) as the MAGNITUDE of linear motion 
            # (since users treat ay as the main punch axis in this codebase usually).
            
            motion_mag = float(np.linalg.norm(motion_vec))
            
            # Absolute values for thresholding
            ax = abs(ax_raw) # Raw tracking still useful?
            ay = motion_mag  # Use Motion Magnitude as primary 'ay' equivalent for threshold
            az = abs(az_raw)
            gx = abs(gx_raw)
            gy = abs(gy_raw)
            gz = abs(gz_raw)
            
            # Peak axis values
            peak_a = motion_mag # Max motion
            peak_g = max(gx, gy, gz)
            
            if self._waiting_for_quiet:
                # Wait for sensor to be relatively still
                # ay is already gravity-corrected from above
                quiet_thresh = 2.0  # Low motion
                gyro_quiet = 0.5
                
                # Log periodically to debug
                if not hasattr(self, '_quiet_log_count'):
                    self._quiet_log_count = 0
                self._quiet_log_count += 1
                if self._quiet_log_count % 50 == 0:
                    self.get_logger().info(f"Waiting quiet: ax={ax:.1f} ay={ay:.1f} az={az:.1f} g={peak_g:.1f} (need <{quiet_thresh})")
                
                if ax < quiet_thresh and ay < quiet_thresh and az < quiet_thresh and peak_g < gyro_quiet:
                    self._waiting_for_quiet = False
                    self._waiting_for_trigger = True
                    self._prev_ay = ay_raw
                    self._quiet_log_count = 0
                    self.get_logger().info("Sensor stable. PUNCH NOW!")
                return

            if self._waiting_for_trigger:
                # ay is already gravity-corrected from above
                # Jerk: sudden change in acceleration
                jerk = abs(ay_raw - getattr(self, '_prev_ay', ay_raw))
                self._prev_ay = ay_raw
                
                # Use sensitivity params from GUI
                accel_thresh = float(self.get_parameter("accel_threshold").value)
                gyro_thresh = float(self.get_parameter("gyro_threshold").value)
                
                # Reasonable thresholds
                motion_thresh = max(3.0, accel_thresh / 4.0)
                jerk_thresh = max(4.0, accel_thresh / 3.0)
                g_thresh = max(0.8, gyro_thresh / 3.0)
                
                # Log what we're seeing
                if not hasattr(self, '_trigger_log_count'):
                    self._trigger_log_count = 0
                self._trigger_log_count += 1
                if self._trigger_log_count % 20 == 0:
                    self.get_logger().info(f"Waiting: peak_a={peak_a:.2f}>{motion_thresh:.2f}? jerk={jerk:.2f}>{jerk_thresh:.2f}? g={peak_g:.2f}>{g_thresh:.2f}?")
                
                # Trigger on significant motion, jerk, or rotation
                triggered = False
                reason = ""
                if peak_a > motion_thresh:
                    triggered = True
                    reason = f"motion={peak_a:.1f}>{motion_thresh:.1f}"
                elif jerk > jerk_thresh:
                    triggered = True
                    reason = f"jerk={jerk:.1f}>{jerk_thresh:.1f}"
                elif peak_g > g_thresh:
                    triggered = True
                    reason = f"gyro={peak_g:.1f}>{g_thresh:.1f}"
                
                if triggered:
                    self._waiting_for_trigger = False
                    self._calibration_end = time.time() + 0.8
                    self.get_logger().info(f"Triggered: {reason} (raw_ay={ay_raw:.1f})")
                else:
                    return

            # Track peak axis values during calibration window
            peak_accel, peak_gyro = self._calibration_peaks.get(self._calibrating, (0.0, 0.0))
            self._calibration_peaks[self._calibrating] = (max(peak_accel, peak_a), max(peak_gyro, peak_g))
            return

        if not self.get_parameter("enable_punch_classification").value:
            return

        self._history.append(msg)
        if len(self._history) < self._history.maxlen:
            return

        now = time.time()
        if now - self._last_time < float(self.get_parameter("cooldown_s").value):
            return

        gyro_thresh = float(self.get_parameter("gyro_threshold").value)
        accel_thresh = float(self.get_parameter("accel_threshold").value)

        peaks = self._window_peaks()
        rms = self._window_rms()
        axis_peaks = self._axis_peaks()
        if not self._passes_filters(peaks, rms, axis_peaks):
            return

        (ax, ay, az), (gx, gy, gz) = axis_peaks
        
        # Debug: log classification inputs periodically
        if not hasattr(self, '_class_log_count'):
            self._class_log_count = 0
        self._class_log_count += 1
        if self._class_log_count % 10 == 0:
            self.get_logger().debug(f"Classify: ax={ax:.1f} ay={ay:.1f} az={az:.1f} | gx={gx:.1f} gy={gy:.1f} gz={gz:.1f} | thresh a={accel_thresh:.1f} g={gyro_thresh:.1f}")

        punch_type = self._classify(gyro_thresh, accel_thresh, axis_peaks)
        if punch_type is None:
            return

        self._last_time = now
        confidence = self._estimate_confidence(punch_type, peaks)
        
        # Log detection
        self.get_logger().info(f"PUNCH DETECTED: {punch_type} (accel={peaks[0]:.1f}, gyro={peaks[1]:.1f}, conf={confidence:.2f})")

        out = ImuPunch()
        out.stamp = self.get_clock().now().to_msg()
        out.glove = str(self.get_parameter("imu_hand").value)
        out.punch_type = punch_type
        out.peak_accel = peaks[0]
        out.peak_gyro = peaks[1]
        out.confidence = confidence
        out.method = "heuristic"
        self.pub.publish(out)

    def _window_peaks(self) -> Tuple[float, float]:
        peak_accel = 0.0
        peak_gyro = 0.0
        for h in self._history:
            peak_accel = max(peak_accel, self._accel_magnitude(h))
            peak_gyro = max(peak_gyro, self._gyro_magnitude(h))
        return peak_accel, peak_gyro

    def _classify(
        self,
        gyro_thresh: float,
        accel_thresh: float,
        axis_peaks: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> Optional[str]:
        (ax, ay, az), (gx, gy, gz) = axis_peaks

        # SIMPLIFIED: Just detect if ANY axis exceeded thresholds significantly
        # We rely on visual inference to classify the TYPE of punch (jab/cross/hook).
        # IMU just provides the precise timing and impact force.
        
        # Check against thresholds (already filtered by _passes_filters logic generally, 
        # but explicit check here ensures valid magnitude).
        max_a = max(ax, ay, az)
        max_g = max(gx, gy, gz)
        
        if max_a > accel_thresh or max_g > gyro_thresh:
             return "straight" # Generic label for "strike"
             
        return None

    def _estimate_confidence(self, punch_type: str, peaks: Tuple[float, float]) -> float:
        if not self.get_parameter("use_calibration").value:
            return 0.6
        template = self._templates.get(punch_type)
        if not template:
            return 0.6
        accel_ratio = peaks[0] / max(0.01, template.get("peak_accel", 1.0))
        gyro_ratio = peaks[1] / max(0.01, template.get("peak_gyro", 1.0))
        return float(min(1.0, 0.5 * accel_ratio + 0.5 * gyro_ratio))

    def _window_rms(self) -> Tuple[float, float]:
        if not self._history:
            return 0.0, 0.0
        accel_sum = 0.0
        gyro_sum = 0.0
        for h in self._history:
            accel = self._accel_magnitude(h)
            gyro = self._gyro_magnitude(h)
            accel_sum += accel * accel
            gyro_sum += gyro * gyro
        n = float(len(self._history))
        return (accel_sum / n) ** 0.5, (gyro_sum / n) ** 0.5

    def _axis_peaks(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        ax = max(abs(h.linear_acceleration.x) for h in self._history)
        ay = max(abs(h.linear_acceleration.y) for h in self._history)
        az = max(abs(h.linear_acceleration.z) for h in self._history)
        gx = max(abs(h.angular_velocity.x) for h in self._history)
        gy = max(abs(h.angular_velocity.y) for h in self._history)
        gz = max(abs(h.angular_velocity.z) for h in self._history)
        return (ax, ay, az), (gx, gy, gz)

    def _passes_filters(
        self,
        peaks: Tuple[float, float],
        rms: Tuple[float, float],
        axis_peaks: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> bool:
        accel_peak, gyro_peak = peaks
        accel_rms, gyro_rms = rms
        accel_ratio = accel_peak / max(0.01, accel_rms)
        gyro_ratio = gyro_peak / max(0.01, gyro_rms)

        accel_ratio_thresh = float(self.get_parameter("accel_peak_ratio").value)
        gyro_ratio_thresh = float(self.get_parameter("gyro_peak_ratio").value)
        
        # Debug: log filter values periodically
        if not hasattr(self, '_filter_log_count'):
            self._filter_log_count = 0
        self._filter_log_count += 1
        if self._filter_log_count % 30 == 0:
            self.get_logger().debug(f"Filter: a_ratio={accel_ratio:.2f}>{accel_ratio_thresh:.1f}? g_ratio={gyro_ratio:.2f}>{gyro_ratio_thresh:.1f}?")
        
        if accel_ratio < accel_ratio_thresh or gyro_ratio < gyro_ratio_thresh:
            return False

        (ax, ay, az), _ = axis_peaks
        axis_sorted = sorted([ax, ay, az], reverse=True)
        dominance = axis_sorted[0] / max(0.01, axis_sorted[1])
        dominance_thresh = float(self.get_parameter("axis_dominance_ratio").value)
        if dominance < dominance_thresh:
            return False

        return True

    @staticmethod
    def _accel_magnitude(msg: Imu) -> float:
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        return float((ax * ax + ay * ay + az * az) ** 0.5)

    @staticmethod
    def _gyro_magnitude(msg: Imu) -> float:
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        return float((gx * gx + gy * gy + gz * gz) ** 0.5)


def main() -> None:
    rclpy.init()
    node = ImuPunchClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Already shut down


if __name__ == "__main__":
    main()
